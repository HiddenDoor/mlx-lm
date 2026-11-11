# Copyright © 2025 Apple Inc.

import os
import tempfile
import types
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models import llama
from mlx_lm.tuner.trainer import (
    check_output_layer,
    chunked_loss,
    default_loss,
    iterate_batches,
    loss_mask,
    raise_on_long_examples,
)
from mlx_lm.tuner.utils import linear_to_lora_layers


class MockDistributedGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class TestTunerTrainer(unittest.TestCase):
    def test_iterate_batches_ddp(self):
        group = MockDistributedGroup(0, 1)

        def run(rank, size, batch):
            group._rank = rank
            group._size = size

            data = mx.arange(128).reshape(-1, 1).tolist()
            data = [(d, 0) for d in data]

            samples = set()
            for i, (b, l) in enumerate(
                iterate_batches(data, batch, 1, comm_group=group)
            ):
                samples.add(tuple(mx.flatten(b).tolist()))

            ref_batches = mx.arange(128).reshape(-1, batch).tolist()
            for b in ref_batches:
                self.assertTrue(tuple(b[rank::size]) in samples)

        run(0, 1, 4)
        run(0, 1, 8)
        run(0, 2, 8)
        run(1, 2, 8)
        run(0, 4, 8)
        run(1, 4, 8)
        run(2, 4, 8)
        run(3, 4, 8)


def tiny_llama(tie_word_embeddings=True, lora=True):
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
        tie_word_embeddings=tie_word_embeddings,
    )
    mx.random.seed(0)
    model = llama.Model(args)
    if lora:
        model.freeze()
        linear_to_lora_layers(model, 2, {"rank": 4, "scale": 1.0, "dropout": 0.0})
        # LoRA's B starts at zero; give it values so every gradient is exercised.
        for layer in model.layers:
            for proj in (layer.self_attn.q_proj, layer.self_attn.v_proj):
                proj.lora_b = mx.random.normal(proj.lora_b.shape) * 0.1
    mx.eval(model.parameters())
    return model


def span_batch():
    """Two sequences with spans, as ChatDataset returns them with train_assistant_messages: all."""
    return [
        (list(range(1, 11)), [(2, 4), (7, 10)]),
        (list(range(20, 26)), [(3, 6)]),
    ]


class TestMaskedTraining(unittest.TestCase):
    def test_mask_marks_exactly_the_span_targets(self):
        data = [span_batch()[0]]
        batch, mask = next(iterate_batches(data, 1, 2048))
        self.assertEqual(mask.dtype, mx.bool_)
        self.assertEqual(mask.shape, (1, batch.shape[1] - 1))
        # Column c predicts the token at position c + 1.
        on = [c + 1 for c in range(mask.shape[1]) if mask[0, c].item()]
        self.assertEqual(on, [2, 3, 7, 8, 9])

    def test_batch_with_no_trained_tokens_raises(self):
        with self.assertRaises(ValueError):
            next(iterate_batches([(list(range(1, 11)), [])], 1, 2048))

    def test_long_example_raises_in_batches_when_truncation_is_off(self):
        with self.assertRaises(ValueError):
            next(iterate_batches([span_batch()[0]], 1, 5, truncate=False))

    def test_long_example_raises_before_training_when_truncation_is_off(self):
        with self.assertRaises(ValueError) as caught:
            raise_on_long_examples(span_batch(), 8)
        self.assertIn("10", str(caught.exception))
        raise_on_long_examples(span_batch(), 10)

    def test_default_loss_with_mask_equals_loss_on_masked_targets(self):
        model = tiny_llama(lora=False)
        batch, mask = next(iterate_batches(span_batch(), 2, 2048))
        loss, ntoks = default_loss(model, batch, mask)
        ce = nn.losses.cross_entropy(model(batch[:, :-1]), batch[:, 1:])
        expected = (ce * mask).sum() / mask.sum()
        self.assertEqual(ntoks.item(), 8)
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)

    def check_chunked_matches(self, tie_word_embeddings):
        model = tiny_llama(tie_word_embeddings=tie_word_embeddings)
        batch, mask = next(iterate_batches(span_batch(), 2, 2048))
        full = nn.value_and_grad(model, default_loss)
        chunked = nn.value_and_grad(
            model, lambda m, b, l: chunked_loss(m, b, l, chunk_size=3)
        )
        (loss, ntoks), grads = full(model, batch, mask)
        (c_loss, c_ntoks), c_grads = chunked(model, batch, mask)
        self.assertEqual(ntoks.item(), c_ntoks.item())
        self.assertAlmostEqual(loss.item(), c_loss.item(), places=4)
        for (name, g), (_, c) in zip(tree_flatten(grads), tree_flatten(c_grads)):
            self.assertTrue(mx.allclose(g, c, atol=1e-4).item(), name)

    def test_chunked_loss_equals_unchunked_loss_with_tied_embeddings(self):
        self.check_chunked_matches(tie_word_embeddings=True)

    def test_chunked_loss_equals_unchunked_loss_with_an_output_layer(self):
        self.check_chunked_matches(tie_word_embeddings=False)

    def test_check_output_layer_raises_when_the_model_rescales_its_logits(self):
        model = tiny_llama(lora=False)
        check_output_layer(model)

        class Scaled(llama.Model):
            def __call__(self, inputs, *args, **kwargs):
                return super().__call__(inputs, *args, **kwargs) * 2.0

        scaled = Scaled(model.args)
        mx.eval(scaled.parameters())
        with self.assertRaises(ValueError):
            check_output_layer(scaled)

    def test_default_path_counts_one_position_past_the_end(self):
        # Records upstream behavior rather than fixing it: with an offset of 0
        # and a length of 3, the mask covers target positions 1, 2 and 3, and
        # position 3 is padding, past the sequence's last token.
        mask = loss_mask(mx.array([[0, 3]]), 32)
        self.assertEqual(mask.sum().item(), 3)


class Processed(list):
    """Examples already tokenized, in the dataset shape train_model wraps in CacheDataset."""

    def process(self, d):
        return d


class TestLoraSettings(unittest.TestCase):
    def args(self, adapter_path, **overrides):
        from mlx_lm import lora

        settings = {**lora.CONFIG_DEFAULTS, "adapter_path": adapter_path, "iters": 1}
        settings.update(overrides)
        return types.SimpleNamespace(**settings)

    def test_train_model_refuses_long_examples_before_training(self):
        from mlx_lm import lora

        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(
                tmp,
                batch_size=1,
                max_seq_length=5,
                truncate_long_sequences=False,
                num_layers=2,
            )
            with self.assertRaisesRegex(ValueError, "longer than max_seq_length"):
                lora.train_model(
                    args, tiny_llama(lora=False), Processed(span_batch()), Processed()
                )
            self.assertFalse(os.path.exists(os.path.join(tmp, "adapters.safetensors")))

    def test_train_model_trains_with_spans_and_chunked_loss(self):
        from mlx_lm import lora

        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(
                tmp,
                batch_size=1,
                num_layers=2,
                loss_chunk_size=3,
                truncate_long_sequences=False,
                save_every=1,
            )
            lora.train_model(
                args, tiny_llama(lora=False), Processed(span_batch()), Processed()
            )
            self.assertTrue(os.path.exists(os.path.join(tmp, "adapters.safetensors")))

    def test_train_model_with_chunked_loss_checks_the_output_layer(self):
        from mlx_lm import lora

        class Scaled(llama.Model):
            def __call__(self, inputs, *args, **kwargs):
                return super().__call__(inputs, *args, **kwargs) * 2.0

        model = Scaled(tiny_llama(lora=False).args)
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(tmp, batch_size=1, num_layers=2, loss_chunk_size=3)
            with self.assertRaisesRegex(ValueError, "changes its logits"):
                lora.train_model(args, model, Processed(span_batch()), Processed())


if __name__ == "__main__":
    unittest.main()
