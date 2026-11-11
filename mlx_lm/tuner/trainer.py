# Copyright © 2024 Apple Inc.


import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map
from tqdm import tqdm

from .callbacks import TrainingCallback
from .datasets import CacheDataset


def _clear_cache(threshold: int):
    if mx.get_cache_memory() > threshold:
        mx.clear_cache()


def grad_checkpoint(layer):
    """
    Update all instances of type(layer) to use gradient checkpointing.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn


@dataclass
class TrainingArgs:
    batch_size: int = field(default=4, metadata={"help": "Minibatch size."})
    iters: int = field(default=100, metadata={"help": "Iterations to train for."})
    val_batches: int = field(
        default=25,
        metadata={
            "help": "Number of validation batches, -1 uses the entire validation set."
        },
    )
    steps_per_report: int = field(
        default=10,
        metadata={"help": "Number of training steps between loss reporting."},
    )
    steps_per_eval: int = field(
        default=200, metadata={"help": "Number of training steps between validations."}
    )
    steps_per_save: int = field(
        default=100, metadata={"help": "Save the model every number steps"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length."}
    )
    adapter_file: str = field(
        default="adapters.safetensors",
        metadata={"help": "Save/load path for the trained adapter weights."},
    )
    grad_checkpoint: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing to reduce memory use."},
    )
    grad_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of steps to accumulate gradients before applying an optimizer update."
        },
    )
    clear_cache_threshold: int = field(
        default=0,
        metadata={
            "help": "Clear the allocator cache between steps if it grows too large."
        },
    )


def loss_mask(lengths, num_targets):
    """
    Which target positions count toward the loss, as a [batch, targets]
    boolean array. ``lengths`` is either that mask already (from datasets that
    return token spans) or one [offset, length] row per sequence.
    """
    if lengths.dtype == mx.bool_:
        return lengths
    steps = mx.arange(1, num_targets + 1)
    return mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])


def default_loss(model, batch, lengths):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(inputs)

    mask = loss_mask(lengths, targets.shape[1])

    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    ce = ce.astype(mx.float32).sum() / ntoks

    return ce, ntoks


def _output_layer(model):
    """
    The model's body and the module that holds its output layer, with the call
    that applies it. Supports models with ``model.model`` and either an
    ``lm_head`` or tied input embeddings.
    """
    body = getattr(model, "model", None)
    if body is None:
        raise ValueError("loss_chunk_size needs a model with a `model` attribute.")
    head = getattr(model, "lm_head", None)
    if head is not None:
        return body, head, head
    embed = getattr(body, "embed_tokens", None)
    if embed is None:
        raise ValueError(
            "loss_chunk_size needs a model with `lm_head` or `model.embed_tokens`."
        )
    return body, embed, embed.as_linear


def check_output_layer(model):
    """
    Raise unless the model's body followed by its output layer gives the same
    logits as the whole model. Models that rescale or cap their logits after
    the output layer would otherwise be trained on the wrong loss.
    """
    body, _, head = _output_layer(model)
    inputs = mx.array([[0, 1, 2, 3]])
    whole = model(inputs).astype(mx.float32)
    split = head(body(inputs)).astype(mx.float32)
    if not mx.allclose(whole, split, atol=1e-3, rtol=1e-3).item():
        raise ValueError(
            "This model changes its logits after the output layer, so "
            "loss_chunk_size can't be used with it."
        )


def chunked_loss(model, batch, lengths, chunk_size):
    """
    The same loss as ``default_loss``, computed with the output layer and
    cross entropy applied ``chunk_size`` positions at a time under
    ``mx.checkpoint``, so only one chunk's logits exist at a time. For long
    sequences the full logits (positions x vocabulary) can take more memory
    than the model. Check the model first with ``check_output_layer``.
    """
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    mask = loss_mask(lengths, targets.shape[1])
    body, owner, head = _output_layer(model)
    hidden = body(inputs)

    def chunk_ce(params, hidden, targets, mask):
        owner.update(params)
        ce = nn.losses.cross_entropy(head(hidden), targets) * mask
        return ce.astype(mx.float32).sum()

    chunk_ce = mx.checkpoint(chunk_ce)
    total = mx.array(0.0)
    for start in range(0, targets.shape[1], chunk_size):
        end = start + chunk_size
        total = total + chunk_ce(
            owner.trainable_parameters(),
            hidden[:, start:end],
            targets[:, start:end],
            mask[:, start:end],
        )
    ntoks = mask.sum()
    return total / ntoks, ntoks


def raise_on_long_examples(dataset, max_seq_length):
    """
    Raise if any example is longer than ``max_seq_length``, before training
    starts rather than when its batch comes up.
    """
    lengths = [len(dataset[i][0]) for i in range(len(dataset))]
    over = [n for n in lengths if n > max_seq_length]
    if over:
        raise ValueError(
            f"{len(over)} of {len(lengths)} examples are longer than "
            f"max_seq_length={max_seq_length} (the longest has {max(over)} "
            "tokens). Raise max_seq_length, or set truncate_long_sequences: true "
            "to cut them."
        )


def _span_mask(spans, lengths, width):
    """A [batch, width - 1] mask of the target positions inside each sequence's spans."""
    mask = np.zeros((len(spans), width - 1), dtype=bool)
    for row, (sequence_spans, length) in enumerate(zip(spans, lengths)):
        for start, end in sequence_spans:
            start, end = max(start, 1), min(end, length)
            if start < end:
                mask[row, start - 1 : end - 1] = True
    if not mask.any():
        raise ValueError("A batch has no tokens to train on.")
    return mx.array(mask)


def iterate_batches(
    dataset,
    batch_size,
    max_seq_length,
    loop=False,
    seed=None,
    comm_group=None,
    truncate=True,
):
    # Sort by length:
    if isinstance(dataset, CacheDataset):
        len_fn = lambda idx: dataset.itemlen(idx)
    else:
        len_fn = lambda idx: len(dataset[idx][0])
    idx = sorted(range(len(dataset)), key=len_fn)
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size}"
            f" examples but only has {len(dataset)}."
        )

    # If running in distributed mode (N machines) then each one should skip N-1
    # samples
    if comm_group is not None:
        offset = comm_group.rank()
        step = comm_group.size()
    else:
        offset = 0
        step = 1
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    # Make the batches:
    batch_idx = [
        idx[i + offset : i + offset + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]
    if seed:
        np.random.seed(seed)
    while True:
        indices = np.random.permutation(len(batch_idx))
        for i in indices:
            batch = [dataset[j] for j in batch_idx[i]]
            if len(batch[0]) == 2:
                batch, offsets = zip(*batch)
            else:
                offsets = [0] * len(batch)
            lengths = [len(x) for x in batch]
            if max(lengths) > max_seq_length and not truncate:
                raise ValueError(
                    f"An example has {max(lengths)} tokens, more than "
                    f"max_seq_length={max_seq_length}."
                )
            if max(lengths) > max_seq_length:
                print(
                    f"[WARNING] Some sequences are longer than {max_seq_length} tokens. "
                    f"The longest sentence {max(lengths)} will be truncated to {max_seq_length}. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to one plus nearest multiple of pad_to or the maximum length
            pad_to = 32
            max_length_in_batch = 1 + pad_to * ((max(lengths) + pad_to - 1) // pad_to)
            max_length_in_batch = min(max_length_in_batch, max_seq_length)

            batch_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)

            for j in range(batch_size // step):
                truncated_length = min(lengths[j], max_seq_length)
                batch_arr[j, :truncated_length] = batch[j][:truncated_length]
                lengths[j] = (
                    truncated_length  # Update lengths to match truncated lengths
                )
            batch = mx.array(batch_arr)
            if isinstance(offsets[0], (list, tuple)):
                yield batch, _span_mask(offsets, lengths, batch_arr.shape[1])
            else:
                yield batch, mx.array(list(zip(offsets, lengths)))

        if not loop:
            break


def evaluate(
    model,
    dataset,
    batch_size,
    num_batches,
    max_seq_length=2048,
    loss: callable = default_loss,
    iterate_batches: callable = iterate_batches,
    clear_cache_threshold: int = 0,
):
    model.eval()
    all_losses = mx.array(0.0)
    ntokens = mx.array(0)

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, batch in tqdm(
        zip(
            index_iterator,
            iterate_batches(
                dataset=dataset,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                comm_group=mx.distributed.init(),
            ),
        ),
        desc="Calculating loss...",
        total=min(len(dataset) // batch_size, num_batches),
    ):
        losses, toks = loss(model, *batch)
        all_losses += losses * toks
        ntokens += toks
        mx.eval(all_losses, ntokens)
        _clear_cache(clear_cache_threshold)

    all_losses = mx.distributed.all_sum(all_losses, stream=mx.cpu)
    ntokens = mx.distributed.all_sum(ntokens, stream=mx.cpu)
    avg_loss = (all_losses / ntokens).item()

    return avg_loss


def train(
    model,
    optimizer,
    train_dataset,
    val_dataset=None,
    args: TrainingArgs = TrainingArgs(),
    loss: callable = default_loss,
    iterate_batches: callable = iterate_batches,
    training_callback: TrainingCallback = None,
):
    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    print(f"Starting training..., iters: {args.iters}")
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    loss_value_and_grad = nn.value_and_grad(model, loss)

    grad_accum_steps = args.grad_accumulation_steps
    if grad_accum_steps < 1:
        raise ValueError("grad_accumulation_steps must be at least 1")

    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            grad = average_gradients(grad)
            if grad_accum_steps > 1:
                grad = tree_map(lambda x: x / grad_accum_steps, grad)
            optimizer.update(model, grad)
            grad = None

        return lvalue, toks, grad

    model.train()
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    train_time = 0
    grad_accum = None

    # Main training loop
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            loop=True,
            comm_group=world,
        ),
    ):
        tic = time.perf_counter()
        # Report validation loss if needed, the first validation loss
        # is always measured before any training.
        if val_dataset and (
            it == 1 or it % args.steps_per_eval == 0 or it == args.iters
        ):
            tic = time.perf_counter()
            val_loss = evaluate(
                model=model,
                dataset=val_dataset,
                loss=loss,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                iterate_batches=iterate_batches,
            )
            model.train()
            val_time = time.perf_counter() - tic
            if rank == 0:
                print(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val took {val_time:.3f}s",
                    flush=True,
                )

            if training_callback is not None:
                val_info = {
                    "iteration": it - 1,
                    "val_loss": val_loss,
                    "val_time": val_time,
                }
                training_callback.on_val_loss_report(val_info)

            tic = time.perf_counter()

        lvalue, toks, grad_accum = step(
            batch,
            grad_accum,
            it % grad_accum_steps == 0,
        )

        losses += lvalue
        n_tokens += toks
        steps += 1
        mx.eval(state, losses, n_tokens, grad_accum)
        _clear_cache(args.clear_cache_threshold)
        train_time += time.perf_counter() - tic

        # Report training loss if needed
        if it % args.steps_per_report == 0 or it == args.iters:
            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * world_size
            n_tokens = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / train_time
            tokens_sec = float(n_tokens) / train_time
            trained_tokens += n_tokens
            peak_mem = mx.get_peak_memory() / 1e9
            if rank == 0:
                print(
                    f"Iter {it}: Train loss {train_loss:.3f}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Trained Tokens {trained_tokens}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            n_tokens = 0
            steps = 0
            train_time = 0

        # Save adapter weights
        if it % args.steps_per_save == 0 and rank == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    # Save final weights
    if rank == 0:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(args.adapter_file), adapter_weights)
        print(f"Saved final weights to {args.adapter_file}.")
