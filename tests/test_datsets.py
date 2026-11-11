# Copyright © 2024 Apple Inc.

import json
import os
import tempfile
import types
import unittest

from transformers import AutoTokenizer

from mlx_lm.tuner import datasets

HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


class TestDatasets(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        if not os.path.isdir(cls.test_dir):
            os.mkdir(cls.test_dir_fid.name)
        # Only one HF request
        AutoTokenizer.from_pretrained(HF_MODEL_PATH)

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def save_data(self, data):
        for ds in ["train", "valid"]:
            with open(os.path.join(self.test_dir, f"{ds}.jsonl"), "w") as fid:
                for l in data:
                    json.dump(l, fid)
                    fid.write("\n")

    def test_text(self):
        data = {"text": "This is an example for the model."}
        self.save_data(4 * [data])
        args = types.SimpleNamespace(train=True, test=False, data=self.test_dir)
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH, local_files_only=True)
        train, valid, test = datasets.load_dataset(args, tokenizer)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(valid), 4)
        self.assertEqual(len(test), 0)
        self.assertTrue(len(train[0]) > 0)
        self.assertTrue(len(valid[0]) > 0)
        self.assertTrue(isinstance(train, datasets.TextDataset))

    def test_completions(self):
        data = {"prompt": "What is the capital of France?", "completion": "Paris."}
        self.save_data(4 * [data])
        args = types.SimpleNamespace(train=True, test=False, data=self.test_dir)
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH, local_files_only=True)
        train, valid, test = datasets.load_dataset(args, tokenizer)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(valid), 4)
        self.assertEqual(len(test), 0)
        self.assertTrue(len(train[0]) > 0)
        self.assertTrue(len(valid[0]) > 0)
        self.assertTrue(isinstance(train, datasets.CompletionsDataset))

    def test_completions_mask_prompt(self):
        data = {"prompt": "What is the capital of France?", "completion": "Paris."}
        self.save_data(4 * [data])
        args = types.SimpleNamespace(
            train=True, test=False, data=self.test_dir, mask_prompt=True
        )
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH, local_files_only=True)
        train, valid, test = datasets.load_dataset(args, tokenizer)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(valid), 4)
        self.assertEqual(len(test), 0)
        expected_prompt_tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": data["prompt"]}],
            add_generation_prompt=True,
            return_dict=False,
        )
        expected_offset = len(expected_prompt_tokens)

        train_tokens, train_offset = train.process(train[0])
        valid_tokens, valid_offset = valid.process(valid[0])

        self.assertTrue(len(train_tokens) > 0)
        self.assertTrue(len(valid_tokens) > 0)
        self.assertEqual(train_offset, expected_offset)
        self.assertEqual(valid_offset, expected_offset)
        self.assertLess(train_offset, len(train_tokens))
        self.assertLess(valid_offset, len(valid_tokens))
        self.assertEqual(train_tokens[:train_offset], expected_prompt_tokens)
        self.assertEqual(valid_tokens[:valid_offset], expected_prompt_tokens)
        self.assertTrue(isinstance(train, datasets.CompletionsDataset))

    def test_chat(self):
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "How can I assistant you today."},
            ]
        }
        self.save_data(4 * [data])
        args = types.SimpleNamespace(train=True, test=False, data=self.test_dir)
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH, local_files_only=True)
        train, valid, test = datasets.load_dataset(args, tokenizer)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(valid), 4)
        self.assertEqual(len(test), 0)
        self.assertTrue(len(train[0]) > 0)
        self.assertTrue(len(valid[0]) > 0)
        self.assertTrue(isinstance(train, datasets.ChatDataset))

    def test_hf(self):
        hf_args = {
            "path": "billsum",
            "prompt_feature": "text",
            "completion_feature": "summary",
            "train_split": "train[:2%]",
            "valid_split": "train[-2%:]",
        }
        args = types.SimpleNamespace(
            hf_dataset=hf_args,
            test=False,
            train=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH, local_files_only=True)
        train, valid, test = datasets.load_dataset(args, tokenizer)
        self.assertTrue(len(train) > 0)
        self.assertTrue(len(train[0]) > 0)
        self.assertTrue(len(valid) > 0)
        self.assertTrue(len(valid[0]) > 0)
        self.assertEqual(len(test), 0)

        args = types.SimpleNamespace(
            hf_dataset=[hf_args, hf_args],
            test=False,
            train=True,
        )
        train_double, valid_double, test_double = datasets.load_dataset(args, tokenizer)
        self.assertEqual(2 * len(train), len(train_double))
        self.assertEqual(2 * len(valid), len(valid_double))
        self.assertEqual(2 * len(test), len(test_double))


class TemplateTokenizer:
    """A stand-in chat template: each message is [role id, one id per character, end id], and the generation prompt
    is the assistant role id. With `rewrite_history`, earlier assistant messages render empty, as some templates do
    with earlier reasoning."""

    ROLES = {"system": 1, "user": 2, "assistant": 3, "tool": 4}
    END = 9

    def __init__(self, rewrite_history=False):
        self.rewrite_history = rewrite_history

    def content_ids(self, text):
        return [10 + ord(c) % 50 for c in text]

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=False, return_dict=False
    ):
        ids = []
        for i, m in enumerate(messages):
            content = self.content_ids(m.get("content") or "")
            if (
                self.rewrite_history
                and m["role"] == "assistant"
                and i < len(messages) - 1
            ):
                content = []
            ids += [self.ROLES[m["role"]], *content, self.END]
        if add_generation_prompt:
            ids.append(self.ROLES["assistant"])
        return ids


def _conversation(first_weight=None):
    first = {"role": "assistant", "content": "ab"}
    if first_weight is not None:
        first["weight"] = first_weight
    return {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            first,
            {"role": "tool", "content": "x"},
            {"role": "assistant", "content": "cd"},
        ]
    }


class TestAssistantSpans(unittest.TestCase):
    def dataset(self, tokenizer, **kwargs):
        config = types.SimpleNamespace(
            mask_prompt=True, train_assistant_messages="all", **kwargs
        )
        return datasets.create_dataset([_conversation()], tokenizer, config)

    def test_all_assistant_messages_get_spans(self):
        tok = TemplateTokenizer()
        ds = self.dataset(tok)
        tokens, spans = ds.process(_conversation())
        self.assertEqual(len(spans), 2)
        for (start, end), text in zip(spans, ["ab", "cd"]):
            self.assertEqual(tokens[start:end], tok.content_ids(text) + [tok.END])

    def test_weight_zero_message_gets_no_span(self):
        tok = TemplateTokenizer()
        tokens, spans = self.dataset(tok).process(_conversation(first_weight=0))
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        self.assertEqual(tokens[start:end], tok.content_ids("cd") + [tok.END])

    def test_spans_raise_when_template_rewrites_earlier_replies(self):
        ds = self.dataset(TemplateTokenizer(rewrite_history=True))
        with self.assertRaises(ValueError):
            ds.process(_conversation())

    def test_last_mode_still_returns_an_offset(self):
        config = types.SimpleNamespace(mask_prompt=True)
        ds = datasets.create_dataset([_conversation()], TemplateTokenizer(), config)
        tokens, offset = ds.process(_conversation())
        self.assertIsInstance(offset, int)

    def test_all_mode_needs_mask_prompt(self):
        config = types.SimpleNamespace(train_assistant_messages="all")
        with self.assertRaises(ValueError):
            datasets.create_dataset([_conversation()], TemplateTokenizer(), config)


if __name__ == "__main__":
    unittest.main()
