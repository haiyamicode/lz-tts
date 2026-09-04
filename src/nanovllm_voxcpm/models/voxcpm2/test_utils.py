"""Tests for the char-splitting tokenizer wrapper used by the VoxCPM2 engine."""

import unittest
from pathlib import Path

from transformers import LlamaTokenizerFast

from src.nanovllm_voxcpm.models.voxcpm2.utils import mask_multichar_chinese_tokens

MODEL_DIR = (
    Path(__file__).resolve().parents[4] / "data" / "voxcpm2-stable"
)

MIXED_TEXT = (
    '它之所以看起来"弱势"，仅仅是因为 gah-5.3-Flash '
    "本身的单位 Token 成本已经低得惊人"
)


class CharTokenizerWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = LlamaTokenizerFast.from_pretrained(str(MODEL_DIR))
        cls.wrapper = mask_multichar_chinese_tokens(cls.base)

    def test_offsets_are_aligned_with_char_split_tokens(self):
        offsets = self.wrapper.tokenize_with_offsets(MIXED_TEXT)
        self.assertEqual(len(offsets), len(self.wrapper(MIXED_TEXT)))

    def test_cjk_span_maps_to_single_char_tokens(self):
        offsets = self.wrapper.tokenize_with_offsets(MIXED_TEXT)
        tokens = self.wrapper.tokenize(MIXED_TEXT)
        start, end = MIXED_TEXT.index("弱势"), MIXED_TEXT.index("弱势") + 2
        gated = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > start and token_start < end
        ]
        self.assertEqual([tokens[index] for index in gated], ["弱", "势"])

    def test_latin_span_maps_to_base_bpe_tokens(self):
        offsets = self.wrapper.tokenize_with_offsets(MIXED_TEXT)
        tokens = self.wrapper.tokenize(MIXED_TEXT)
        start, end = MIXED_TEXT.index("gah"), MIXED_TEXT.index("gah") + 3
        gated = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > start and token_start < end
        ]
        self.assertEqual([tokens[index] for index in gated], ["▁g", "ah"])

    def test_split_offsets_cover_their_own_cjk_characters(self):
        for text in ("因为 弱势 所以", "  弱势  ", "弱势", "abc 弱势 def 强弱"):
            offsets = self.wrapper.tokenize_with_offsets(text)
            self.assertEqual(len(offsets), len(self.wrapper(text)), text)
            for token_start, token_end in offsets:
                if token_end - token_start == 1:
                    self.assertTrue(
                        "\u4e00" <= text[token_start] <= "\u9fff"
                        or text[token_start].isspace(),
                        (text, offsets),
                    )

    def test_english_tokenization_is_unchanged_by_wrapper(self):
        for text in ("hello world", "GLM-5.3-Flash", "Q&A: $100,000 — Day 4"):
            self.assertEqual(
                self.wrapper(text),
                self.base(text, add_special_tokens=False)["input_ids"],
                text,
            )

    def test_wrapper_splits_multichar_cjk_tokens(self):
        offsets = self.wrapper.tokenize_with_offsets("因为 弱势 所以")
        tokens = self.wrapper.tokenize("因为 弱势 所以")
        self.assertIn("弱", tokens)
        self.assertIn("势", tokens)
        self.assertNotIn("弱势", tokens)
        self.assertEqual(len(offsets), len(tokens))


if __name__ == "__main__":
    unittest.main()
