"""
Test suite for context_replacer.py

Data-driven tests for contextual text replacement (AI -> ây ai, numbers, etc.)

Run with: python -m pytest src/piper/test_context_replacer.py -v
Or directly: python -m unittest src/piper/test_context_replacer.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src/ is on path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from piper.context_replacer import (
    ContextReplacerClassifier,
    _Match,
    _PatternRule,
    _RuleEntry,
    _find_pattern_matches,
    _get_num2words,
    _icu_word_spans,
    _segment_and_find_matches,
)


# =============================================================================
# ICU SEGMENTATION TESTS
# =============================================================================

class TestIcuSegmentation(unittest.TestCase):
    def test_basic_vietnamese(self):
        spans = _icu_word_spans("hệ thống AI mới ra mắt", "vi_VN")
        words = [piece for _, _, piece in spans]
        self.assertIn("AI", words)
        self.assertIn("mới", words)

    def test_basic_english(self):
        spans = _icu_word_spans("A new AI system", "en_US")
        words = [piece for _, _, piece in spans]
        self.assertIn("AI", words)
        self.assertIn("new", words)

    def test_numbers_separated(self):
        spans = _icu_word_spans("gọi vốn 100 triệu đô", "vi_VN")
        words = [piece for _, _, piece in spans]
        self.assertIn("100", words)
        self.assertIn("triệu", words)

    def test_preserves_offsets(self):
        text = "hệ thống AI mới"
        spans = _icu_word_spans(text, "vi_VN")
        for start, end, piece in spans:
            self.assertEqual(text[start:end], piece)


# =============================================================================
# NUMBER WORDS TESTS (num2words)
# =============================================================================

class TestNum2Words(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from num2words.lang_VI import Num2Word_VI
        cls.converter = Num2Word_VI()

    def _convert(self, s):
        return self.converter.to_cardinal(s)

    def test_integer(self):
        self.assertEqual(self._convert(42), "bốn mươi hai")

    def test_small_number(self):
        self.assertEqual(self._convert(5), "năm")

    def test_large_number(self):
        self.assertEqual(self._convert(1000000), "một triệu")

    def test_thousand_separators_dot(self):
        self.assertEqual(self._convert("1.000.000"), "một triệu")

    def test_thousand_separators_comma(self):
        self.assertEqual(self._convert("1,000,000"), "một triệu")

    def test_thousand_separators_mixed(self):
        self.assertEqual(self._convert("3.500.000"), "ba triệu năm trăm nghìn")

    def test_decimal_two_digits(self):
        self.assertEqual(self._convert(3.14), "ba phẩy mười bốn")

    def test_decimal_one_digit(self):
        self.assertEqual(self._convert(1.5), "một phẩy năm")

    def test_decimal_many_digits(self):
        self.assertEqual(self._convert(3.14159), "ba phẩy một bốn một năm chín")

    def test_decimal_with_comma(self):
        self.assertEqual(self._convert("1,5"), "một phẩy năm")

    def test_zero(self):
        self.assertEqual(self._convert(0), "không")

    def test_single_dot_three_digits_is_decimal(self):
        """1.000 = 1.0 (decimal), NOT 1000 (thousand)"""
        self.assertEqual(self._convert("1.000"), "một")


# =============================================================================
# PATTERN MATCHING TESTS
# =============================================================================

class TestPatternMatching(unittest.TestCase):
    def setUp(self):
        self.percent_rule = _PatternRule(
            pattern=__import__("re").compile(r"%"),
            replacement_template=" phần trăm",
            language="vi-VN",
            always_replace=True,
        )
        self.number_rule = _PatternRule(
            pattern=__import__("re").compile(
                r"\b(\d{1,3}(?:[.,]\d{3}){2,}|\d+[.,]\d+|\d+)\b"
            ),
            replacement_template="$1",
            language="vi-VN",
            always_replace=True,
            transforms={1: _get_num2words("vi")},
        )

    def test_percent_match(self):
        text = "Pin còn 3%"
        matches = _find_pattern_matches(text, 0, [self.percent_rule], "vi-VN")
        self.assertEqual(len(matches), 1)
        self.assertEqual(text[matches[0].start:matches[0].end], "%")
        self.assertEqual(matches[0].rule.replacement, " phần trăm")

    def test_number_match(self):
        matches = _find_pattern_matches("Có 42 người", 0, [self.number_rule], "vi-VN")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].rule.replacement, "bốn mươi hai")

    def test_number_with_thousand_sep(self):
        matches = _find_pattern_matches("gọi vốn 1.000.000 đô", 0, [self.number_rule], "vi-VN")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].rule.replacement, "một triệu")

    def test_decimal_number(self):
        matches = _find_pattern_matches("Giá 3.14 đô", 0, [self.number_rule], "vi-VN")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].rule.replacement, "ba phẩy mười bốn")

    def test_language_filter(self):
        matches = _find_pattern_matches("Có 42 người", 0, [self.number_rule], "en-US")
        self.assertEqual(len(matches), 0)

    def test_percent_and_number_together(self):
        rules = [self.percent_rule, self.number_rule]
        matches = _find_pattern_matches("tăng 25%", 0, rules, "vi-VN")
        self.assertEqual(len(matches), 2)
        # number first, then percent (sorted by start)
        self.assertEqual(matches[0].rule.replacement, "hai mươi lăm")
        self.assertEqual(matches[1].rule.replacement, " phần trăm")


# =============================================================================
# TOKEN MATCHING TESTS (ICU + dict lookup)
# =============================================================================

class TestTokenMatching(unittest.TestCase):
    def setUp(self):
        self.lookup = {
            "ai": _RuleEntry(token="AI", replacement="ây ai", language="vi-VN"),
            "ceo": _RuleEntry(token="CEO", replacement="xi i ô", language="vi-VN", always_replace=True),
            "api": _RuleEntry(token="API", replacement="ây pi ai", language="vi-VN", always_replace=True),
        }

    def test_basic_match(self):
        matches = _segment_and_find_matches("hệ thống AI mới", 0, self.lookup, "vi-VN")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word, "AI")
        self.assertEqual(matches[0].rule.replacement, "ây ai")

    def test_case_insensitive(self):
        matches = _segment_and_find_matches("hệ thống ai mới", 0, self.lookup, "vi-VN")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].word, "ai")

    def test_no_match(self):
        matches = _segment_and_find_matches("hôm nay trời đẹp", 0, self.lookup, "vi-VN")
        self.assertEqual(len(matches), 0)

    def test_language_filter(self):
        matches = _segment_and_find_matches("A new AI system", 0, self.lookup, "en-US")
        self.assertEqual(len(matches), 0)

    def test_multiple_matches(self):
        matches = _segment_and_find_matches("CEO dùng API mới", 0, self.lookup, "vi-VN")
        self.assertEqual(len(matches), 2)
        words = [m.word for m in matches]
        self.assertIn("CEO", words)
        self.assertIn("API", words)

    def test_always_replace_flag(self):
        matches = _segment_and_find_matches("dùng API mới", 0, self.lookup, "vi-VN")
        self.assertTrue(matches[0].rule.always_replace)

    def test_word_boundary(self):
        """API should not match inside other words"""
        matches = _segment_and_find_matches("VAPID comment", 0, self.lookup, "vi-VN")
        api_matches = [m for m in matches if m.rule.token == "API"]
        self.assertEqual(len(api_matches), 0)

    def test_offsets_correct(self):
        text = "hệ thống AI mới"
        matches = _segment_and_find_matches(text, 0, self.lookup, "vi-VN")
        self.assertEqual(len(matches), 1)
        self.assertEqual(text[matches[0].start:matches[0].end], "AI")


# =============================================================================
# FULL REPLACEMENT TESTS (requires trained model checkpoint)
# =============================================================================

_CHECKPOINT_PATH = Path(__file__).parent.parent.parent / "data" / "replacements" / "best.pt"
_RULES_PATH = Path(__file__).parent.parent.parent / "data" / "replacements" / "rules.jsonl"


@unittest.skipUnless(_CHECKPOINT_PATH.exists(), "No trained checkpoint found")
class TestContextReplacer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from piper.context_replacer import ContextReplacer
        cls.replacer = ContextReplacer(
            checkpoint_path=_CHECKPOINT_PATH,
            rules_path=_RULES_PATH,
            device="cpu",
        )
        cls.replacer.load()

    # --- AI classifier tests ---

    def test_ai_tech_approved(self):
        result = self.replacer.apply_replacements("hệ thống AI mới ra mắt", "vi-VN")
        self.assertIn("ây ai", result)
        self.assertNotIn("AI", result)

    def test_ai_tech_in_sentence(self):
        result = self.replacer.apply_replacements(
            "tôi dùng AI để phân tích dữ liệu", "vi-VN"
        )
        self.assertIn("ây ai", result)

    def test_ai_vietnamese_pronoun_rejected(self):
        result = self.replacer.apply_replacements("NGƯỜI ẤY LÀ AI?", "vi-VN")
        self.assertNotIn("ây ai", result)
        self.assertIn("AI", result)

    def test_ai_vietnamese_question_rejected(self):
        result = self.replacer.apply_replacements(
            "AI ĐANG GỌI ĐIỆN VẬY?", "vi-VN"
        )
        self.assertNotIn("ây ai", result)

    def test_ai_khong_co_rejected(self):
        result = self.replacer.apply_replacements("KHÔNG CÓ AI Ở NHÀ.", "vi-VN")
        self.assertNotIn("ây ai", result)

    # --- always_replace token tests ---

    def test_ceo_always_replaced(self):
        result = self.replacer.apply_replacements("CEO công ty phát biểu", "vi-VN")
        self.assertIn("xi i ô", result)

    def test_api_always_replaced(self):
        result = self.replacer.apply_replacements("dùng API để kết nối", "vi-VN")
        self.assertIn("ây pi ai", result)

    def test_it_always_replaced(self):
        result = self.replacer.apply_replacements("Phòng IT mới", "vi-VN")
        self.assertIn("ai ti", result)

    # --- Number pattern tests ---

    def test_number_replaced(self):
        result = self.replacer.apply_replacements("Có 42 người", "vi-VN")
        self.assertIn("bốn mươi hai", result)

    def test_thousand_sep_replaced(self):
        result = self.replacer.apply_replacements("gọi vốn 1.000.000 đô", "vi-VN")
        self.assertIn("một triệu", result)

    def test_decimal_replaced(self):
        result = self.replacer.apply_replacements("Giá 3.14 đô", "vi-VN")
        self.assertIn("ba phẩy mười bốn", result)

    # --- Percent pattern tests ---

    def test_percent_replaced(self):
        result = self.replacer.apply_replacements("Pin còn 3%", "vi-VN")
        self.assertIn("ba phần trăm", result)

    # --- Mixed tests ---

    def test_ai_and_percent(self):
        result = self.replacer.apply_replacements(
            "Startup về AI tăng 25%", "vi-VN"
        )
        self.assertIn("ây ai", result)
        self.assertIn("hai mươi lăm phần trăm", result)

    def test_ai_and_number_and_percent(self):
        result = self.replacer.apply_replacements(
            "Startup về AI tăng 25% sau khi nhận 10.000.000 đô.", "vi-VN"
        )
        self.assertIn("ây ai", result)
        self.assertIn("hai mươi lăm phần trăm", result)
        self.assertIn("mười triệu", result)

    # --- Language filter tests ---

    def test_english_no_change(self):
        text = "A new AI system for real estate"
        result = self.replacer.apply_replacements(text, "en-US")
        self.assertEqual(result, text)

    def test_plain_vietnamese_no_change(self):
        text = "Hôm nay trời đẹp quá."
        result = self.replacer.apply_replacements(text, "vi-VN")
        self.assertEqual(result, text)

    # --- Batch tests ---

    def test_batch_resolve(self):
        texts = [
            "hệ thống AI mới",
            "KHÔNG CÓ AI Ở NHÀ.",
            "CEO công ty",
        ]
        results = self.replacer.apply_replacements_many(texts, "vi-VN")
        self.assertEqual(len(results), 3)
        self.assertIn("ây ai", results[0])
        self.assertNotIn("ây ai", results[1])
        self.assertIn("xi i ô", results[2])


if __name__ == "__main__":
    unittest.main()
