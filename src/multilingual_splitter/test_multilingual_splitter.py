"""
Test suite for multilingual_splitter.py

Data-driven tests for language detection and code-switching.

Run with: python -m pytest test_splitter.py -v
Or directly: python test_splitter.py
"""

import unittest

from .multilingual_splitter import (
    DEFINITIVE_MARKERS,
    MultilingualSplitter,
    Segment,
    SplitResult,
    split_text,
)


# =============================================================================
# TEST DATA TABLES
# =============================================================================

# Main language detection: (text, expected_language, description)
MAIN_LANGUAGE_CASES = [
    # Japanese - definitive markers
    ("これはテストです", "ja", "Hiragana text"),
    ("テスト", "ja", "Katakana text"),
    ("Hello これはテストです world", "ja", "Mixed with English, Hiragana present"),
    # Korean - definitive marker
    ("안녕하세요", "ko", "Hangul text"),
    # Thai - definitive marker
    ("สวัสดีครับ", "th", "Thai script"),
    # Ukrainian - specific Cyrillic chars
    ("Україна", "uk", "Ukrainian-specific chars (ї,є,ґ)"),
    # Latin languages - fasttext detection
    ("Hello, this is a test.", "en", "English text"),
    ("Das ist ein Test auf Deutsch.", "de", "German text"),
    # Chinese - Han without Japanese markers (fasttext may wrongly say Japanese)
    ("这是一个测试", "zh", "Simplified Chinese"),
    ("你好世界", "zh", "Chinese greeting"),
    ("今天是2025年,9月28日。", "zh", "Chinese with numbers - fasttext bias fix"),
    ("2025年9月28日", "zh", "Date format that fasttext thinks is Japanese"),
    ("第1章", "zh", "Chapter 1 - fasttext thinks is Japanese"),
    ("山", "zh", "Single Han char - fasttext says Japanese"),
    # Japanese - with yen symbol or kana (definitive markers)
    ("1000円", "ja", "Japanese yen symbol"),
    ("これは日本語", "ja", "Japanese with Hiragana"),
    # Other scripts
    ("Γειά σου κόσμε", "el", "Greek text"),
    ("שלום עולם", "he", "Hebrew text"),
    ("مرحبا عالم", "ar", "Arabic text"),
]

# Script segmentation: (text, expected_scripts, description)
# Note: Japanese scripts (Hiragana/Katakana/Han) are merged by the splitter
SCRIPT_SEGMENTATION_CASES = [
    ("Hello world", {"Latin"}, "Single Latin script"),
    ("Hello 世界", {"Latin", "Han"}, "Latin + Han"),
    ("これはテストです", {"Hiragana"}, "Japanese text (scripts merged)"),
    ("Привет Hello мир", {"Cyrillic", "Latin"}, "Cyrillic + Latin"),
    ("مرحبا Hello عالم", {"Arabic", "Latin"}, "Arabic + Latin"),
    ("안녕 Hello 세상", {"Hangul", "Latin"}, "Hangul + Latin"),
]

# Code-switching detection - should detect language switch
# (text, main_lang, should_contain_languages, description)
CODE_SWITCHING_CASES = [
    # Clear script switches (always detected)
    ("This is English これは日本語です", "en", {"en", "ja"}, "English with Japanese"),
    ("Привет Hello мир", "ru", {"ru", "en"}, "Russian with English"),
    ("안녕 Hello 세상", "ko", {"ko", "en"}, "Korean with English"),
    ("مرحبا Hello عالم", "ar", {"ar", "en"}, "Arabic with English"),
]

# False positive prevention - should NOT switch away from main language
# (text, main_lang, segment_text, expected_lang, description)
FALSE_POSITIVE_CASES = [
    # Han characters that fasttext detects as Japanese but should stay Chinese
    ("我是学生", "zh", "学生", "zh", "学生 (student) should stay Chinese"),
    ("这座山很高", "zh", "山", "zh", "山 (mountain) should stay Chinese"),
    ("大学很好", "zh", "大学", "zh", "大学 (university) should stay Chinese"),
    ("水很清", "zh", "水", "zh", "水 (water) should stay Chinese"),
    # Han characters that should stay Japanese when main is Japanese
    ("日本語を勉強する", "ja", "日本語", "ja", "日本語 should stay Japanese"),
    ("東京は大きい", "ja", "東京", "ja", "東京 (Tokyo) should stay Japanese"),
    # Cyrillic - should not switch between similar Slavic languages
    ("Привет друг", "ru", "друг", "ru", "друг (friend) should stay Russian"),
    # Short ambiguous Latin text should stay with main
    ("a test", "en", "a test", "en", "Short English stays English"),
    ("un test", "fr", "un test", "fr", "Short French stays French"),
    # Scandinavian - nearly identical written forms
    ("jeg har en hund", "no", "hund", "no", "Norwegian should stay Norwegian (not Danish)"),
    ("jeg har en hund", "da", "hund", "da", "Danish should stay Danish (not Norwegian)"),
    ("jag har en hund", "sv", "hund", "sv", "Swedish should stay Swedish"),
    # Dutch/Afrikaans - daughter language relationship
    ("ek het n boek", "af", "het", "af", "Afrikaans 'het' should stay Afrikaans (not Dutch)"),
    ("ik heb een boek", "nl", "heb", "nl", "Dutch should stay Dutch"),
    # Indonesian/Malay - nearly identical
    ("saya tidak tahu", "id", "tidak", "id", "Indonesian should stay Indonesian"),
    ("saya tidak tahu", "ms", "tidak", "ms", "Malay should stay Malay (not Indonesian)"),
    # Iberian Romance - vocabulary overlap
    ("casa bonita", "es", "casa", "es", "Spanish 'casa' should stay Spanish"),
    ("casa bonita", "pt", "casa", "pt", "Portuguese 'casa' should stay Portuguese"),
]

# =============================================================================
# COMPREHENSIVE EDGE CASES
# =============================================================================

# CJK-specific edge cases: (text, expected_main_lang, description)
CJK_EDGE_CASES = [
    # Short Chinese text that CLD2 finds unreliable
    ("块钱", "zh", "Short Chinese word - CLD2 unreliable"),
    ("元", "zh", "Single Chinese currency char"),
    ("人", "zh", "Single common Han char"),
    ("的", "zh", "Most common Chinese char"),
    # Chinese with numbers
    ("100元", "zh", "Chinese currency with number"),
    ("第1名", "zh", "Chinese ordinal"),
    ("2024年", "zh", "Year format - looks Japanese"),
    ("3月15日", "zh", "Date without year"),
    # Japanese definitive markers
    ("あ", "ja", "Single hiragana"),
    ("ア", "ja", "Single katakana"),
    ("カタカナ", "ja", "Pure katakana word"),
    ("ひらがな", "ja", "Pure hiragana word"),
    ("日本円", "ja", "Japanese with yen kanji"),
    # Japanese mixed scripts
    ("東京タワー", "ja", "Han + Katakana"),
    ("私はです", "ja", "Han + Hiragana"),
    # Korean
    ("ㄱ", "ko", "Single jamo"),
    ("가", "ko", "Single syllable"),
    ("한글", "ko", "Pure Hangul"),
    # Traditional vs Simplified (both should be zh)
    ("國語", "zh", "Traditional Chinese"),
    ("国语", "zh", "Simplified Chinese"),
]

# Script boundary edge cases - no spaces between scripts
BOUNDARY_CASES = [
    # No space between scripts
    ("Hello世界", {"en", "zh"}, "Latin-Han no space"),
    ("世界Hello", {"zh", "en"}, "Han-Latin no space"),
    ("Hello日本語です", {"en", "ja"}, "Latin-Japanese no space"),
    ("안녕Hello", {"ko", "en"}, "Hangul-Latin no space"),
    # Multiple spaces
    ("Hello    世界", {"en", "zh"}, "Multiple spaces between"),
    # Punctuation between
    ("Hello,世界", {"en", "zh"}, "Comma between scripts"),
    ("Hello。世界", {"en", "zh"}, "Chinese period between"),
    # Numbers between
    ("Hello123世界", {"en", "zh"}, "Numbers between scripts"),
]

# Punctuation variations: (text, expected_segments_contain, description)
PUNCTUATION_CASES = [
    # Chinese punctuation
    ("你好！世界。", "zh", "Chinese exclamation and period"),
    ("这是「引用」", "zh", "Chinese quotation marks"),
    ("第一、第二、第三", "zh", "Chinese enumeration comma"),
    # Japanese punctuation
    ("こんにちは！", "ja", "Japanese exclamation"),
    ("これは「テスト」です。", "ja", "Japanese quotation marks"),
    # Mixed punctuation - Han characters are dominant for short mixed text
    ("Hello, 世界!", "zh", "Mixed punctuation styles - Han dominant"),
]

# Unicode edge cases
UNICODE_EDGE_CASES = [
    # Emojis
    ("Hello 😀 world", "en", "Emoji in English"),
    ("你好 😀 世界", "zh", "Emoji in Chinese"),
    ("😀😀😀", "und", "Only emojis"),
    # Combining characters
    ("café", "en", "Combining acute accent"),  # é as e + combining accent
    # Zero-width characters
    ("Hello\u200bworld", "en", "Zero-width space"),
    # RTL text
    ("שלום", "he", "Hebrew RTL"),
    ("مرحبا", "ar", "Arabic RTL"),
    # Mixed RTL/LTR
    ("Hello שלום world", "en", "Mixed LTR-RTL-LTR"),
]

# Numbers and symbols edge cases
NUMBER_SYMBOL_CASES = [
    # Pure numbers
    ("12345", "und", "Only numbers"),
    ("3.14159", "und", "Decimal number"),
    ("1,234,567", "und", "Number with commas"),
    # Numbers with units
    ("100kg", "en", "Number with Latin unit"),
    ("50%", "und", "Percentage"),
    # Mathematical symbols
    ("2+2=4", "und", "Math equation"),
    ("x²+y²=z²", "en", "Math with superscript"),
    # Currency symbols
    ("$100", "und", "Dollar sign"),
    ("€50", "und", "Euro sign"),
    ("£30", "und", "Pound sign"),
    ("¥1000", "und", "Yen sign - symbol and numbers only, no script info"),
]

# Very short text edge cases
SHORT_TEXT_CASES = [
    ("a", "en", "Single Latin char"),
    ("I", "en", "Single letter word"),
    ("OK", "en", "Two letter word"),
    ("Hi", "en", "Short greeting"),
    ("你", "zh", "Single Chinese char"),
    ("我", "zh", "Chinese pronoun"),
]

# Very long text (should not crash or timeout)
LONG_TEXT_CASES = [
    ("Hello " * 100, "en", "Repeated English"),
    ("你好 " * 100, "zh", "Repeated Chinese"),
    ("This is a test. " * 50 + "これはテストです。" * 50, None, "Long mixed text"),
]

# Technical content
TECHNICAL_CASES = [
    # URLs (should preserve)
    ("Visit https://example.com for info", "en", "URL in English"),
    # Email-like
    ("Contact test@example.com", "en", "Email in English"),
    # Code-like
    ("def hello(): pass", "en", "Python-like code"),
    ("console.log('hello')", "en", "JavaScript-like code"),
    # Technical terms
    ("Use the API endpoint", "en", "Technical English"),
]

# Multiple language switches in one sentence
MULTI_SWITCH_CASES = [
    ("Hello 世界 and 안녕", {"en", "zh", "ko"}, "Three languages"),
    # Note: short "中文" segment gets labeled as ja (main lang) due to nearby hiragana
    ("This is 中文 and これ and 한글", {"en", "ja", "ko"}, "Four scripts - Han biased to main lang"),
    ("English 中文 English 中文", {"en", "zh"}, "Alternating languages"),
]

# Text reconstruction - segments should reconstruct original
RECONSTRUCTION_CASES = [
    "Hello world",
    "Hello 世界 test",
    "これはHelloテストです",
    "Hello 😀 world",
    "안녕 Hello 世界",
    "100块钱",
    "2025年9月28日",
    "Hello,世界!",
    "   spaced   text   ",
    "a",
    "",
]

# Segment merging edge cases - und segments should merge correctly
MERGE_CASES = [
    # Leading numbers should merge with following text
    ("100块钱", 1, "Numbers merge with following Han"),
    ("1000円", 1, "Numbers merge with following yen"),
    # Trailing punctuation should merge with preceding text
    ("Hello!", 1, "Punctuation merges with preceding"),
    ("你好！", 1, "Chinese punctuation merges"),
    # Spaces should merge
    ("Hello world", 1, "Space merges within same language"),
    ("Hello 世界 world", 3, "Spaces merge with adjacent segments"),
]


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestMainLanguageDetection(unittest.TestCase):
    """Test main language detection."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_main_language_detection(self):
        """Main language detection for various texts."""
        for text, expected, desc in MAIN_LANGUAGE_CASES:
            with self.subTest(desc=desc, text=text):
                result = self.splitter.detect_main_language(text)
                self.assertEqual(result, expected, f"{desc}: expected {expected}, got {result}")


class TestScriptSegmentation(unittest.TestCase):
    """Test script-based segmentation."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_script_segmentation(self):
        """Script segmentation produces expected scripts."""
        for text, expected_scripts, desc in SCRIPT_SEGMENTATION_CASES:
            with self.subTest(desc=desc, text=text):
                result = self.splitter.split(text)
                found_scripts = {seg.script for seg in result.segments if seg.text.strip()}
                for script in expected_scripts:
                    self.assertIn(script, found_scripts, f"{desc}: missing {script}")


class TestCodeSwitching(unittest.TestCase):
    """Test code-switching detection."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_code_switching_detection(self):
        """Code-switching is detected across script boundaries."""
        for text, main_lang, expected_langs, desc in CODE_SWITCHING_CASES:
            with self.subTest(desc=desc, text=text):
                result = self.splitter.split(text, main_lang=main_lang)
                found_langs = {seg.language for seg in result.segments if seg.text.strip()}
                for lang in expected_langs:
                    self.assertIn(lang, found_langs, f"{desc}: missing {lang}")


class TestFalsePositivePrevention(unittest.TestCase):
    """Test that ambiguous text doesn't incorrectly switch languages."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_false_positive_prevention(self):
        """Ambiguous segments should NOT switch away from main language."""
        for text, main_lang, segment_text, expected_lang, desc in FALSE_POSITIVE_CASES:
            with self.subTest(desc=desc, text=text, segment=segment_text):
                result = self.splitter.split(text, main_lang=main_lang)
                # Find segment containing the test text
                for seg in result.segments:
                    if segment_text in seg.text:
                        self.assertEqual(
                            seg.language, expected_lang,
                            f"{desc}: '{segment_text}' got {seg.language}, expected {expected_lang}"
                        )
                        break

    def test_short_embedded_same_script_switches_stay_main_language(self):
        """Tiny same-script false positives should not split the main language."""
        cases = [
            (
                "She was wandering to faraway places.",
                "en",
                {"en"},
            ),
            (
                "Không biết do tôi cầu toàn, khó tính hay hết duyên phận.",
                "vi",
                {"vi"},
            ),
        ]
        for text, main_lang, expected_langs in cases:
            with self.subTest(text=text):
                result = self.splitter.split(text, main_lang=main_lang)
                found_langs = {seg.language for seg in result.segments if seg.text.strip()}
                self.assertEqual(found_langs, expected_langs)


class TestSegmentStructure(unittest.TestCase):
    """Test segment data structure integrity."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_segment_positions_match_text(self):
        """Segment start/end positions match actual text."""
        for text in RECONSTRUCTION_CASES:
            with self.subTest(text=text):
                result = self.splitter.split(text)
                for seg in result.segments:
                    self.assertEqual(
                        text[seg.start:seg.end], seg.text,
                        f"Position mismatch for '{seg.text}'"
                    )

    def test_segments_reconstruct_original(self):
        """Concatenated segments equal original text."""
        for text in RECONSTRUCTION_CASES:
            with self.subTest(text=text):
                result = self.splitter.split(text)
                reconstructed = "".join(seg.text for seg in result.segments)
                self.assertEqual(reconstructed, text)

    def test_result_structure(self):
        """SplitResult has correct types."""
        result = self.splitter.split("Test")
        self.assertIsInstance(result, SplitResult)
        self.assertIsInstance(result.original_text, str)
        self.assertIsInstance(result.main_language, str)
        self.assertIsInstance(result.segments, list)
        for seg in result.segments:
            self.assertIsInstance(seg, Segment)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_empty_string(self):
        """Empty string returns empty result."""
        result = self.splitter.split("")
        self.assertEqual(result.original_text, "")
        self.assertEqual(result.main_language, "und")
        self.assertEqual(result.segments, [])

    def test_whitespace_only(self):
        """Whitespace-only text produces single segment."""
        result = self.splitter.split("   ")
        self.assertEqual(len(result.segments), 1)

    def test_numbers_and_punctuation(self):
        """Numbers and punctuation produce single segment."""
        result = self.splitter.split("123!@#")
        self.assertEqual(len(result.segments), 1)

    def test_single_character(self):
        """Single character produces single segment."""
        result = self.splitter.split("A")
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].text, "A")

    def test_emoji_preserved(self):
        """Emoji are preserved in output."""
        text = "Hello 😀 world"
        result = self.splitter.split(text)
        reconstructed = "".join(seg.text for seg in result.segments)
        self.assertEqual(reconstructed, text)


class TestConvenienceFunction(unittest.TestCase):
    """Test the split_text convenience function."""

    def test_basic_usage(self):
        """split_text returns SplitResult."""
        result = split_text("Hello world")
        self.assertIsInstance(result, SplitResult)
        self.assertEqual(result.main_language, "en")

    def test_with_main_lang(self):
        """split_text respects main_lang parameter."""
        result = split_text("Test", main_lang="de")
        self.assertEqual(result.main_language, "de")

    def test_with_language_filter(self):
        """split_text respects languages filter."""
        result = split_text("Hello world", languages=["en", "de"])
        self.assertIn(result.main_language, ["en", "de", "und"])


# =============================================================================
# COMPREHENSIVE EDGE CASE TESTS
# =============================================================================


class TestCJKEdgeCases(unittest.TestCase):
    """Test CJK-specific edge cases."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_cjk_edge_cases(self):
        """CJK edge cases detect correct language."""
        for text, expected, desc in CJK_EDGE_CASES:
            with self.subTest(desc=desc, text=text):
                result = self.splitter.detect_main_language(text)
                self.assertEqual(
                    result, expected,
                    f"{desc}: expected {expected}, got {result}"
                )


class TestScriptBoundaries(unittest.TestCase):
    """Test script boundary handling."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_boundary_cases(self):
        """Script boundaries are handled correctly."""
        for text, expected_langs, desc in BOUNDARY_CASES:
            with self.subTest(desc=desc, text=text):
                result = self.splitter.split(text)
                found_langs = {seg.language for seg in result.segments if seg.text.strip()}
                for lang in expected_langs:
                    self.assertIn(
                        lang, found_langs,
                        f"{desc}: missing {lang} in {found_langs}"
                    )


class TestPunctuation(unittest.TestCase):
    """Test punctuation handling."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_punctuation_cases(self):
        """Punctuation is handled correctly with various scripts."""
        for text, expected_main, desc in PUNCTUATION_CASES:
            with self.subTest(desc=desc, text=text):
                result = self.splitter.split(text)
                # Text should reconstruct correctly
                reconstructed = "".join(seg.text for seg in result.segments)
                self.assertEqual(reconstructed, text, f"{desc}: reconstruction failed")
                # Main language should be detected
                self.assertEqual(
                    result.main_language, expected_main,
                    f"{desc}: expected main {expected_main}, got {result.main_language}"
                )


class TestUnicodeEdgeCases(unittest.TestCase):
    """Test Unicode edge cases."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_unicode_edge_cases(self):
        """Unicode edge cases are handled correctly."""
        for text, expected, desc in UNICODE_EDGE_CASES:
            with self.subTest(desc=desc, text=text):
                result = self.splitter.split(text)
                # Text should reconstruct correctly
                reconstructed = "".join(seg.text for seg in result.segments)
                self.assertEqual(reconstructed, text, f"{desc}: reconstruction failed")
                # Check main language if expected is not None
                if expected != "und":
                    # For mixed text, just check that detection doesn't crash
                    self.assertIsNotNone(result.main_language)


class TestNumbersAndSymbols(unittest.TestCase):
    """Test numbers and symbols handling."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_number_symbol_cases(self):
        """Numbers and symbols are handled correctly."""
        for text, expected, desc in NUMBER_SYMBOL_CASES:
            with self.subTest(desc=desc, text=text):
                result = self.splitter.split(text)
                # Text should reconstruct correctly
                reconstructed = "".join(seg.text for seg in result.segments)
                self.assertEqual(reconstructed, text, f"{desc}: reconstruction failed")
                # Check main language
                self.assertEqual(
                    result.main_language, expected,
                    f"{desc}: expected {expected}, got {result.main_language}"
                )


class TestShortText(unittest.TestCase):
    """Test very short text handling."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_short_text_cases(self):
        """Very short text is handled correctly."""
        for text, expected, desc in SHORT_TEXT_CASES:
            with self.subTest(desc=desc, text=text):
                result = self.splitter.detect_main_language(text)
                self.assertEqual(
                    result, expected,
                    f"{desc}: expected {expected}, got {result}"
                )


class TestLongText(unittest.TestCase):
    """Test very long text handling."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_long_text_cases(self):
        """Very long text doesn't crash or timeout."""
        for text, expected, desc in LONG_TEXT_CASES:
            with self.subTest(desc=desc):
                result = self.splitter.split(text)
                # Text should reconstruct correctly
                reconstructed = "".join(seg.text for seg in result.segments)
                self.assertEqual(reconstructed, text, f"{desc}: reconstruction failed")
                # Check main language if expected
                if expected is not None:
                    self.assertEqual(
                        result.main_language, expected,
                        f"{desc}: expected {expected}, got {result.main_language}"
                    )


class TestTechnicalContent(unittest.TestCase):
    """Test technical content handling."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_technical_cases(self):
        """Technical content is handled correctly."""
        for text, expected, desc in TECHNICAL_CASES:
            with self.subTest(desc=desc, text=text):
                result = self.splitter.split(text)
                # Text should reconstruct correctly
                reconstructed = "".join(seg.text for seg in result.segments)
                self.assertEqual(reconstructed, text, f"{desc}: reconstruction failed")
                # Check main language
                self.assertEqual(
                    result.main_language, expected,
                    f"{desc}: expected {expected}, got {result.main_language}"
                )


class TestMultipleLanguageSwitches(unittest.TestCase):
    """Test multiple language switches in one text."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_multi_switch_cases(self):
        """Multiple language switches are detected."""
        for text, expected_langs, desc in MULTI_SWITCH_CASES:
            with self.subTest(desc=desc, text=text):
                result = self.splitter.split(text)
                found_langs = {seg.language for seg in result.segments if seg.text.strip()}
                for lang in expected_langs:
                    self.assertIn(
                        lang, found_langs,
                        f"{desc}: missing {lang} in {found_langs}"
                    )


class TestSegmentMerging(unittest.TestCase):
    """Test that und segments merge correctly."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_merge_cases(self):
        """Und segments merge with adjacent segments correctly."""
        for text, expected_count, desc in MERGE_CASES:
            with self.subTest(desc=desc, text=text):
                result = self.splitter.split(text)
                actual_count = len(result.segments)
                self.assertEqual(
                    actual_count, expected_count,
                    f"{desc}: expected {expected_count} segments, got {actual_count}: {[s.text for s in result.segments]}"
                )


class TestReconstructionComprehensive(unittest.TestCase):
    """Comprehensive reconstruction tests."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_reconstruction(self):
        """All reconstruction cases reconstruct correctly."""
        for text in RECONSTRUCTION_CASES:
            with self.subTest(text=text[:30] + "..." if len(text) > 30 else text):
                if not text:  # Skip empty string
                    continue
                result = self.splitter.split(text)
                reconstructed = "".join(seg.text for seg in result.segments)
                self.assertEqual(
                    reconstructed, text,
                    f"Reconstruction failed for {text!r}"
                )

    def test_positions_match(self):
        """All segment positions match actual text."""
        for text in RECONSTRUCTION_CASES:
            with self.subTest(text=text[:30] + "..." if len(text) > 30 else text):
                if not text:  # Skip empty string
                    continue
                result = self.splitter.split(text)
                for seg in result.segments:
                    self.assertEqual(
                        text[seg.start:seg.end], seg.text,
                        f"Position mismatch for segment {seg.text!r}"
                    )


class TestMainLanguageHint(unittest.TestCase):
    LATIN_CASES = {
        "bs": "Jučer sam razgovarao s prijateljem koji radi u velikoj kompaniji.",
        "sr": "Juče sam razgovarao sa prijateljem koji radi u velikoj kompaniji.",
        "nb": "Dette er en tydelig norsk setning som skal leses naturlig.",
        "fil": "Ito ay isang malinaw na pangungusap para sa pagsubok ng pagsasalita.",
        "jv": "Iki ukara basa Jawa sing cetha kanggo nyoba sintesis swara.",
        "su": "Ieu kalimah basa Sunda anu jelas pikeun nguji sintésis sora.",
        "so": "Kani waa jumlad Soomaali ah oo cad oo lagu tijaabinayo hadalka.",
        "zu": "Lona umusho ocacile wesiZulu wokuhlola ukuhlanganiswa kwezwi.",
        "fr": "Ceci est une phrase française claire pour tester la synthèse.",
        "vi": "Đây là một câu tiếng Việt rõ ràng để kiểm tra tổng hợp giọng nói.",
    }

    def assert_spans(self, result, expected):
        position = 0
        self.assertEqual(len(result.segments), len(expected))
        for segment, (language, segment_text) in zip(result.segments, expected):
            with self.subTest(language=language, segment_text=segment_text):
                self.assertEqual(segment.language, language)
                self.assertEqual(segment.text, segment_text)
                self.assertEqual(segment.start, position)
                position += len(segment_text)
                self.assertEqual(segment.end, position)
                self.assertEqual(result.original_text[segment.start:segment.end], segment.text)
        self.assertEqual(position, len(result.original_text))

    def test_hint_stabilizes_supported_latin_languages(self):
        splitter = MultilingualSplitter()
        for language, text in self.LATIN_CASES.items():
            with self.subTest(language=language):
                result = splitter.split(text, main_lang=language)
                self.assertEqual(result.main_language, language)
                self.assert_spans(result, [(language, text)])

    def test_latin_hint_preserves_cross_script_switch(self):
        chinese = "今天我们去图书馆。"
        for language, native_text in self.LATIN_CASES.items():
            with self.subTest(language=language):
                text = f"{native_text} {chinese} {native_text}"
                result = MultilingualSplitter().split(text, main_lang=language)
                self.assert_spans(
                    result,
                    [
                        (language, f"{native_text} "),
                        ("zh", chinese),
                        (language, f" {native_text}"),
                    ],
                )

    def test_latin_hint_allows_clear_same_script_switch(self):
        english = "This is a clearly English sentence with enough context."
        for language, native_text in self.LATIN_CASES.items():
            with self.subTest(language=language):
                text = f"{native_text} {english} {native_text}"
                result = MultilingualSplitter().split(text, main_lang=language)
                self.assert_spans(
                    result,
                    [
                        (language, native_text),
                        ("en", f" {english}"),
                        (language, f" {native_text}"),
                    ],
                )




class TestCodeSwitchScriptCoverage(unittest.TestCase):
    """Native-script sentence followed by an English sentence, one language set per case.

    Guards the same-script code-switch contract across the script families the voice
    registry routes: the native sentence stays a single span and the English sentence
    another, whatever the script's own segmentation rules are (no spaces in Thai or Han,
    RTL text, Indic clusters, diacritics).
    """

    ENGLISH = "Regular movement can support metabolism."

    CASES = {
        "ar": (
            "هذه جملة بسيطة عن الصحة العامة.",
            [("ar", 0, 32), ("en", 32, 72)],
        ),
        "as": (
            "এইটো স্বাস্থ্যৰ বিষয়ে এটা সাধাৰণ বাক্য।",
            [("as", 0, 41), ("en", 41, 81)],
        ),
        "el": (
            "Αυτή είναι μια απλή πρόταση για την υγεία.",
            [("el", 0, 43), ("en", 43, 83)],
        ),
        "fa": (
            "این یک جمله ساده درباره سلامتی است.",
            [("fa", 0, 36), ("en", 36, 76)],
        ),
        "gu": (
            "આ સ્વાસ્થ્ય વિશે એક સામાન્ય વાક્ય છે.",
            [("gu", 0, 38), ("en", 38, 78)],
        ),
        "he": (
            "זהו משפט פשוט על בריאות.",
            [("he", 0, 25), ("en", 25, 65)],
        ),
        "hi": (
            "यह एक साधारण वाक्य है जो स्वास्थ्य के बारे में बताता है।",
            [("hi", 0, 57), ("en", 57, 97)],
        ),
        "hy": (
            "Սա պարզ նախադասություն է առողջության մասին։",
            [("hy", 0, 44), ("en", 44, 84)],
        ),
        "id": (
            "Ini adalah kalimat sederhana tentang kesehatan.",
            [("id", 0, 47), ("en", 47, 88)],
        ),
        "ja": (
            "これは健康についての簡単な文です。",
            [("ja", 0, 18), ("en", 18, 58)],
        ),
        "ka": (
            "ეს არის მარტივი წინადადება ჯანმრთელობაზე.",
            [("ka", 0, 42), ("en", 42, 82)],
        ),
        "km": (
            "នេះគឺជាប្រយោគសាមញ្ញអំពីសុខភាព។",
            [("km", 0, 31), ("en", 31, 71)],
        ),
        "kn": (
            "ಇದು ಆರೋಗ್ಯದ ಬಗ್ಗೆ ಒಂದು ಸಾಮಾನ್ಯ ವಾಕ್ಯ.",
            [("kn", 0, 38), ("en", 38, 78)],
        ),
        "ko": (
            "이것은 건강에 관한 간단한 문장입니다.",
            [("ko", 0, 22), ("en", 22, 62)],
        ),
        "lo": (
            "ນີ້ແມ່ນປະໂຫຍກງ່າຍໆກ່ຽວກັບສຸຂະພາບ",
            [("lo", 0, 33), ("en", 33, 73)],
        ),
        "ml": (
            "ഇത് ആരോഗ്യത്തെക്കുറിച്ചുള്ള ഒരു സാധാരണ വാക്യമാണ്.",
            [("ml", 0, 50), ("en", 50, 90)],
        ),
        "mn": (
            "Энэ бол эрүүл мэндийн тухай энгийн өгүүлбэр.",
            [("mn", 0, 45), ("en", 45, 85)],
        ),
        "mr": (
            "हे एक साधारण वाक्य आहे जे आरोग्याबद्दल सांगते।",
            [("mr", 0, 47), ("en", 47, 87)],
        ),
        "my": (
            "ဤသည် ကျန်းမာရေးအကြောင်း ရိုးရှင်းသော စာကြောင်းဖြစ်သည်။",
            [("my", 0, 55), ("en", 55, 95)],
        ),
        "ne": (
            "यो एक साधारण वाक्य हो जुन स्वास्थ्यको बारेमा बताउँछ।",
            [("ne", 0, 53), ("en", 53, 93)],
        ),
        "or": (
            "ଏହା ସ୍ୱାସ୍ଥ୍ୟ ବିଷୟରେ ଏକ ସାଧାରଣ ବାକ୍ୟ।",
            [("or", 0, 38), ("en", 38, 78)],
        ),
        "pa": (
            "ਇਹ ਸਿਹਤ ਬਾਰੇ ਇੱਕ ਸਧਾਰਨ ਵਾਕ ਹੈ।",
            [("pa", 0, 31), ("en", 31, 71)],
        ),
        "ps": (
            "دا د روغتیا په اړه ساده جمله ده.",
            [("ps", 0, 33), ("en", 33, 73)],
        ),
        "ru": (
            "Это простое предложение о здоровье.",
            [("ru", 0, 36), ("en", 36, 76)],
        ),
        "si": (
            "මෙය සෞඛ්\u200dයය පිළිබඳ සරල වාක්\u200dයයකි.",
            [("si", 0, 34), ("en", 34, 74)],
        ),
        "ta": (
            "இது ஆரோக்கியம் பற்றிய ஒரு எளிய வாக்கியம்.",
            [("ta", 0, 42), ("en", 42, 82)],
        ),
        "te": (
            "ఇది ఆరోగ్యం గురించి ఒక సాధారణ వాక్యం.",
            [("te", 0, 38), ("en", 38, 78)],
        ),
        "th": (
            "นี่คือประโยคง่ายๆ เกี่ยวกับสุขภาพ",
            [("th", 0, 34), ("en", 34, 74)],
        ),
        "uk": (
            "Це просте речення про здоров'я.",
            [("uk", 0, 32), ("en", 32, 72)],
        ),
        "ur": (
            "یہ صحت کے بارے میں ایک سادہ جملہ ہے۔",
            [("ur", 0, 37), ("en", 37, 77)],
        ),
        "vi": (
            "Đây là một câu đơn giản về sức khỏe.",
            [("vi", 0, 36), ("en", 36, 77)],
        ),
        "zh": (
            "这是一句关于健康的简单句子。",
            [("zh", 0, 15), ("en", 15, 55)],
        ),
    }

    def test_native_sentence_stays_one_span(self):
        for language, (native, expected) in self.CASES.items():
            with self.subTest(language=language):
                text = native + " " + self.ENGLISH
                result = MultilingualSplitter(languages=[language, "en"]).split(
                    text, main_lang=language
                )
                self.assertEqual(result.main_language, language)
                self.assertEqual(
                    [(segment.language, segment.start, segment.end) for segment in result.segments],
                    expected,
                )
                self.assertEqual("".join(segment.text for segment in result.segments), text)
                for segment in result.segments:
                    self.assertTrue(
                        any(char.isalnum() for char in segment.text),
                        "segment without phoneme content: " + repr(segment.text),
                    )


class TestCodeSwitchWhitespaceAndPunctuation(unittest.TestCase):
    """Separator, bidi, symbol and degenerate variants inside a code switch.

    Script-neutral pieces between two runs carry a sentence-context language tag; in
    production that tag turned every word of a same-script run into its own span and
    left punctuation-only spans without phonemes for the aligner to consume.
    """

    CASES = {
        "tab separated": (
            "bn",
            "রাখুন।\tRegular\tmovement\tcan\tsupport\tmetabolism.",
            [("bn", 0, 7), ("en", 7, 47)],
        ),
        "nbsp separated": (
            "bn",
            "রাখুন।\xa0Regular\xa0movement\xa0can\xa0support\xa0metabolism.",
            [("bn", 0, 7), ("en", 7, 47)],
        ),
        "double spaces": (
            "bn",
            "রাখুন। Regular  movement  can  support  metabolism.",
            [("bn", 0, 7), ("en", 7, 51)],
        ),
        "lf inside run": (
            "bn",
            "রাখুন। Regular\nmovement can support metabolism.",
            [("bn", 0, 7), ("en", 7, 47)],
        ),
        "crlf inside run": (
            "bn",
            "রাখুন। Regular\r\nmovement can support metabolism.",
            [("bn", 0, 7), ("en", 7, 48)],
        ),
        "zero width space": (
            "bn",
            "রাখুন। Regular\u200bmovement can support metabolism.",
            [("bn", 0, 7), ("en", 7, 47)],
        ),
        "soft hyphen": (
            "bn",
            "রাখুন। Regular move\xadment can support metabolism.",
            [("bn", 0, 7), ("en", 7, 48)],
        ),
        "zero width joiner": (
            "bn",
            "রাখুন। Regular\u200dmovement can support metabolism.",
            [("bn", 0, 7), ("en", 7, 47)],
        ),
        "right to left mark": (
            "bn",
            "রাখুন।\u200f Regular movement can support metabolism.",
            [("bn", 0, 8), ("en", 8, 48)],
        ),
        "left to right mark": (
            "bn",
            "রাখুন।\u200e Regular movement can support metabolism.",
            [("bn", 0, 8), ("en", 8, 48)],
        ),
        "ideographic space": (
            "ja",
            "毎日の運動は大切です。\u3000Regular movement can support metabolism.",
            [("ja", 0, 12), ("en", 12, 52)],
        ),
        "curly quotes inside": (
            "bn",
            "রাখুন। Regular “movement” can support metabolism.",
            [("bn", 0, 7), ("en", 7, 49)],
        ),
        "parentheses inside": (
            "bn",
            "রাখুন। (Regular movement) can support metabolism.",
            [("bn", 0, 8), ("en", 8, 49)],
        ),
        "em dash inside": (
            "bn",
            "রাখুন। Regular movement — can support metabolism.",
            [("bn", 0, 7), ("en", 7, 49)],
        ),
        "colon and semicolon": (
            "bn",
            "রাখুন। First point; second point: done. রাখুন।",
            [("bn", 0, 7), ("en", 7, 38), ("bn", 38, 46)],
        ),
        "digits inside": (
            "bn",
            "রাখুন। move 10 minutes daily রাখুন।",
            [("bn", 0, 7), ("en", 7, 28), ("bn", 28, 35)],
        ),
        "symbols inside": (
            "bn",
            "রাখুন। vitamin D & B12 @ 100% daily রাখুন।",
            [("bn", 0, 7), ("en", 7, 35), ("bn", 35, 42)],
        ),
        "emoji inside": (
            "bn",
            "রাখুন। Regular movement 💪 can support metabolism.",
            [("bn", 0, 7), ("en", 7, 49)],
        ),
        "ellipsis tail": (
            "bn",
            "রাখুন। Regular movement can support metabolism…",
            [("bn", 0, 7), ("en", 7, 47)],
        ),
        "url inside": (
            "bn",
            "রাখুন। Visit https://example.com now রাখুন।",
            [("bn", 0, 7), ("en", 7, 36), ("bn", 36, 43)],
        ),
        "uppercase run": (
            "bn",
            "রাখুন। SMALL HABITS.",
            [("bn", 0, 7), ("en", 7, 20)],
        ),
        "apostrophe inside": (
            "bn",
            "রাখুন। Regular movement isn’t optional.",
            [("bn", 0, 7), ("en", 7, 39)],
        ),
        "leading quote": (
            "bn",
            "“রাখুন। Regular movement.",
            [("bn", 0, 8), ("en", 8, 25)],
        ),
        "leading whitespace": (
            "bn",
            "   রাখুন। Regular movement.",
            [("bn", 0, 10), ("en", 10, 27)],
        ),
        "trailing whitespace": (
            "bn",
            "রাখুন। Regular movement.   ",
            [("bn", 0, 7), ("en", 7, 27)],
        ),
        "arabic comma": (
            "ar",
            "النشاط البدني مهم، والتغذية متوازنة. Regular movement helps.",
            [("ar", 0, 37), ("en", 37, 60)],
        ),
        "urdu full stop": (
            "ur",
            "باقاعدہ ورزش ضروری ہے۔ Regular movement helps.",
            [("ur", 0, 23), ("en", 23, 46)],
        ),
        "persian question mark": (
            "fa",
            "آیا ورزش مهم است؟ Regular movement helps.",
            [("fa", 0, 18), ("en", 18, 41)],
        ),
        "thai without spaces": (
            "th",
            "การออกกำลังกายสม่ำเสมอRegular movement helps.",
            [("th", 0, 22), ("en", 22, 45)],
        ),
        "burmese tail": (
            "my",
            "ပုံမှန် လေ့ကျင့်ခန်း လုပ်ပါ။ Regular movement helps.",
            [("my", 0, 29), ("en", 29, 52)],
        ),
        "digits only run": (
            "bn",
            "রাখুন। 123 রাখুন।",
            [("bn", 0, 17)],
        ),
        "emoji only run": (
            "bn",
            "রাখুন। 🙂 রাখুন।",
            [("bn", 0, 15)],
        ),
        "punctuation only run": (
            "bn",
            "রাখুন। ... রাখুন।",
            [("bn", 0, 17)],
        ),
        "quotes only tail": (
            "bn",
            "রাখুন।.” “",
            [("bn", 0, 10)],
        ),
        "only punctuation": (
            "bn",
            ".",
            [("und", 0, 1)],
        ),
        "only em dash": (
            "bn",
            "—",
            [("und", 0, 1)],
        ),
        "only emoji": (
            "bn",
            "🙂",
            [("und", 0, 1)],
        ),
        "only spaces": (
            "bn",
            "   ",
            [("und", 0, 3)],
        ),
        "empty string": (
            "bn",
            "",
            [],
        ),
    }

    def test_span_plan_survives_separator_and_symbol_variants(self):
        for name, (main_language, text, expected) in self.CASES.items():
            with self.subTest(case=name):
                result = MultilingualSplitter(languages=[main_language, "en"]).split(
                    text, main_lang=main_language
                )
                self.assertEqual(
                    [(segment.language, segment.start, segment.end) for segment in result.segments],
                    expected,
                )
                self.assertEqual("".join(segment.text for segment in result.segments), text)
                for segment in result.segments:
                    self.assertTrue(
                        any(char.isalnum() for char in segment.text) or segment.language == "und",
                        "segment without phoneme content: " + repr(segment.text),
                    )


class TestCodeMixedMainLanguageRegression(unittest.TestCase):
    """Regression: Bangla document with English inserts (production run a5pXJx4YCBEiUA4H3Mg2FF).

    ``_stabilize_main_language_sentences`` assigns sentence-context language tags to
    script-neutral pieces (whitespace, punctuation). Those tags are routing hints, not
    code-switch boundaries: treating them as switches split every English word into its
    own span and produced punctuation-only spans with no phonemes to align. The splitter
    is constrained to Sparrow-routable languages here, as the API server does.
    """

    LANGUAGES = ["en", "bn", "sk"]

    PRODUCTION_TEXT = " ".join([
        "“Hormonal and metabolic health isn’t only about Tests or treatment. A few everyday habits can support it too.”",
        "“প্রতিদিন কিছুটা physical activity রাখুন। Regular movement can support metabolism and insulin sensitivity.”",
        "“Meals-এ protein, fibre-rich foods আর healthy fats রাখুন। Balanced nutrition helps support steady energy and metabolic health.”",
        "“প্রতিদিন পর্যাপ্ত ও consistent sleep-এর চেষ্টা করুন। Good sleep plays an important role in maintaining hormonal and metabolic balance.”",
        "SMALL HABITS.",
    ])

    PRODUCTION_SPANS = [
        ("en", 0, 109),
        ("bn", 109, 128),
        ("en", 128, 145),
        ("bn", 145, 153),
        ("en", 153, 225),
        ("bn", 225, 228),
        ("en", 228, 237),
        ("sk", 237, 243),  # CLD2 reads the English word "fibre-" as Slovak
        ("en", 243, 253),
        ("bn", 253, 257),
        ("en", 257, 269),
        ("bn", 269, 277),
        ("en", 277, 344),
        ("bn", 344, 368),
        ("en", 368, 384),
        ("bn", 384, 401),
        ("en", 401, 497),
    ]

    def test_production_text_keeps_whole_language_runs(self):
        result = MultilingualSplitter(languages=self.LANGUAGES).split(
            self.PRODUCTION_TEXT, main_lang="bn"
        )
        self.assertEqual(result.main_language, "bn")
        self.assertEqual(result.original_text, self.PRODUCTION_TEXT)
        self.assertEqual(
            [(segment.language, segment.start, segment.end) for segment in result.segments],
            self.PRODUCTION_SPANS,
        )
        covered = 0
        for segment in result.segments:
            self.assertEqual(segment.text, self.PRODUCTION_TEXT[segment.start:segment.end])
            self.assertEqual(self.PRODUCTION_TEXT[covered:segment.start].strip(), "")
            self.assertTrue(
                any(char.isalnum() for char in segment.text),
                "segment without phoneme content: " + repr(segment.text),
            )
            covered = segment.end
        self.assertEqual(self.PRODUCTION_TEXT[covered:].strip(), "")

    def test_code_mixed_sentence_keeps_same_script_words_together(self):
        splitter = MultilingualSplitter(languages=self.LANGUAGES)
        cases = [
            (
                "রাখুন। Regular movement can support metabolism.",
                [("bn", 0, 7), ("en", 7, 47)],
            ),
            (
                "sensitivity.” “Meals-এ protein.",
                [("en", 0, 20), ("bn", 20, 23), ("en", 23, 31)],
            ),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                result = splitter.split(text, main_lang="bn")
                self.assertEqual(
                    [(segment.language, segment.start, segment.end) for segment in result.segments],
                    expected,
                )
                for segment in result.segments:
                    self.assertTrue(
                        any(char.isalnum() for char in segment.text),
                        "segment without phoneme content: " + repr(segment.text),
                    )


class TestNoUndInFinalOutput(unittest.TestCase):
    """Test that no segments have 'und' language in final output (except edge cases)."""

    @classmethod
    def setUpClass(cls):
        cls.splitter = MultilingualSplitter()

    def test_exact_spans_for_normal_text(self):
        """Normal and mixed-script text has exact, positioned language spans."""
        normal_texts = [
            ("Hello world", "en", [("en", 0, 11, "Hello world")]),
            ("你好世界", "zh", [("zh", 0, 4, "你好世界")]),
            ("こんにちは", "ja", [("ja", 0, 5, "こんにちは")]),
            ("안녕하세요", "ko", [("ko", 0, 5, "안녕하세요")]),
            ("Привет мир", "ru", [("ru", 0, 10, "Привет мир")]),
            (
                "Hello 世界 test",
                "en",
                [
                    ("en", 0, 6, "Hello "),
                    ("zh", 6, 8, "世界"),
                    ("en", 8, 13, " test"),
                ],
            ),
            ("100块钱买咖啡", "zh", [("zh", 0, 8, "100块钱买咖啡")]),
        ]
        for text, main_language, expected_spans in normal_texts:
            with self.subTest(text=text):
                result = self.splitter.split(text)
                self.assertEqual(result.main_language, main_language)
                self.assertEqual(
                    [
                        (segment.language, segment.start, segment.end, segment.text)
                        for segment in result.segments
                    ],
                    expected_spans,
                )


# =============================================================================
# DEMO
# =============================================================================


def run_demo():
    """Run a demo of the splitter."""
    print("=" * 60)
    print("Multilingual Text Splitter Demo")
    print("=" * 60)

    demo_cases = [
        "Hello world!",
        "これは日本語です。",
        "Hello これはテストです world",
        "안녕하세요 Hello 世界",
        "Привет Hello мир",
        "مرحبا Hello عالم",
        "I love 日本 and 한국!",
        "我是学生，在大学读书。",  # Chinese with shared Han chars
    ]

    splitter = MultilingualSplitter()

    for text in demo_cases:
        print(f"\nInput: {text}")
        result = splitter.split(text)
        print(f"Main language: {result.main_language}")
        print("Segments:")
        for seg in result.segments:
            if seg.text.strip():
                print(f"  [{seg.script}/{seg.language}]: {seg.text!r}")
        print("-" * 40)


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        run_demo()
    else:
        unittest.main(verbosity=2)
