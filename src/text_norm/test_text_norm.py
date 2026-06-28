import unittest

from src.text_norm import normalize_text


class TestTextNorm(unittest.TestCase):
    def test_english_time_preserves_following_space(self):
        self.assertEqual(
            normalize_text("Meet at 14:05 on 21st St.", "en-us"),
            "meet at two oh five p.m. on 21st st.",
        )

    def test_english_bare_clock_time_does_not_invent_meridiem(self):
        self.assertEqual(
            normalize_text("The briefing starts at 9:30 tomorrow morning.", "en-us"),
            "the briefing starts at 9:30 tomorrow morning.",
        )
        self.assertEqual(
            normalize_text("Does boarding start at 7:45?", "en-us"),
            "does boarding start at 7:45?",
        )

    def test_english_explicit_meridiem_is_preserved(self):
        self.assertEqual(
            normalize_text("Meet at 5:00 p.m.", "en-us"),
            "meet at five p.m.",
        )

    def test_english_24_hour_midnight_uses_am(self):
        self.assertEqual(
            normalize_text("The server restarted at 00:05.", "en-us"),
            "the server restarted at twelve oh five a.m.",
        )

    def test_english_keeps_numbers_espeak_handles(self):
        self.assertEqual(
            normalize_text("Call 555-1234 after version 1.2.3 ships.", "en-us"),
            "call 555-1234 after version 1.2.3 ships.",
        )
        self.assertEqual(
            normalize_text("There were 1,234 attendees and the score was 3-2.", "en-us"),
            "there were 1,234 attendees and the score was 3-2.",
        )

    def test_english_keeps_espeak_handled_abbreviations(self):
        self.assertEqual(
            normalize_text("Dr. Chen lives on Main St. near St. Louis.", "en-us"),
            "dr. chen lives on main st. near st. louis.",
        )

    def test_english_expands_currency(self):
        self.assertEqual(
            normalize_text("It cost $12.50.", "en-us"),
            "it cost 12 dollars 50 cents.",
        )

    def test_english_expands_abbreviations_espeak_does_not_handle(self):
        self.assertEqual(
            normalize_text("Gen. Adams met Capt. Lee at Ft. Knox.", "en-us"),
            "general adams met captain lee at fort knox.",
        )

    def test_chinese_mixed_english_is_not_normalized(self):
        text = "今天 shopping 123。"
        self.assertEqual(normalize_text(text, "zh-mix-en"), text)

    def test_cmn_latin_pinyin_is_not_stripped(self):
        self.assertEqual(normalize_text("ni3 hao3", "cmn-latn-pinyin"), "ni3 hao3")

    def test_han_chinese_numbers_are_normalized(self):
        self.assertEqual(normalize_text("今天有123个苹果。", "zh"), "今天有一百二十三个苹果.")

    def test_japanese_normalization_preserves_written_text(self):
        self.assertEqual(normalize_text("今日は１２３個のリンゴがあります。", "ja"), "今日は123個のリンゴがあります.")

    def test_korean_normalization_preserves_written_text(self):
        self.assertEqual(normalize_text("ＡＢＣ와 ３＋ 테스트입니다。", "ko"), "ABC와 3+ 테스트입니다.")

    def test_vietnamese_number_percent_normalization(self):
        self.assertEqual(
            normalize_text("Pin còn 3%", "vi-VN"),
            "Pin còn ba phần trăm",
        )

    def test_vietnamese_grouped_and_decimal_numbers(self):
        self.assertEqual(
            normalize_text("Nhận 1.000.000 đô, giá 3.14.", "vi-VN"),
            "Nhận một triệu đô, giá ba phẩy mười bốn.",
        )

    def test_vietnamese_mixed_context_replacement_input(self):
        self.assertEqual(
            normalize_text("Startup về AI tăng 25% sau khi nhận 10.000.000 đô.", "vi-VN"),
            "Startup về AI tăng hai mươi lăm phần trăm sau khi nhận mười triệu đô.",
        )

    def test_alignment_keeps_japanese_written_text(self):
        from src.piper.preprocess import _normalize_text_for_mapping

        self.assertEqual(
            _normalize_text_for_mapping("今日は１２３個のリンゴがあります。", "ja"),
            "今日は123個のリンゴがあります. ",
        )

    def test_alignment_keeps_korean_written_text(self):
        from src.piper.preprocess import _normalize_text_for_mapping

        self.assertEqual(
            _normalize_text_for_mapping("ＡＢＣ와 ３＋ 테스트입니다。", "ko"),
            "ABC와 3+ 테스트입니다. ",
        )

    def test_alignment_still_normalizes_english(self):
        from src.piper.preprocess import _normalize_text_for_mapping

        self.assertEqual(
            _normalize_text_for_mapping("Meet at 14:05 on 21st St.", "en-us"),
            "meet at two oh five p.m. on 21st st.",
        )


if __name__ == "__main__":
    unittest.main()
