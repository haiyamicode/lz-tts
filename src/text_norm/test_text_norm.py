import unittest
from unittest.mock import patch

from src.text_norm import (
    canonicalize_text,
    normalize_spoken_text,
    normalize_text,
    prepare_tts_texts,
)


class TestTextNorm(unittest.TestCase):
    def test_assamese_normalizes_known_nukta_spellings(self):
        decomposed = "\u09af\u09bc \u09a1\u09bc \u09a2\u09bc"
        precomposed = "\u09df \u09dc \u09dd"
        expected = "\u09df \u09f0 \u09f0\u09cd\u09b9"

        self.assertEqual(normalize_text(decomposed, "as-IN"), expected)
        self.assertEqual(normalize_text(precomposed, "asm"), expected)

    def test_thai_normalizes_decomposed_sara_am(self):
        self.assertEqual(normalize_text("น้\u0e4d\u0e32", "th-TH"), "น้ำ")

    def test_khmer_verbalizes_numbers_and_percent(self):
        self.assertEqual(
            normalize_text("ឆ្នាំ២០២៦ មាន ១៥%", "km-KH"),
            "ឆ្នាំពីរពាន់ម្ភៃប្រាំមួយ មាន ដប់ប្រាំ ភាគរយ",
        )

    def test_khmer_handles_grouped_and_dotted_numbers(self):
        self.assertEqual(
            normalize_text("១២០ ០០០ និង ៨.៨.៨.៨", "khm"),
            (
                "មួយសែនពីរម៉ឺន និង ប្រាំបី ចុច ប្រាំបី ចុច "
                "ប្រាំបី ចុច ប្រាំបី"
            ),
        )

    def test_khmer_verbalizes_arithmetic_operators(self):
        self.assertEqual(
            normalize_text("២ + ៣ = ៥ និង ១០ - ៤ = ៦", "km"),
            "ពីរ បូក បី ស្មើ ប្រាំ និង ដប់ ដក បួន ស្មើ ប្រាំមួយ",
        )

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

    def test_vietnamese_round_the_clock_normalization(self):
        self.assertEqual(
            normalize_text("Hỗ trợ 24/7 và trực 24/24.", "vi-VN"),
            "Hỗ trợ hai mươi tư trên bảy và trực hai mươi tư trên hai mươi tư.",
        )

    def test_vietnamese_terminal_four_after_tens_uses_tu(self):
        self.assertEqual(
            normalize_text("14, 24, 34, 44, 104, 124, 240.", "vi-VN"),
            (
                "mười bốn, hai mươi tư, ba mươi tư, bốn mươi tư, "
                "một trăm lẻ bốn, một trăm hai mươi tư, hai trăm bốn mươi."
            ),
        )

    def test_english_round_the_clock_normalization(self):
        self.assertEqual(
            normalize_text("Support is available 24/7.", "en-US"),
            "support is available twenty four seven.",
        )

    def test_shared_preparation_normalizes_before_batched_replacements(self):
        class FakeReplacer:
            def apply_replacements_many(self, texts, language=None):
                return [text.replace("AI", "ây ai") for text in texts]

        with patch(
            "src.piper.context_replacer.get_replacer",
            return_value=FakeReplacer(),
        ) as get_replacer:
            self.assertEqual(
                prepare_tts_texts(
                    ["AI hỗ trợ 24/7.", "Keep AI"],
                    ["vi-VN", "en-US"],
                    context_replacer_device="cpu",
                ),
                [
                    "ây ai hỗ trợ hai mươi tư trên bảy.",
                    "keep ai",
                ],
            )
        get_replacer.assert_called_once_with(device="cpu")

    def test_canonical_normalization_does_not_verbalize(self):
        self.assertEqual(
            canonicalize_text("  Ｈỗ trợ 24／7。\n"),
            "Hỗ trợ 24/7.",
        )

    def test_spoken_round_the_clock_per_language(self):
        expected = {
            "de-DE": "vierundzwanzig Stunden am Tag, sieben Tage die Woche",
            "en-US": "twenty four seven",
            "es-ES": "veinticuatro horas al día, siete días a la semana",
            "fr-FR": "vingt-quatre heures sur vingt-quatre, sept jours sur sept",
            "it-IT": "ventiquattro ore su ventiquattro, sette giorni su sette",
            "ja-JP": "二十四時間、週七日",
            "ko-KR": "하루 이십사 시간, 주 칠일",
            "pt-BR": "vinte e quatro horas por dia, sete dias por semana",
            "vi-VN": "hai mươi tư trên bảy",
            "zh-CN": "二十四小时，每周七天",
        }
        for language, spoken in expected.items():
            with self.subTest(language=language):
                self.assertEqual(normalize_spoken_text("24/7", language), spoken)

    def test_spoken_normalization_uses_locale_owned_aliases(self):
        cases = {
            ("24 hours/24", "en-US"): "twenty four hours a day",
            ("24 heures/24", "fr-FR"): "vingt-quatre heures sur vingt-quatre",
            ("24 Stunden/24", "de-DE"): "vierundzwanzig Stunden am Tag",
            ("14:30 Uhr", "de-DE"): "vierzehn Uhr dreißig",
            ("2 miles/hours", "en-US"): "two miles per hour",
            ("2 Meilen/Stunden", "de-DE"): "zwei Meilen pro Stunde",
            ("2 kilomètres/secondes", "fr-FR"): (
                "deux kilomètres par seconde"
            ),
        }
        for (source, language), expected in cases.items():
            with self.subTest(source=source, language=language):
                self.assertEqual(
                    normalize_spoken_text(source, language),
                    expected,
                )

    def test_spoken_normalization_expands_safe_numeric_forms(self):
        self.assertEqual(
            normalize_spoken_text(
                "Mở 14:30, tải 23kg, pin 30%, giá $12.50.",
                "vi-VN",
            ),
            (
                "Mở mười bốn giờ ba mươi, tải hai mươi ba ki lô gam, "
                "pin ba mươi phần trăm, giá mười hai phẩy năm không đô la."
            ),
        )
        self.assertEqual(
            normalize_spoken_text("At 9:05, carry 1kg.", "en-US"),
            "At nine oh five, carry one kilogram.",
        )

    def test_spoken_normalization_does_not_drop_unsupported_meridiem(self):
        expected = {
            "de": "Meet at zwei Uhr dreißig p.m.",
            "es": "Meet at dos horas y treinta minutos p.m.",
            "fr": "Meet at deux heures trente p.m.",
            "it": "Meet at due e trenta p.m.",
            "ja": "Meet at 二時三十分 p.m.",
            "ko": "Meet at 이시 삼십분 p.m.",
            "pt": "Meet at dois horas e trinta minutos p.m.",
            "vi": "Meet at hai giờ ba mươi p.m.",
        }
        for language, spoken in expected.items():
            with self.subTest(language=language):
                self.assertEqual(
                    normalize_spoken_text("Meet at 2:30p.m.", language),
                    spoken,
                )
        self.assertEqual(
            normalize_spoken_text("Meet at 2:30 p.m.", "en"),
            "Meet at two thirty p.m.",
        )

    def test_spoken_normalization_speaks_negative_signs(self):
        expected = {
            "de": "minus fünf Grad Celsius",
            "en": "minus five degrees Celsius",
            "es": "menos cinco grados Celsius",
            "fr": "moins cinq degrés Celsius",
            "it": "meno cinque gradi Celsius",
            "ja": "マイナス五度",
            "ko": "마이너스 오 도",
            "pt": "menos cinco graus Celsius",
            "vi": "âm năm độ C",
        }
        for language, spoken in expected.items():
            with self.subTest(language=language):
                self.assertEqual(
                    normalize_spoken_text("-5 °C", language),
                    spoken,
                )

    def test_spoken_normalization_expands_compound_speed_units(self):
        cases = {
            "de-DE": "siebzig Kilometer pro Stunde",
            "en-US": "seventy kilometers per hour",
            "es-ES": "setenta kilómetros por hora",
            "fr-FR": "soixante-dix kilomètres par heure",
            "it-IT": "settanta chilometri per ora",
            "ja-JP": "一時間あたり七十キロメートル",
            "ko-KR": "시간당 칠십 킬로미터",
            "pt-BR": "setenta quilômetros por hora",
            "vi-VN": "bảy mươi ki lô mét trên giờ",
            "zh-CN": "每小时七十公里",
        }
        for language, expected in cases.items():
            with self.subTest(language=language):
                self.assertEqual(
                    normalize_spoken_text("70 km/h", language),
                    expected,
                )
        self.assertEqual(
            normalize_spoken_text("1 km/h", "en-US"),
            "one kilometer per hour",
        )
        feminine_singular_cases = {
            "de-DE": "eine Meile pro Stunde",
            "es-ES": "una milla por hora",
            "pt-BR": "uma milha por hora",
        }
        for language, expected in feminine_singular_cases.items():
            with self.subTest(language=language):
                self.assertEqual(
                    normalize_spoken_text("1 mi/h", language),
                    expected,
                )

    def test_spoken_normalization_generalizes_compound_measurements(self):
        cases = {
            "70 miles/h": "seventy miles per hour",
            "1 mi/h": "one mile per hour",
            "5 m/h": "five meters per hour",
            "5 m/s": "five meters per second",
            "5 km/s": "five kilometers per second",
            "2 MB/s": "two megabytes per second",
            "3 kg/day": "three kilograms per day",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(
                    normalize_spoken_text(source, "en-US"),
                    expected,
                )
        self.assertEqual(
            normalize_spoken_text("Throughput is 5 M/s.", "en-US"),
            "Throughput is 5 M/s.",
        )

    def test_spoken_normalization_requires_standard_unit_symbol_case(self):
        self.assertEqual(
            normalize_spoken_text("5w, 2mb, 3ghz, 4mah, 20°c", "en-US"),
            "5w, 2mb, 3ghz, 4mah, 20°c",
        )
        self.assertEqual(
            normalize_spoken_text("5W, 2MB, 3GHz, 4mAh, 20°C", "en-US"),
            (
                "five watts, two megabytes, three gigahertz, "
                "four milliamp hours, twenty degrees Celsius"
            ),
        )

    def test_spoken_normalization_expands_currency_in_every_language(self):
        cases = {
            "de-DE": ("50 €", "fünfzig Euro"),
            "en-US": ("$1.00", "one dollar"),
            "es-ES": ("2 USD", "dos dólares"),
            "fr-FR": ("50 €", "cinquante euros"),
            "it-IT": ("2 GBP", "due sterline"),
            "ja-JP": ("¥1,000", "千円"),
            "ko-KR": ("₩1,000", "천 원"),
            "pt-BR": ("R$ 15,00", "quinze reais brasileiros"),
            "vi-VN": ("50.000₫", "năm mươi nghìn đồng"),
            "zh-CN": ("¥1,000", "一千元"),
        }
        for language, (source, expected) in cases.items():
            with self.subTest(language=language):
                self.assertEqual(
                    normalize_spoken_text(source, language),
                    expected,
                )

    def test_chinese_spoken_normalization_uses_chinese_word_order(self):
        self.assertEqual(
            normalize_spoken_text(
                "速度70 km/h，完成50%，温度-5°C。",
                "zh-CN",
            ),
            "速度每小时七十公里,完成百分之五十,温度负五摄氏度.",
        )

    def test_spoken_normalization_expands_currency_aliases_and_magnitudes(self):
        cases = [
            ("C$2", "en-US", "two Canadian dollars"),
            ("A$2", "en-US", "two Australian dollars"),
            ("CN¥2", "en-US", "two Chinese yuan"),
            ("₹2", "en-US", "two Indian rupees"),
            ("$250 million", "en-US", "two hundred and fifty million dollars"),
            (
                "$45 million investment",
                "en-US",
                "forty-five million dollar investment",
            ),
            ("1 GBP", "es-ES", "una libra"),
            ("1 GBP", "fr-FR", "une livre"),
            ("1 GBP", "it-IT", "una sterlina"),
            ("1 GBP", "pt-BR", "uma libra"),
            (
                "$12‐million investment",
                "en-US",
                "twelve million dollar investment",
            ),
            ("3 millions €", "fr-FR", "trois millions d'euros"),
            ("4 millones EUR", "es-ES", "cuatro millones de euros"),
            ("5 milioni EUR", "it-IT", "cinque milioni di euro"),
            ("R$ 2 milhões", "pt-BR", "dois milhões de reais brasileiros"),
            ("5 tỷ USD", "vi-VN", "năm tỷ u ét đê"),
            ("50.000đ", "vi-VN", "năm mươi nghìn đồng"),
            ("50.000 VND", "vi-VN", "năm mươi nghìn Việt Nam đồng"),
        ]
        for source, language, expected in cases:
            with self.subTest(source=source, language=language):
                self.assertEqual(
                    normalize_spoken_text(source, language),
                    expected,
                )

    def test_spoken_normalization_does_not_expand_bare_currency_codes(self):
        self.assertEqual(
            normalize_spoken_text("Prepare the CAD models.", "en-US"),
            "Prepare the CAD models.",
        )

    def test_vietnamese_currency_symbol_does_not_consume_word_prefixes(self):
        self.assertEqual(
            normalize_spoken_text(
                "Giá 200.000 đồng, từ 7 đến 8 giờ, nhiệt độ 22 độ C.",
                "vi-VN",
            ),
            (
                "Giá hai trăm nghìn đồng, từ bảy đến tám giờ, "
                "nhiệt độ hai mươi hai độ C."
            ),
        )
        self.assertEqual(
            normalize_spoken_text("Giá 50.000đ.", "vi-VN"),
            "Giá năm mươi nghìn đồng.",
        )

    def test_spoken_normalization_expands_safe_compact_forms(self):
        self.assertEqual(
            normalize_spoken_text(
                "Video 8K, mạng 5G, mô hình 3D và khí CO2, trực 24h/24.",
                "vi-VN",
            ),
            (
                "Video tám ca, mạng năm gi, mô hình ba đê và khí xê ô hai, "
                "trực hai mươi tư trên hai mươi tư."
            ),
        )
        self.assertEqual(
            normalize_spoken_text(
                "A 3D model in 4K marked its 25th release.",
                "en-US",
            ),
            "A three D model in four K marked its twenty-fifth release.",
        )
        self.assertEqual(
            normalize_spoken_text("Du 1er au 3D, avec du CO2.", "fr-FR"),
            "Du premier au trois D, avec du C O deux.",
        )

    def test_spoken_normalization_expands_vietnamese_legal_document_ids(self):
        self.assertEqual(
            normalize_spoken_text(
                "Nghị định 12/2026/QĐ-TP và 7 / 2024 / NĐ-CP.",
                "vi-VN",
            ),
            (
                "Nghị định mười hai hai không hai sáu QĐ-TP "
                "và bảy hai không hai bốn NĐ-CP."
            ),
        )
        self.assertEqual(
            normalize_spoken_text("Ngày 12/06/2026.", "vi-VN"),
            "Ngày mười hai tháng sáu năm hai nghìn lẻ hai mươi sáu.",
        )
        self.assertEqual(
            normalize_spoken_text("Ngày 29/02/2025.", "vi-VN"),
            "Ngày 29/02/2025.",
        )

    def test_spoken_normalization_expands_common_language_specific_forms(self):
        self.assertEqual(
            normalize_spoken_text(
                "Chào anh/chị, giá 30.000 đồng/kg, lãi suất 5%/năm.",
                "vi-VN",
            ),
            (
                "Chào anh hoặc chị, giá ba mươi nghìn đồng một ki lô gam, "
                "lãi suất năm phần trăm một năm."
            ),
        )
        self.assertEqual(
            normalize_spoken_text("Le train n° 842.", "fr-FR"),
            "Le train numéro huit cent quarante-deux.",
        )
        self.assertEqual(
            normalize_spoken_text("Join the Q&A session.", "en-US"),
            "Join the Q and A session.",
        )

    def test_spoken_normalization_preserves_ambiguous_machine_text(self):
        text = (
            "Visit https://x.test/v2.4 at 192.168.0.1 "
            "on 2026-07-24 for build 1.2.3."
        )
        self.assertEqual(normalize_spoken_text(text, "vi-VN"), text)
        self.assertEqual(normalize_spoken_text(text, "fr-FR"), text)
        machine_tokens = (
            "Use v1.2.3, CUDA12.4, model-v2.4, 3.10rc1, "
            "and 192.168.001.001."
        )
        self.assertEqual(
            normalize_spoken_text(machine_tokens, "en-US"),
            machine_tokens,
        )

    def test_spoken_normalization_verbalizes_signed_ranges_atomically(self):
        self.assertEqual(
            normalize_spoken_text("Expected -5-9 or 5--9.", "en-US"),
            "Expected minus five to nine or five to minus nine.",
        )

    def test_spoken_normalization_does_not_modify_opaque_spans(self):
        cases = [
            (
                "Open https://example.com/video/4K/25th?x=Q&A.",
                "en-US",
            ),
            (
                "Voir https://x.fr/train/n°842/24/7.",
                "fr-FR",
            ),
            (
                "Keep /srv/models/4K/24/7 unchanged.",
                "en-US",
            ),
            (
                "Call 123-456-7890 or use ticket 123-456.",
                "en-US",
            ),
            (
                "Call 0800‐123‐4567 or 0909.123.456.",
                "vi-VN",
            ),
            (
                "Téléphonez au 01 45 78 90 12.",
                "fr-FR",
            ),
            (
                "電話090-1234-5678、パス/srv/models/4K/file.",
                "ja-JP",
            ),
        ]
        for text, language in cases:
            with self.subTest(text=text, language=language):
                self.assertEqual(normalize_spoken_text(text, language), text)

    def test_spoken_normalization_uses_unicode_token_boundaries(self):
        cases = [
            ("Mã á24/7 và é3D không phải token riêng.", "vi-VN"),
            ("Identifiant é25th et à3D.", "fr-FR"),
        ]
        for text, language in cases:
            with self.subTest(text=text, language=language):
                self.assertEqual(normalize_spoken_text(text, language), text)

    def test_spoken_normalization_allows_complete_tokens_next_to_cjk(self):
        cases = {
            ("進捗は70%です。", "ja"): "進捗は七十パーセントです.",
            ("会議は14:30に始まります。", "ja"): "会議は十四時三十分に始まります.",
            ("速度は70km/hです。", "ja"): (
                "速度は一時間あたり七十キロメートルです."
            ),
            ("진행률은70%입니다.", "ko"): "진행률은칠십 퍼센트입니다.",
        }
        for (text, language), expected in cases.items():
            with self.subTest(text=text, language=language):
                self.assertEqual(
                    normalize_spoken_text(text, language),
                    expected,
                )

    def test_spoken_normalization_preserves_ambiguous_hour_only_tokens(self):
        for language in ("de", "en", "es", "fr", "it", "ja", "ko", "pt", "vi"):
            with self.subTest(language=language):
                self.assertEqual(
                    normalize_spoken_text("Work for 10h.", language),
                    "Work for 10h.",
                )

    def test_shared_preparation_accepts_spoken_profile(self):
        self.assertEqual(
            prepare_tts_texts(
                ["24/7", "14:30"],
                ["de-DE", "ja-JP"],
                normalization_profile="spoken",
                context_replacements_enabled=False,
            ),
            [
                "vierundzwanzig Stunden am Tag, sieben Tage die Woche",
                "十四時三十分",
            ],
        )

    def test_shared_preparation_rejects_unknown_profile(self):
        with self.assertRaisesRegex(ValueError, "Unknown normalization profile"):
            prepare_tts_texts(
                ["hello"],
                ["en-US"],
                normalization_profile="unsupported",
                context_replacements_enabled=False,
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
