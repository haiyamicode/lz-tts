from __future__ import annotations

import unittest

from .locale_utils import normalize_locale


class NormalizeLocaleTests(unittest.TestCase):
    def test_language_region(self) -> None:
        self.assertEqual(normalize_locale("en-us"), "en-US")
        self.assertEqual(normalize_locale("pt_BR"), "pt-BR")

    def test_script_is_not_a_region(self) -> None:
        self.assertEqual(normalize_locale("zh-Hant"), "zh")
        self.assertEqual(normalize_locale("sr-Latn"), "sr")

    def test_region_after_script_is_preserved(self) -> None:
        self.assertEqual(normalize_locale("zh-Hant-HK"), "zh-HK")
        self.assertEqual(normalize_locale("sr-Latn-RS"), "sr-RS")

    def test_variants_are_discarded(self) -> None:
        self.assertEqual(normalize_locale("zh-CN-guangxi"), "zh-CN")


if __name__ == "__main__":
    unittest.main()
