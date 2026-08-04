from collections import Counter

from piper_phonemize import phoneme_ids_espeak

from .preprocess import (
    _map_cld2_to_espeak,
    _phonemize_espeak_for_voice_with_spans,
)


def _identity(text: str) -> str:
    return text


def _assert_supported(phonemes: list[str]) -> None:
    missing: Counter[str] = Counter()
    phoneme_ids_espeak(phonemes, missing_phonemes=missing)
    assert not missing


def test_khmer_uses_supported_ipa_and_word_spans() -> None:
    text = "នាងម៉ាលីញញឹមទាំង ទឹកភ្នែក។"
    phonemes, word_spans, semantic_text = _phonemize_espeak_for_voice_with_spans(
        text, "km", _identity, None
    )

    assert semantic_text == text
    assert phonemes
    assert word_spans
    assert not {"ĕ", "ŏ", "ŭ"}.intersection(phonemes)
    _assert_supported(phonemes)


def test_khmer_numbers_are_verbalized_before_phonemization() -> None:
    text = "ឆ្នាំ ២០២៦ មាន ១៥% និង ៨.៨.៨.៨"
    phonemes, word_spans, semantic_text = _phonemize_espeak_for_voice_with_spans(
        text, "km", _identity, None
    )

    assert not any("\u17e0" <= char <= "\u17e9" for char in semantic_text)
    assert "ភាគរយ" in semantic_text
    assert semantic_text.count("ចុច") == 3
    assert phonemes
    assert word_spans
    _assert_supported(phonemes)


def test_lao_uses_supported_segmental_ipa_and_phrase_spans() -> None:
    text = "ມື້ນີ້ເປັນມື້ ທີ່ມີແດດສົດໃສ"
    phonemes, word_spans, semantic_text = _phonemize_espeak_for_voice_with_spans(
        text, "lo", _identity, None
    )

    assert semantic_text == text
    assert phonemes
    assert word_spans
    assert not {"˩", "˨", "˧", "˥", "ʷ"}.intersection(phonemes)
    _assert_supported(phonemes)


def test_myanmar_uses_epitran_with_supported_ipa_and_phrase_spans() -> None:
    text = "မင်္ဂလာပါ။ ဒီနေ့ မနက်ခင်းမှာ"
    phonemes, word_spans, semantic_text = _phonemize_espeak_for_voice_with_spans(
        text, "my", _identity, None
    )

    assert semantic_text == text
    assert phonemes
    assert word_spans
    assert "".join(phonemes).startswith("mɪɴɡəlapa.")
    assert not any("\u1000" <= char <= "\u109f" for char in phonemes)
    assert not {"͡", "̥"}.intersection(phonemes)
    _assert_supported(phonemes)


def test_thai_uses_epitran_with_supported_ipa_and_normalizes_sara_am() -> None:
    text = "การบริหารจัดการทรัพยากรน้ําจําเป็นต้องอาศัย"
    phonemes, word_spans, semantic_text = _phonemize_espeak_for_voice_with_spans(
        text, "th", _identity, None
    )

    assert semantic_text == "การบริหารจัดการทรัพยากรน้ำจำเป็นต้องอาศัย"
    assert phonemes
    assert word_spans
    assert "thai letter" not in "".join(phonemes).lower()
    assert not any("\u0e00" <= char <= "\u0e7f" for char in phonemes)
    assert "͡" not in phonemes
    _assert_supported(phonemes)


def test_mongolian_uses_epitran_without_cyrillic_letter_names() -> None:
    text = "Урд шөнө Улаанбаатарт бороо орсон."
    phonemes, word_spans, semantic_text = _phonemize_espeak_for_voice_with_spans(
        text, "mn", _identity, None
    )

    output = "".join(phonemes)
    assert semantic_text == text
    assert word_spans
    assert "ɭetə" not in output
    assert not any("\u0400" <= char <= "\u052f" for char in output)
    _assert_supported(phonemes)


def test_pashto_uses_epitran_with_dataset_specific_cleanup() -> None:
    text = "د پښتو ژبې ږغ، ګڼ خلک او ۱۵ ورځې."
    phonemes, word_spans, semantic_text = _phonemize_espeak_for_voice_with_spans(
        text, "ps", _identity, None
    )

    output = "".join(phonemes)
    assert semantic_text == text.replace("،", ",")
    assert word_spans
    assert "ɡ" in output
    assert "ɳ" in output
    assert "15" in output
    assert not any("\u0600" <= char <= "\u06ff" for char in output)
    _assert_supported(phonemes)


def test_assamese_espeak_normalizes_nukta_without_spelling_letter_names() -> None:
    text = "এতিয়া তুমি কেনে আছা? বৰ সোৱাদ জড়িত, পঢ়া ভাল।"
    phonemes, word_spans, semantic_text = _phonemize_espeak_for_voice_with_spans(
        text, "as", _identity, None
    )

    output = "".join(phonemes)
    assert semantic_text == "এতিয়া তুমি কেনে আছা? বৰ সোৱাদ জৰিত, পৰ্হা ভাল."
    assert word_spans
    assert "\u09bc" not in semantic_text
    assert not any("\u0980" <= char <= "\u09ff" for char in output)
    assert "bˈindu" not in output
    assert "ˈakaɾ" not in output
    _assert_supported(phonemes)


def test_language_mapping_selects_custom_g2p() -> None:
    assert _map_cld2_to_espeak("km-KH") == "km"
    assert _map_cld2_to_espeak("lo-LA") == "lo"
    assert _map_cld2_to_espeak("my-MM") == "my"
    assert _map_cld2_to_espeak("th-TH") == "th"
    assert _map_cld2_to_espeak("mn-MN") == "mn"
    assert _map_cld2_to_espeak("ps-AF") == "ps"
    assert _map_cld2_to_espeak("as-IN") == "as"
