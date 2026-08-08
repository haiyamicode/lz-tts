from __future__ import annotations

import pytest

from .ssml import BreakOperation, PronunciationOperation, parse_ssml


def test_parse_ssml_records_decoded_text_spans_and_operations() -> None:
    document = parse_ssml(
        '<speak>  Tom &amp; <phoneme alphabet="ipa" ph="sˈɜːɹʃə">Saoirse</phoneme>'
        '<break time="1500ms"/> left.  </speak>'
    )

    assert document.text == "Tom & Saoirse left."
    assert document.operations == (
        PronunciationOperation(6, 13, "ipa", "sˈɜːɹʃə"),
        BreakOperation(13, 1.5),
    )


def test_parse_ssml_accepts_standard_namespace_and_xml_character_references() -> None:
    document = parse_ssml(
        '<speak xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        '&lt;hello&gt; &quot;Tom&apos;s&quot; &amp; Jerry'
        "</speak>"
    )
    assert document.text == '<hello> "Tom\'s" & Jerry'
    assert document.operations == ()


@pytest.mark.parametrize(
    "ssml, message",
    [
        ("<speak>Tom & Jerry</speak>", "Invalid SSML"),
        ("<speak><break time='2'/></speak>", "'ms' or 's'"),
        ("<speak><break time='0s'/>hello</speak>", "greater than 0"),
        ("<speak>hello<break strength='strong'/></speak>", "Unsupported SSML <break> attribute"),
        ("<speak><phoneme alphabet='x-sampa' ph='x'>word</phoneme></speak>", "alphabet='ipa'"),
        ("<speak><phoneme alphabet='ipa' ph=''>word</phoneme></speak>", "non-empty 'ph'"),
        ("<speak><phoneme alphabet='ipa' ph='wɜːd' ignored='x'>word</phoneme></speak>", "Unsupported SSML <phoneme> attribute"),
        ("<speak><prosody rate='slow'>hello</prosody></speak>", "Unsupported SSML element"),
        ("<voice>hello</voice>", "root element"),
    ],
)
def test_parse_ssml_rejects_invalid_or_unsupported_input(ssml: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_ssml(ssml)


def test_parse_ssml_rejects_entity_declarations() -> None:
    with pytest.raises(ValueError, match="Invalid SSML"):
        parse_ssml('<!DOCTYPE speak [<!ENTITY x "expanded">]><speak>&x;</speak>')


def test_break_defaults_to_lazybird_one_second() -> None:
    document = parse_ssml("<speak>Hello<break/>world</speak>")
    assert document.text == "Hello world"
    assert document.breaks == (BreakOperation(5, 1.0),)


def test_break_only_adds_a_boundary_when_words_would_be_joined() -> None:
    document = parse_ssml(
        "<speak>Hello,<break time='200ms'/>world and hello<break time='300ms'/>.</speak>"
    )

    assert document.text == "Hello,world and hello."
    assert document.breaks == (BreakOperation(6, 0.2), BreakOperation(21, 0.3))


@pytest.mark.parametrize(
    ("ssml", "expected_text", "position"),
    [
        ("<speak>สวัสดี<break/>โลก</speak>", "สวัสดี โลก", 6),
        ("<speak>ສະບາຍດີ<break/>ໂລກ</speak>", "ສະບາຍດີ ໂລກ", 7),
        ("<speak>សួស្តី<break/>ពិភពលោក</speak>", "សួស្តី ពិភពលោក", 6),
        ("<speak>မင်္ဂလာပါ<break/>ကမ္ဘာ</speak>", "မင်္ဂလာပါ ကမ္ဘာ", 9),
    ],
)
def test_break_adds_an_explicit_boundary_across_scripts(
    ssml: str,
    expected_text: str,
    position: int,
) -> None:
    document = parse_ssml(ssml)

    assert document.text == expected_text
    assert document.breaks == (BreakOperation(position, 1.0),)


def test_break_boundary_space_shifts_adjacent_phoneme_spans_correctly() -> None:
    document = parse_ssml(
        '<speak><phoneme alphabet="ipa" ph="fuː">foo</phoneme>'
        '<break time="200ms"/><break time="300ms"/>'
        '<phoneme alphabet="ipa" ph="bɑː">bar</phoneme></speak>'
    )

    assert document.text == "foo bar"
    assert document.breaks == (BreakOperation(3, 0.2), BreakOperation(3, 0.3))
    assert document.pronunciations == (
        PronunciationOperation(0, 3, "ipa", "fuː"),
        PronunciationOperation(4, 7, "ipa", "bɑː"),
    )


def test_unicode_offsets_follow_decoded_python_text() -> None:
    document = parse_ssml(
        '<speak>今天🌏，<phoneme alphabet="ipa" ph="tɕʰuŋ˥˩tɕʰīŋ˥˩">'
        '重庆</phoneme><break time="250ms"/>很热闹 &amp; 有趣。</speak>'
    )
    assert document.text == "今天🌏，重庆 很热闹 & 有趣。"
    assert document.pronunciations == (
        PronunciationOperation(4, 6, "ipa", "tɕʰuŋ˥˩tɕʰīŋ˥˩"),
    )
    assert document.breaks == (BreakOperation(6, 0.25),)


def test_xml_escapes_are_decoded_in_phoneme_text_and_attribute() -> None:
    document = parse_ssml(
        '<speak>Say <phoneme alphabet="ipa" ph="a&amp;b&quot;c">A&amp;B</phoneme>.</speak>'
    )
    assert document.text == "Say A&B."
    assert document.pronunciations == (PronunciationOperation(4, 7, "ipa", 'a&b"c'),)
