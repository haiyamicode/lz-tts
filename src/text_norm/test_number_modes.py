"""Numbers read in the mode picked for each one (cardinal, year, ordinal, digits)."""

import pytest

from src.piper.number_modes import DEFAULT_CHECKPOINT
from src.text_norm.spoken import verbalize_spoken_text


def picked(*modes):
    """The modes a classifier picked, in span order."""

    def number_modes(text, spans):
        assert len(spans) == len(modes), [text[start:end] for start, end in spans]
        return list(modes)

    return number_modes


@pytest.mark.parametrize(
    "language, text, modes, spoken",
    [
        ("en", "In 1982 we had 1982 apples.", ("year", "cardinal"),
         "In nineteen eighty-two we had one thousand, nine hundred and eighty-two apples."),
        # Numbered things of 3-4 digits are read in pairs, which is the year reading.
        ("en", "He flew a Boeing 747 to room 1205.", ("year", "year"),
         "He flew a Boeing seven forty-seven to room twelve oh-five."),
        ("en", "My PIN is 4821.", ("digits",), "My PIN is four eight two one."),
        ("en", "We met on May 3, 1999.", ("ordinal", "year"), "We met on May third, nineteen ninety-nine."),
        ("en", "Say 1. Say 2.", ("cardinal", "cardinal"), "Say one. Say two."),
        ("vi", "Em đạt giải 1, bạn nhận giải 2.", ("cardinal", "cardinal"), "Em đạt giải nhất, bạn nhận giải nhì."),
        ("vi", "Giải 2 bài toán.", ("cardinal",), "Giải hai bài toán."),
        ("vi", "Thứ 4 tôi về, tháng 4 nghỉ, đội xếp hạng 2.", ("cardinal", "cardinal", "cardinal"),
         "Thứ tư tôi về, tháng tư nghỉ, đội xếp hạng nhì."),
        ("vi", "Năm 2024.", ("year",), "Năm hai nghìn không trăm hai mươi tư."),
        ("de", "Am 3. Oktober 1990.", ("ordinal", "year"), "Am dritten Oktober neunzehnhundertneunzig."),
        ("fr", "Appelez le 18 en 1999.", ("cardinal", "year"),
         "Appelez le dix-huit en mille neuf cent quatre-vingt-dix-neuf."),
        ("es", "En 1985 llamé al 112.", ("year", "digits"),
         "En mil novecientos ochenta y cinco llamé al uno uno dos."),
        ("it", "Chiama il 118, sono 2118 euro.", ("cardinal", "cardinal"),
         "Chiama il centodiciotto, sono duemilacentodiciotto euro."),
        ("pt", "Em 1985 liguei 190.", ("year", "cardinal"),
         "Em mil novecentos e oitenta e cinco liguei cento e noventa."),
        ("zh", "他1949年住在302房间。", ("year", "digits"), "他一九四九年住在三零二房间。"),
        ("zh", "坐302路公交。", ("cardinal",), "坐三百零二路公交。"),
        ("ja", "110番に電話して。", ("cardinal",), "百十番に電話して。"),
        # A German ordinal is written with a period; without one it is a cardinal.
        ("de", "Um 20 Uhr im 3. Stock.", ("ordinal", "ordinal"), "Um zwanzig Uhr im dritten Stock."),
        # A degree sign after an ordinal is the ordinal indicator; otherwise degrees.
        ("it", "L'ufficio è al 2° piano, fuori ci sono 30°.", ("ordinal", "cardinal"),
         "L'ufficio è al secondo piano, fuori ci sono trenta gradi."),
        ("es", "Vivo en el 1° piso.", ("ordinal",), "Vivo en el primer piso."),
        ("es", "Vivo en el 1° de la calle.", ("ordinal",), "Vivo en el primero de la calle."),
        ("de", "Es sind 30° draußen.", ("cardinal",), "Es sind dreißig Grad draußen."),
        ("en", "It is 1° outside.", ("cardinal",), "It is one degree outside."),
        ("fr", "Le billet coûte 12,50 euros.", ("cardinal",), "Le billet coûte douze virgule cinquante euros."),
        ("fr", "Il fait 3,05 degrés.", ("cardinal",), "Il fait trois virgule zéro cinq degrés."),
        ("en", "It costs 3.14 dollars.", ("cardinal",), "It costs three point one four dollars."),
        ("ko", "비밀번호는 4096입니다.", ("digits",), "비밀번호는 사 공 구 육입니다."),
        ("ko", "010-2345-6789로 연락 주세요.", ("digits", "digits", "digits"),
         "공 일 공, 이 삼 사 오, 육 칠 팔 구로 연락 주세요."),
        ("ja", "090-1234-5678に電話してください。", ("digits", "digits", "digits"),
         "零九零、一二三四、五六七八に電話してください。"),
        # Digit groups read digit by digit throughout once any of them does.
        ("pt", "Ligue para 11 98765 4321 amanhã.", ("cardinal", "digits", "cardinal"),
         "Ligue para um um, nove oito sete seis cinco, quatro três dois um amanhã."),
    ],
)
def test_numbers_read_in_the_picked_mode(language, text, modes, spoken):
    assert verbalize_spoken_text(text, language, picked(*modes)) == spoken


def test_italian_ordinal_indicator_keeps_diciotto():
    assert verbalize_spoken_text("Il 118º posto.", "it", picked()) == "Il centodiciottesimo posto."


@pytest.mark.parametrize(
    "language, text, spoken",
    [
        ("es", "Vivo en el 4º piso.", "Vivo en el cuarto piso."),
        ("es", "La 1ª planta.", "La primera planta."),
        ("pt", "A aula começa às 14h.", "A aula começa às catorze horas."),
        ("fr", "Il a roulé 3h.", "Il a roulé trois heures."),
    ],
)
def test_fixed_forms_need_no_mode(language, text, spoken):
    assert verbalize_spoken_text(text, language, picked()) == spoken


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="No trained number mode checkpoint")
def test_classifier_picks_the_reading_from_context():
    from src.piper.number_modes import NumberModeResolver

    number_modes = NumberModeResolver(device="cpu")
    for language, text, spoken in (
        ("en", "In 1982 we had 1982 apples.",
         "In nineteen eighty-two we had one thousand, nine hundred and eighty-two apples."),
        ("en", "My PIN is 4821.", "My PIN is four eight two one."),
        ("de", "Er wurde 1949 geboren.", "Er wurde neunzehnhundertneunundvierzig geboren."),
        ("zh", "他1949年出生。", "他一九四九年出生。"),
    ):
        assert verbalize_spoken_text(text, language, number_modes) == spoken
