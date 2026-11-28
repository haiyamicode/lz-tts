import pytest

from src.text_norm.speakable import speakable_caps_runs, speakable_word_slashes


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("By Leo Tolstoy/Tolstoi", "By Leo Tolstoy, Tolstoi"),
        ("and/or", "and, or"),
        ("he / she", "he, she"),
        ("Tolstoy/Tolstoi/Tolstoj.", "Tolstoy, Tolstoi, Tolstoj."),
        ("12/25", "12/25"),
        ("1/2", "1/2"),
        ("km/h", "km/h"),
        ("120 mg/dL", "120 mg/dL"),
        ("see example.com/about", "see example.com/about"),
        ("https://a.b/c/d", "https://a.b/c/d"),
        ("/usr/bin/env", "/usr/bin/env"),
        ("src/api/server.py", "src/api/server.py"),
    ],
)
def test_word_slashes(text: str, spoken: str) -> None:
    assert speakable_word_slashes(text) == spoken


@pytest.mark.parametrize(
    ("text", "language", "spoken"),
    [
        ("WAR AND PEACE", "en-us", "War And Peace"),
        ("THE FBI AND NASA", "en", "The FBI And Nasa"),
        ("USA TODAY", "en", "USA Today"),
        ("DON'T STOP", "en", "Don't Stop"),
        ("NASA and the FBI", "en", "NASA and the FBI"),
        ("I said NO", "en", "I said NO"),
        ("GUERRE ET PAIX, SNCF", "fr", "Guerre Et Paix, SNCF"),
        ("CHIẾN TRANH VÀ HÒA BÌNH VTV", "vi", "Chiến Tranh Và Hòa Bình VTV"),
    ],
)
def test_caps_runs(text: str, language: str, spoken: str) -> None:
    assert speakable_caps_runs(text, language) == spoken
