from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.transcribe import ForceAlignRequest


def test_force_align_times_a_transcript_on_the_clip_it_names() -> None:
    request = ForceAlignRequest(audio_url="https://cdn/a.mp3", text="hello there")

    assert request.text == "hello there"
    assert request.ssml is None
    assert request.voice_id is None
    assert request.language_override is False


def test_force_align_times_a_take_by_the_ssml_it_was_spoken_from() -> None:
    request = ForceAlignRequest(
        audio_base64="SUQzBAA=",
        ssml="<speak>hello there</speak>",
        voice_id="msa.en-US.Serena",
        language="en-US",
        language_override=True,
    )

    assert request.ssml == "<speak>hello there</speak>"
    assert request.voice_id == "msa.en-US.Serena"
    assert request.text is None


@pytest.mark.parametrize(
    "says",
    [{}, {"text": "hello there", "ssml": "<speak>hello there</speak>"}],
)
def test_force_align_takes_one_reading_of_the_clip_never_two_or_none(
    says: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        ForceAlignRequest(audio_url="https://cdn/a.mp3", **says)
