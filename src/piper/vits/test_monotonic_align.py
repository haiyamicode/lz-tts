import torch

from .monotonic_align import maximum_path


def test_maximum_path_maps_each_audio_frame_to_one_text_token() -> None:
    audio_length = 7
    text_length = 3
    values = torch.randn(1, audio_length, text_length)
    mask = torch.ones_like(values)

    path = maximum_path(values, mask)

    assert path.shape == values.shape
    assert torch.all((path == 0) | (path == 1))
    assert torch.equal(
        path.sum(dim=2),
        torch.ones(1, audio_length, dtype=path.dtype),
    )
    assert torch.all(path.sum(dim=1) >= 1)
