import numpy as np
import torch

try:
    from src.starling.utils.monotonic_align.core import maximum_path_c
except ModuleNotFoundError:
    maximum_path_c = None


def _maximum_path_python(path, value, t_xs, t_ys, max_neg_val=-1e9):
    for batch_idx in range(value.shape[0]):
        t_x = int(t_xs[batch_idx])
        t_y = int(t_ys[batch_idx])
        index = t_x - 1

        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                if x == y:
                    v_cur = max_neg_val
                else:
                    v_cur = value[batch_idx, x, y - 1]
                if x == 0:
                    v_prev = 0.0 if y == 0 else max_neg_val
                else:
                    v_prev = value[batch_idx, x - 1, y - 1]
                value[batch_idx, x, y] = max(v_cur, v_prev) + value[batch_idx, x, y]

        for y in range(t_y - 1, -1, -1):
            path[batch_idx, index, y] = 1
            if index != 0 and (index == y or value[batch_idx, index, y - 1] < value[batch_idx, index - 1, y - 1]):
                index -= 1


def maximum_path(value, mask):
    """Cython optimised version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    value = value * mask
    device = value.device
    dtype = value.dtype
    value = value.data.cpu().numpy().astype(np.float32)
    path = np.zeros_like(value).astype(np.int32)
    mask = mask.data.cpu().numpy()

    t_x_max = mask.sum(1)[:, 0].astype(np.int32)
    t_y_max = mask.sum(2)[:, 0].astype(np.int32)
    if maximum_path_c is not None:
        maximum_path_c(path, value, t_x_max, t_y_max)
    else:
        _maximum_path_python(path, value, t_x_max, t_y_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)
