import torch
import torch.nn as nn

backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    k = (
        str(tenFlow.device),
        str(tenFlow.dtype),
        tenFlow.shape[2],
        tenFlow.shape[3],
    )
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(
            -1.0, 1.0, tenFlow.shape[3], device=tenFlow.device, dtype=tenFlow.dtype
        ).view(1, 1, 1, tenFlow.shape[3]).expand(1, -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(
            -1.0, 1.0, tenFlow.shape[2], device=tenFlow.device, dtype=tenFlow.dtype
        ).view(1, 1, tenFlow.shape[2], 1).expand(1, -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    base_grid = backwarp_tenGrid[k].expand(tenFlow.shape[0], -1, -1, -1)
    g = (base_grid + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
