import os
from dataclasses import dataclass
from pydantic import BaseModel
from typing import Generic, TypeVar, List, Any

import torch

T = TypeVar("T", bound=BaseModel)


def resolve_torch_dtype(name: str) -> torch.dtype:
    """Resolve the model precision declared in a VoxCPM model config."""
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported VoxCPM dtype {name!r}; expected 'bfloat16' or 'float16'"
        ) from exc


def resolve_model_dtype(name: str, device_index: int) -> str:
    """Resolve automatic precision according to native CUDA dtype support."""
    if name != "auto":
        resolve_torch_dtype(name)
        return name

    major, _minor = torch.cuda.get_device_capability(device_index)
    return "bfloat16" if major >= 8 else "float16"


@dataclass
class Config(Generic[T]):
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    enable_prefix_caching: bool = True

    model_config: T | None = None
    devices: List[int] | None = None
    lora_config: Any = None  # Optional[LoRAConfig]
    ipa_adapter_path: str | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.max_num_batched_tokens >= self.max_model_len
