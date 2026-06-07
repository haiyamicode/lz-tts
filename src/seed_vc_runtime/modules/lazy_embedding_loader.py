"""
HDF5-based embedding loader for memory-efficient access to large embedding files.

Uses compression and lazy loading to minimize RAM usage.
"""
from pathlib import Path
from typing import Optional, Any
import torch
import numpy as np
from functools import lru_cache
import io

try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False


class HDF5EmbeddingLoader:
    """
    HDF5-based embedding loader with compression and lazy loading.

    This is the recommended approach for large embedding files:
    - Compressed storage (2-3x reduction with light compression)
    - True lazy loading (only loads requested embeddings)
    - Fast random access via memory mapping
    - Single file format
    - Minimal CPU overhead during reads
    """

    def __init__(self, hdf5_path: Path, cache_size: int = 256):
        """
        Initialize the HDF5 embedding loader.

        Args:
            hdf5_path: Path to the .h5 file
            cache_size: Number of embeddings to keep in LRU cache (default: 256)
        """
        if not HAS_HDF5:
            raise ImportError("h5py is required for HDF5 embedding loader. Install with: uv add h5py")

        self.hdf5_path = hdf5_path
        self.cache_size = cache_size
        self._file = None
        self._keys = set()

        if self.hdf5_path.exists():
            # Open in read-only mode with memory mapping
            self._file = h5py.File(str(hdf5_path), 'r')
            self._keys = set(self._file.keys())
            print(f"Opened HDF5 file with {len(self._keys)} embeddings from {hdf5_path}")

            # Get file size for info
            file_size_gb = hdf5_path.stat().st_size / (1024 ** 3)
            print(f"HDF5 file size: {file_size_gb:.2f}GB (compressed)")
        else:
            print(f"No HDF5 file found at {hdf5_path}")

        # Set up LRU cache for the get method
        self._cached_get = lru_cache(maxsize=cache_size)(self._load_embedding)

    def _load_embedding(self, key: str) -> Optional[Any]:
        """
        Load an individual embedding from HDF5.

        Args:
            key: Embedding key

        Returns:
            The loaded embedding (tensor or dict of tensors) or None if not found
        """
        if self._file is None or key not in self._file:
            return None

        item = self._file[key]

        # Check if it's a group (dict of tensors)
        if isinstance(item, h5py.Group):
            # Reconstruct the dict
            result = {}
            for sub_key in item.keys():
                dataset = item[sub_key]
                if dataset.dtype == 'object' or dataset.dtype.kind == 'S':
                    # Pickled data
                    data = dataset[()]
                    if isinstance(data, bytes):
                        buffer = io.BytesIO(data)
                        result[sub_key] = torch.load(buffer, map_location="cpu", weights_only=False)
                    else:
                        result[sub_key] = data
                else:
                    # Numpy array - convert to tensor
                    result[sub_key] = torch.from_numpy(dataset[:])
            return result
        else:
            # It's a dataset (single tensor or pickled data)
            if item.dtype == 'object' or item.dtype.kind == 'S':
                # Stored as pickled bytes
                data = item[()]
                if isinstance(data, bytes):
                    buffer = io.BytesIO(data)
                    return torch.load(buffer, map_location="cpu", weights_only=False)
                return data
            else:
                # Stored as raw numpy array - convert to tensor
                return torch.from_numpy(item[:])

    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Get an embedding by key. Uses LRU cache for frequently accessed embeddings.

        Args:
            key: Embedding key to retrieve
            default: Default value if key not found

        Returns:
            The embedding if found, otherwise default value
        """
        if key not in self._keys:
            return default

        embedding = self._cached_get(key)
        return embedding if embedding is not None else default

    def keys(self):
        """Return an iterator over embedding keys."""
        return iter(self._keys)

    def __len__(self):
        """Return the number of embeddings."""
        return len(self._keys)

    def __contains__(self, key: str):
        """Check if a key exists in the embeddings."""
        return key in self._keys

    def close(self):
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()


def convert_monolithic_to_hdf5(
    input_path: Path,
    output_path: Path,
    compression: str = "gzip",
    compression_level: int = 1
) -> None:
    """
    Convert a monolithic embeddings.pt file to HDF5 format with compression.

    Args:
        input_path: Path to the monolithic .pt file
        output_path: Path for the output .h5 file
        compression: Compression algorithm ('gzip', 'lzf', or None)
        compression_level: Compression level (0-9 for gzip, default 1 for minimal CPU overhead)
    """
    if not HAS_HDF5:
        raise ImportError("h5py is required. Install with: uv add h5py")

    print(f"Converting {input_path} to HDF5 format...")
    print(f"Output: {output_path}")
    print(f"Compression: {compression} (level {compression_level})")
    print()

    # Get input file size
    input_size_gb = input_path.stat().st_size / (1024 ** 3)
    print(f"Input file size: {input_size_gb:.2f}GB")
    print()

    # Load the monolithic file
    print("Loading embeddings from disk (this will take a moment)...")
    embeddings = torch.load(str(input_path), map_location="cpu", weights_only=False)
    keys = list(embeddings.keys())
    print(f"Loaded {len(keys)} embeddings")
    print()

    # Create HDF5 file
    print("Creating HDF5 file...")
    with h5py.File(str(output_path), 'w') as h5f:
        for i, key in enumerate(keys):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(keys)} ({(i+1)/len(keys)*100:.1f}%)")

            embedding = embeddings[key]

            # Handle different embedding types
            if isinstance(embedding, torch.Tensor):
                # Simple tensor embedding
                numpy_data = embedding.cpu().numpy()
                h5f.create_dataset(
                    key,
                    data=numpy_data,
                    compression=compression,
                    compression_opts=compression_level if compression == "gzip" else None
                )
            elif isinstance(embedding, dict):
                # Dict of tensors - create a group and store each tensor separately
                group = h5f.create_group(key)
                for sub_key, tensor in embedding.items():
                    if isinstance(tensor, torch.Tensor):
                        numpy_data = tensor.cpu().numpy()
                        group.create_dataset(
                            sub_key,
                            data=numpy_data,
                            compression=compression,
                            compression_opts=compression_level if compression == "gzip" else None
                        )
                    else:
                        # Non-tensor in dict - pickle it
                        buffer = io.BytesIO()
                        torch.save(tensor, buffer)
                        group.create_dataset(sub_key, data=np.void(buffer.getvalue()))
            else:
                # Other types - pickle and store as bytes
                buffer = io.BytesIO()
                torch.save(embedding, buffer)
                bytes_data = buffer.getvalue()
                h5f.create_dataset(
                    key,
                    data=np.void(bytes_data),
                    dtype=h5py.special_dtype(vlen=np.uint8)
                )

    # Get output file size
    output_size_gb = output_path.stat().st_size / (1024 ** 3)
    compression_ratio = input_size_gb / output_size_gb if output_size_gb > 0 else 0

    print()
    print("=" * 70)
    print("Conversion Complete!")
    print("=" * 70)
    print(f"Output file: {output_path}")
    print(f"Output size: {output_size_gb:.2f}GB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Space saved: {input_size_gb - output_size_gb:.2f}GB")
    print()
    print(f"Memory savings during inference: ~{input_size_gb:.1f}GB RAM freed!")
