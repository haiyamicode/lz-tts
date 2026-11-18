"""Server config: YAML fragments composed into the one mapping the worker validates as ``ServerConfig``.

- ``LZAI_SERVER_CONFIG``: one file or a comma-separated list, merged left to right (later wins). Unset =
  ``local/server.yml`` (a checkout's untracked config; ``local/server.json`` while it has not been converted),
  or no file at all (code defaults). A file named here that does not exist is an error.
- ``extends:`` in a file (a path or a list, relative to that file) names its bases: they merge first, in order,
  then the file itself. A base may extend others; a cycle is an error.
- ``LZAI_SERVER_CONFIG_YAML``: an inline YAML mapping merged last, on top of those files (no ``extends``). A
  deployment keeps its own settings in its manifest this way (``{media: {music: false}, load: {mode: on-demand}}``);
  only tracked bases ship in the image.
- Merge: a mapping merges into a mapping key by key, recursively; anything else (lists, scalars, ``null``)
  replaces. A mapping tagged ``!replace`` replaces the base's mapping instead of merging into it.

JSON is YAML, so a ``server.json`` loads unchanged. Prints the composed config:
``python -m src.server_config [file,file...]``.

Imports only PyYAML: ``src.model_sets`` reads the config before ``uv sync`` (``uv run --with pyyaml``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

ENV = "LZAI_SERVER_CONFIG"
INLINE_ENV = "LZAI_SERVER_CONFIG_YAML"
DEFAULT_PATH = Path("local/server.yml")
LEGACY_PATH = Path("local/server.json")
EXTENDS = "extends"


class Replace(dict):
    """A ``!replace`` mapping: replaces the base's value instead of merging into it."""


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor("!replace", lambda loader, node: Replace(loader.construct_mapping(node, deep=True)))


def config_paths(value: str | os.PathLike | list | None = None) -> list[Path]:
    """The files named by ``value`` (else ``LZAI_SERVER_CONFIG``, else the checkout default), in merge order."""
    if value is None:
        value = os.environ.get(ENV, "").strip() or None
    if value is None:
        for default in (DEFAULT_PATH, LEGACY_PATH):
            if default.exists():
                return [default]
        return []
    if isinstance(value, (list, tuple)):
        return [Path(item) for item in value]
    return [Path(item.strip()) for item in str(value).split(",") if item.strip()]


def merge(base: dict, override: dict) -> dict:
    """``override`` merged into ``base`` (neither changed): mappings merge, anything else replaces."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and not isinstance(value, Replace) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = _plain(value)
    return merged


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _parse(text: str, source: str) -> dict:
    data = yaml.load(text, Loader=_Loader)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"server config {source}: the top level must be a mapping, got {type(data).__name__}")
    return data


def _read(path: Path) -> dict:
    return _parse(path.read_text(encoding="utf-8"), str(path))


def _compose(path: Path, stack: tuple[Path, ...], files: list[Path]) -> dict:
    resolved = path.resolve()
    if resolved in stack:
        chain = " -> ".join(str(p) for p in (*stack, resolved))
        raise ValueError(f"server config extends itself: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"server config {path} not found" + (f" (extended by {stack[-1]})" if stack else ""))
    data = _read(path)
    bases = data.pop(EXTENDS, None) or []
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or not all(isinstance(base, str) for base in bases):
        raise ValueError(f"server config {path}: {EXTENDS} must be a path or a list of paths")
    merged: dict = {}
    for base in bases:
        merged = merge(merged, _compose(Path(os.path.normpath(path.parent / base)), (*stack, resolved), files))
    files.append(path)
    return merge(merged, data)


def load(value: str | os.PathLike | list | None = None) -> tuple[dict, list[Path]]:
    """The composed config of ``value`` (see :func:`config_paths`) and every file read, in merge order. With no
    ``value`` (the environment's config), ``LZAI_SERVER_CONFIG_YAML`` merges last, listed as ``$LZAI_SERVER_CONFIG_YAML``."""
    merged: dict = {}
    files: list[Path] = []
    for path in config_paths(value):
        merged = merge(merged, _compose(path, (), files))
    inline = os.environ.get(INLINE_ENV, "").strip() if value is None else ""
    if inline:
        data = _parse(inline, f"${INLINE_ENV}")
        if EXTENDS in data:
            raise ValueError(f"${INLINE_ENV} cannot use {EXTENDS}: name the bases in ${ENV}")
        merged = merge(merged, data)
        files.append(Path(f"${INLINE_ENV}"))
    return merged, files


class _Dumper(yaml.SafeDumper):
    pass


def _represent_list(dumper: yaml.SafeDumper, data: list):
    flow = all(not isinstance(item, (dict, list)) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _represent_list)


def dump(data: dict) -> str:
    """``data`` as YAML in file order, lists of scalars on one line."""
    return yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=120)


def main(argv: list[str]) -> None:
    data, files = load(",".join(argv) if argv else None)
    print("# composed from: " + (", ".join(str(path) for path in files) or "no file (code defaults)"))
    print(dump(data), end="")


if __name__ == "__main__":
    main(sys.argv[1:])
