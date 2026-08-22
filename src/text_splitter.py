"""Recursive international text splitting shared with Lazybird."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache

import tiktoken


DEFAULT_SEPARATOR_LIST = (
    "\n\n",
    ".\n",
    "\uff0e\n",
    "\u3002\n",
    ". ",
    "? ",
    "! ",
    "\uff0e",
    "\u3002",
    ";",
    ",",
    "\uff0c",
    "\u3001",
    "\n",
    "\t",
    " ",
    "\u200b",
    "",
)

DEFAULT_HARD_SEPARATOR_LIST = ("\n\n", ".\n", "\uff0e\n", "\u3002\n", "\n")


def _utf16_length(text: str) -> int:
    """Match JavaScript's ``String.length`` used by Lazybird."""
    return len(text.encode("utf-16-le")) // 2


@lru_cache(maxsize=1)
def _cl100k_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_cl100k_tokens(text: str) -> int:
    """Count text tokens the same way as Lazybird's token splitter."""
    return len(_cl100k_encoder().encode_ordinary(text))


class RecursiveTextSplitter:
    """Python port of Lazybird's ``RecursiveCharacterTextSplitter``."""

    def __init__(
        self,
        *,
        chunk_size: int,
        soft_chunk_size: int | None = None,
        chunk_overlap: int = 0,
        keep_separator: str | bool = "before",
        separators: Sequence[str] = DEFAULT_SEPARATOR_LIST,
        hard_separators: Sequence[str] = DEFAULT_HARD_SEPARATOR_LIST,
        length_function: Callable[[str], int] = _utf16_length,
        measure_merged_length: bool = False,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("Cannot have chunk_overlap >= chunk_size")
        if soft_chunk_size is not None and soft_chunk_size <= 0:
            raise ValueError("Cannot have soft_chunk_size <= 0")
        if soft_chunk_size is not None and soft_chunk_size > chunk_size:
            raise ValueError("Cannot have soft_chunk_size > chunk_size")
        if keep_separator not in {"before", "after", False}:
            raise ValueError("keep_separator must be 'before', 'after', or False")
        if not separators:
            raise ValueError("separators cannot be empty")

        self.chunk_size = chunk_size
        self.soft_chunk_size = soft_chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_separator = keep_separator
        self.separators = tuple(separators)
        self.hard_separators = frozenset(hard_separators)
        self.length_function = length_function
        self.measure_merged_length = measure_merged_length

    def _split_on_separator(self, text: str, separator: str) -> list[str]:
        if not separator:
            return list(text)
        if not self.keep_separator:
            return [part for part in text.split(separator) if part]

        positions: list[int] = []
        search_start = 0
        while (index := text.find(separator, search_start)) >= 0:
            positions.append(index)
            search_start = index + len(separator)
        boundaries = [
            position + len(separator) if self.keep_separator == "before" else position
            for position in positions
        ]
        parts: list[str] = []
        start = 0
        for boundary in boundaries:
            if boundary > start:
                parts.append(text[start:boundary])
            start = boundary
        if start < len(text):
            parts.append(text[start:])
        return [part for part in parts if part]

    def _join(self, parts: Sequence[str], separator: str) -> str | None:
        text = separator.join(parts).strip()
        return text or None

    def _merge_splits(
        self,
        splits: Sequence[str],
        separator: str,
        split_separator: str,
    ) -> list[str]:
        documents: list[str] = []
        current: list[str] = []
        separator_length = self.length_function(separator)
        total = 0
        use_soft_limit = (
            self.soft_chunk_size is not None and split_separator in self.hard_separators
        )

        def merged_length(parts: Sequence[str]) -> int:
            if not self.measure_merged_length:
                return sum(self.length_function(part) for part in parts) + max(
                    0, len(parts) - 1
                ) * separator_length
            document = self._join(parts, separator)
            return self.length_function(document) if document is not None else 0

        for index, split in enumerate(splits):
            split_length = self.length_function(split)
            candidate_length = merged_length([*current, split])
            if candidate_length > self.chunk_size:
                if current:
                    document = self._join(current, separator)
                    if document is not None:
                        documents.append(document)
                    while current and (
                        merged_length(current) > self.chunk_overlap
                        or (merged_length([*current, split]) > self.chunk_size and total > 0)
                    ):
                        total -= self.length_function(current.pop(0))

            current.append(split)
            total += split_length
            if (
                use_soft_limit
                and index < len(splits) - 1
                and merged_length(current) >= self.soft_chunk_size
            ):
                document = self._join(current, separator)
                if document is not None:
                    documents.append(document)
                current.clear()
                total = 0

        document = self._join(current, separator)
        if document is not None:
            documents.append(document)
        return documents

    def _split_text(self, text: str, separators: Sequence[str]) -> list[str]:
        separator = separators[-1]
        remaining: Sequence[str] | None = None
        for index, candidate in enumerate(separators):
            if not candidate or candidate in text:
                separator = candidate
                remaining = separators[index + 1 :]
                break

        splits = self._split_on_separator(text, separator)
        final_chunks: list[str] = []
        good_splits: list[str] = []
        join_separator = "" if self.keep_separator else separator

        for split in splits:
            if self.length_function(split) < self.chunk_size:
                good_splits.append(split)
                continue

            if good_splits:
                final_chunks.extend(
                    self._merge_splits(good_splits, join_separator, separator)
                )
                good_splits.clear()
            if not remaining:
                final_chunks.append(split)
            else:
                final_chunks.extend(self._split_text(split, remaining))

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, join_separator, separator))
        return [chunk.strip() for chunk in final_chunks]

    def split_text(self, text: str) -> list[str]:
        return self._split_text(text, self.separators)


def split_text(
    text: str,
    max_length: int,
    *,
    soft_max_length: int | None = None,
    hard_separators: Sequence[str] = DEFAULT_HARD_SEPARATOR_LIST,
    length_function: Callable[[str], int] = _utf16_length,
    measure_merged_length: bool = False,
) -> list[str]:
    """Split text with the same defaults and boundary priority as Lazybird."""
    return RecursiveTextSplitter(
        chunk_size=max_length,
        soft_chunk_size=soft_max_length,
        chunk_overlap=0,
        keep_separator="before",
        separators=DEFAULT_SEPARATOR_LIST,
        hard_separators=hard_separators,
        length_function=length_function,
        measure_merged_length=measure_merged_length,
    ).split_text(text)
