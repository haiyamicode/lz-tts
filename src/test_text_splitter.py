from src.text_splitter import RecursiveTextSplitter, count_cl100k_tokens, split_text


def _splitter(
    *,
    chunk_size: int,
    soft_chunk_size: int | None = None,
    hard_separators: tuple[str, ...] = (),
) -> RecursiveTextSplitter:
    return RecursiveTextSplitter(
        chunk_size=chunk_size,
        soft_chunk_size=soft_chunk_size,
        chunk_overlap=0,
        keep_separator=False,
        separators=("\n\n", " ", ""),
        hard_separators=hard_separators,
        length_function=len,
    )


def test_recursive_splitter_matches_lazybird_hard_limit() -> None:
    assert _splitter(chunk_size=20).split_text("aaaa\n\nbbbb\n\ncccc\n\ndddd") == [
        "aaaa\n\nbbbb\n\ncccc",
        "dddd",
    ]


def test_recursive_splitter_matches_lazybird_soft_limit() -> None:
    assert _splitter(
        chunk_size=20,
        soft_chunk_size=8,
        hard_separators=("\n\n",),
    ).split_text("aaaa\n\nbbbb\n\ncccc\n\ndddd") == [
        "aaaa\n\nbbbb",
        "cccc\n\ndddd",
    ]


def test_default_splitter_preserves_international_sentence_punctuation() -> None:
    assert split_text("第一句。第二句。第三句。", 8, length_function=len) == [
        "第一句。第二句。",
        "第三句。",
    ]


def test_default_splitter_uses_character_fallback_only_when_needed() -> None:
    assert split_text("abcdefghij", 4, length_function=len) == ["abcd", "efgh", "ij"]


def test_exact_merge_length_handles_non_additive_tokenizer() -> None:
    def non_additive_length(text: str) -> int:
        return len(text) + (4 if "alpha beta" in text else 0)

    assert split_text(
        "alpha beta gamma",
        12,
        length_function=non_additive_length,
        measure_merged_length=True,
    ) == ["alpha", "beta gamma"]


def test_cl100k_count_treats_special_token_text_as_plain_input() -> None:
    assert count_cl100k_tokens("say <|endoftext|> literally") > 0


def test_token_splitter_uses_soft_and_hard_limits() -> None:
    sentence = "This sentence contains enough ordinary words to consume some tokens."
    text = "\n".join([sentence] * 12)

    chunks = split_text(
        text,
        60,
        soft_max_length=45,
        length_function=count_cl100k_tokens,
        measure_merged_length=True,
    )

    assert len(chunks) > 1
    assert all(count_cl100k_tokens(chunk) <= 60 for chunk in chunks)
    assert any(count_cl100k_tokens(chunk) > 45 for chunk in chunks)
    assert all(chunk.endswith("tokens.") for chunk in chunks)
