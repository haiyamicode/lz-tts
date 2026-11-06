import re
from typing import List

from transformers import PreTrainedTokenizer

# An IPA span written into the text: "/mˈexiko/". The opening slash follows no word character or slash, so
# "and/or" and URLs are not spans.
IPA_SPAN = re.compile(r"(?<![\w/])/[^/\n]+/(?![\w/])")
BYTE_TOKEN = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def mask_multichar_chinese_tokens(tokenizer: PreTrainedTokenizer):
    """Char-split wrapper: multi-character Chinese tokens, and every multi-character token inside an IPA span,
    become one token per character. IPA written in Latin letters otherwise merges into the language's own
    subwords (/mˈexiko/ -> "ex" "iko"), which the model reads with that language's spelling rules."""
    multichar_tokens = {
        token for token in tokenizer.vocab.keys() if len(token) >= 2 and all("一" <= c <= "鿿" for c in token)
    }
    vocab = tokenizer.vocab

    class CharTokenizerWrapper:
        def __init__(self, base_tokenizer: PreTrainedTokenizer) -> None:
            self.tokenizer = base_tokenizer
            self.multichar_tokens = multichar_tokens

        def _split(self, text: str) -> List[tuple[str, tuple[int, int]]]:
            """(token, character offsets) pairs of the char-split tokenization."""
            base_tokens = self.tokenizer.tokenize(text)
            encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            if len(base_tokens) != len(encoded["offset_mapping"]):
                raise ValueError(
                    "Base tokenizer offsets do not align with tokens: "
                    f"{len(base_tokens)} tokens vs {len(encoded['offset_mapping'])} offsets"
                )
            spans = [m.span() for m in IPA_SPAN.finditer(text)]
            out: List[tuple[str, tuple[int, int]]] = []
            for token, (start, end) in zip(base_tokens, encoded["offset_mapping"]):
                clean_token = token.replace("▁", "")
                in_ipa = any(start < span_end and end > span_start for span_start, span_end in spans)
                if clean_token in self.multichar_tokens:
                    # The token's characters always end at `end`; any "▁" prefix (which may or may not consume a
                    # character in the offset span) sits on the left.
                    first = end - len(clean_token)
                    out.extend((c, (first + i, first + i + 1)) for i, c in enumerate(clean_token))
                elif (in_ipa and len(clean_token) >= 2 and not BYTE_TOKEN.match(token)
                      and all(c in vocab for c in clean_token)):
                    first = end - len(clean_token)
                    prefix = token[: len(token) - len(token.lstrip("▁"))]
                    if prefix:
                        out.append((prefix, (start, first)))
                    out.extend((c, (first + i, first + i + 1)) for i, c in enumerate(clean_token))
                else:
                    out.append((token, (start, end)))
            return out

        def tokenize(self, text: str, **kwargs) -> List[str]:
            if not isinstance(text, str):
                raise TypeError(f"Expected string input, got {type(text)}")
            return [token for token, _ in self._split(text)]

        def __call__(self, text: str, **kwargs) -> List[int]:
            try:
                tokens = self.tokenize(text, **kwargs)
                return self.tokenizer.convert_tokens_to_ids(tokens)
            except Exception as e:
                raise ValueError(f"Tokenization failed: {str(e)}") from e

        def tokenize_with_offsets(self, text: str) -> List[tuple[int, int]]:
            """Character offsets aligned 1:1 with the char-split tokenization (the token space used for model
            prefill), expanded from the base tokenizer's offset mapping."""
            return [offsets for _, offsets in self._split(text)]

    return CharTokenizerWrapper(tokenizer)
