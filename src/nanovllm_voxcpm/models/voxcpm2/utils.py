from typing import List

from transformers import PreTrainedTokenizer


def mask_multichar_chinese_tokens(tokenizer: PreTrainedTokenizer):
    multichar_tokens = {
        token for token in tokenizer.vocab.keys() if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
    }

    class CharTokenizerWrapper:
        def __init__(self, base_tokenizer: PreTrainedTokenizer) -> None:
            self.tokenizer = base_tokenizer
            self.multichar_tokens = multichar_tokens

        def tokenize(self, text: str, **kwargs) -> List[str]:
            if not isinstance(text, str):
                raise TypeError(f"Expected string input, got {type(text)}")

            tokens = self.tokenizer.tokenize(text, **kwargs)
            processed = []
            for token in tokens:
                clean_token = token.replace("▁", "")
                if clean_token in self.multichar_tokens:
                    processed.extend(list(clean_token))
                else:
                    processed.append(token)
            return processed

        def __call__(self, text: str, **kwargs) -> List[int]:
            try:
                tokens = self.tokenize(text, **kwargs)
                return self.tokenizer.convert_tokens_to_ids(tokens)
            except Exception as e:
                raise ValueError(f"Tokenization failed: {str(e)}") from e

        def tokenize_with_offsets(self, text: str) -> List[tuple[int, int]]:
            """Character offsets aligned 1:1 with the char-split tokenization.

            The base tokenizer exposes offset mappings for its own (possibly
            multi-character CJK) tokens; expand those so each token produced by
            :meth:`tokenize` has exactly one entry, matching the token space
            used for model prefill.
            """
            base_tokens = self.tokenizer.tokenize(text)
            encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            if len(base_tokens) != len(encoded["offset_mapping"]):
                raise ValueError(
                    "Base tokenizer offsets do not align with tokens: "
                    f"{len(base_tokens)} tokens vs {len(encoded['offset_mapping'])} offsets"
                )
            offsets: List[tuple[int, int]] = []
            for token, (start, end) in zip(base_tokens, encoded["offset_mapping"]):
                clean_token = token.replace("▁", "")
                if clean_token in self.multichar_tokens:
                    # The token's characters always end at `end`; any "▁"
                    # prefix (which may or may not consume a character in the
                    # offset span) sits on the left.
                    first = end - len(clean_token)
                    for position in range(first, first + len(clean_token)):
                        offsets.append((position, position + 1))
                else:
                    offsets.append((start, end))
            return offsets

    return CharTokenizerWrapper(tokenizer)
