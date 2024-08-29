"""
This type stub file was generated by pyright.
"""

from typing import List, Optional, Tuple
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Tokenization classes for RoFormer."""
logger = ...
VOCAB_FILES_NAMES = ...
class RoFormerTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" RoFormer tokenizer (backed by HuggingFace's *tokenizers* library).

    [`RoFormerTokenizerFast`] is almost identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece. There are some difference between them when tokenizing Chinese.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Example:

    ```python
    >>> from transformers import RoFormerTokenizerFast

    >>> tokenizer = RoFormerTokenizerFast.from_pretrained("junnyu/roformer_chinese_base")
    >>> tokenizer.tokenize("今天天气非常好。")
    ['今', '天', '天', '气', '非常', '好', '。']
    ```"""
    vocab_files_names = ...
    slow_tokenizer_class = ...
    def __init__(self, vocab_file=..., tokenizer_file=..., do_lower_case=..., unk_token=..., sep_token=..., pad_token=..., cls_token=..., mask_token=..., tokenize_chinese_chars=..., strip_accents=..., **kwargs) -> None:
        ...

    def __getstate__(self): # -> dict[str, Any]:
        ...

    def __setstate__(self, d): # -> None:
        ...

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=...): # -> list[int | None]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A RoFormer sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        ...

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A RoFormer
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        ...

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...

    def save_pretrained(self, save_directory, legacy_format=..., filename_prefix=..., push_to_hub=..., **kwargs): # -> Tuple[str]:
        ...
