"""
This type stub file was generated by pyright.
"""

from typing import List, Optional, Tuple
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Tokenization classes."""
logger = ...
VOCAB_FILES_NAMES = ...
class CpmTokenizerFast(PreTrainedTokenizerFast):
    """Runs pre-tokenization with Jieba segmentation tool. It is used in CPM models."""
    def __init__(self, vocab_file=..., tokenizer_file=..., do_lower_case=..., remove_space=..., keep_accents=..., bos_token=..., eos_token=..., unk_token=..., sep_token=..., pad_token=..., cls_token=..., mask_token=..., additional_special_tokens=..., **kwargs) -> None:
        """
        Construct a CPM tokenizer. Based on [Jieba](https://pypi.org/project/jieba/) and
        [SentencePiece](https://github.com/google/sentencepiece).

        This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should
        refer to this superclass for more information regarding those methods.

        Args:
            vocab_file (`str`):
                [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
                contains the vocabulary necessary to instantiate a tokenizer.
            do_lower_case (`bool`, *optional*, defaults to `True`):
                Whether to lowercase the input when tokenizing.
            remove_space (`bool`, *optional*, defaults to `True`):
                Whether to strip the text when tokenizing (removing excess spaces before and after the string).
            keep_accents (`bool`, *optional*, defaults to `False`):
                Whether to keep accents when tokenizing.
            bos_token (`str`, *optional*, defaults to `"<s>"`):
                The beginning of sequence token that was used during pretraining. Can be used a sequence classifier
                token.

                <Tip>

                When building a sequence using special tokens, this is not the token that is used for the beginning of
                sequence. The token used is the `cls_token`.

                </Tip>

            eos_token (`str`, *optional*, defaults to `"</s>"`):
                The end of sequence token.

                <Tip>

                When building a sequence using special tokens, this is not the token that is used for the end of
                sequence. The token used is the `sep_token`.

                </Tip>

            unk_token (`str`, *optional*, defaults to `"<unk>"`):
                The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be
                this token instead.
            sep_token (`str`, *optional*, defaults to `"<sep>"`):
                The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
                for sequence classification or for a text and a question for question answering. It is also used as the
                last token of a sequence built with special tokens.
            pad_token (`str`, *optional*, defaults to `"<pad>"`):
                The token used for padding, for example when batching sequences of different lengths.
            cls_token (`str`, *optional*, defaults to `"<cls>"`):
                The classifier token which is used when doing sequence classification (classification of the whole
                sequence instead of per-token classification). It is the first token of the sequence when built with
                special tokens.
            mask_token (`str`, *optional*, defaults to `"<mask>"`):
                The token used for masking values. This is the token used when training this model with masked language
                modeling. This is the token which the model will try to predict.
            additional_special_tokens (`List[str]`, *optional*, defaults to `["<eop>", "<eod>"]`):
                Additional special tokens used by the tokenizer.

        Attributes:
            sp_model (`SentencePieceProcessor`):
                The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
        """
        ...

    @property
    def can_save_slow_tokenizer(self) -> bool:
        ...

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An XLNet sequence has the following format:

        - single sequence: `X <sep> <cls>`
        - pair of sequences: `A <sep> B <sep> <cls>`

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
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet
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
