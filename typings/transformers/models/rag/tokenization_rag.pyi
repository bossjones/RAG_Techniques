"""
This type stub file was generated by pyright.
"""

from typing import List, Optional
from ...tokenization_utils_base import BatchEncoding

"""Tokenization classes for RAG."""
logger = ...
class RagTokenizer:
    def __init__(self, question_encoder, generator) -> None:
        ...

    def save_pretrained(self, save_directory): # -> None:
        ...

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs): # -> Self:
        ...

    def __call__(self, *args, **kwargs):
        ...

    def batch_decode(self, *args, **kwargs):
        ...

    def decode(self, *args, **kwargs):
        ...

    def prepare_seq2seq_batch(self, src_texts: List[str], tgt_texts: Optional[List[str]] = ..., max_length: Optional[int] = ..., max_target_length: Optional[int] = ..., padding: str = ..., return_tensors: str = ..., truncation: bool = ..., **kwargs) -> BatchEncoding:
        ...
