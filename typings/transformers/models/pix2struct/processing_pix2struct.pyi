"""
This type stub file was generated by pyright.
"""

from typing import List, Optional, Union
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType

"""
Processor class for Pix2Struct.
"""
class Pix2StructProcessor(ProcessorMixin):
    r"""
    Constructs a PIX2STRUCT processor which wraps a BERT tokenizer and PIX2STRUCT image processor into a single
    processor.

    [`Pix2StructProcessor`] offers all the functionalities of [`Pix2StructImageProcessor`] and [`T5TokenizerFast`]. See
    the docstring of [`~Pix2StructProcessor.__call__`] and [`~Pix2StructProcessor.decode`] for more information.

    Args:
        image_processor (`Pix2StructImageProcessor`):
            An instance of [`Pix2StructImageProcessor`]. The image processor is a required input.
        tokenizer (Union[`T5TokenizerFast`, `T5Tokenizer`]):
            An instance of ['T5TokenizerFast`] or ['T5Tokenizer`]. The tokenizer is a required input.
    """
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor, tokenizer) -> None:
        ...

    def __call__(self, images=..., text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = ..., add_special_tokens: bool = ..., padding: Union[bool, str, PaddingStrategy] = ..., truncation: Union[bool, str, TruncationStrategy] = ..., max_length: Optional[int] = ..., max_patches: Optional[int] = ..., stride: int = ..., pad_to_multiple_of: Optional[int] = ..., return_attention_mask: Optional[bool] = ..., return_overflowing_tokens: bool = ..., return_special_tokens_mask: bool = ..., return_offsets_mapping: bool = ..., return_token_type_ids: bool = ..., return_length: bool = ..., verbose: bool = ..., return_tensors: Optional[Union[str, TensorType]] = ..., **kwargs) -> BatchEncoding:
        """
        This method uses [`Pix2StructImageProcessor.preprocess`] method to prepare image(s) for the model, and
        [`T5TokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        ...

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Pix2StructTokenizerFast's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        ...

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Pix2StructTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please
        refer to the docstring of this method for more information.
        """
        ...

    @property
    def model_input_names(self): # -> list[Any]:
        ...
