"""
This type stub file was generated by pyright.
"""

from transformers.utils import is_torch_available
from transformers.utils.generic import ExplicitEnum
from ...processing_utils import ProcessorMixin

"""Processor class for MGP-STR."""
if is_torch_available():
    ...
class DecodeType(ExplicitEnum):
    CHARACTER = ...
    BPE = ...
    WORDPIECE = ...


SUPPORTED_ANNOTATION_FORMATS = ...
class MgpstrProcessor(ProcessorMixin):
    r"""
    Constructs a MGP-STR processor which wraps an image processor and MGP-STR tokenizers into a single

    [`MgpstrProcessor`] offers all the functionalities of `ViTImageProcessor`] and [`MgpstrTokenizer`]. See the
    [`~MgpstrProcessor.__call__`] and [`~MgpstrProcessor.batch_decode`] for more information.

    Args:
        image_processor (`ViTImageProcessor`, *optional*):
            An instance of `ViTImageProcessor`. The image processor is a required input.
        tokenizer ([`MgpstrTokenizer`], *optional*):
            The tokenizer is a required input.
    """
    attributes = ...
    image_processor_class = ...
    char_tokenizer_class = ...
    def __init__(self, image_processor=..., tokenizer=..., **kwargs) -> None:
        ...

    def __call__(self, text=..., images=..., return_tensors=..., **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to ViTImageProcessor's
        [`~ViTImageProcessor.__call__`] and returns its output. This method also forwards the `text` and `kwargs`
        arguments to MgpstrTokenizer's [`~MgpstrTokenizer.__call__`] if `text` is not `None` to encode the text. Please
        refer to the doctsring of the above methods for more information.
        """
        ...

    def batch_decode(self, sequences): # -> dict[Any, Any]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`torch.Tensor`):
                List of tokenized input ids.

        Returns:
            `Dict[str, any]`: Dictionary of all the outputs of the decoded results.
                generated_text (`List[str]`): The final results after fusion of char, bpe, and wp. scores
                (`List[float]`): The final scores after fusion of char, bpe, and wp. char_preds (`List[str]`): The list
                of character decoded sentences. bpe_preds (`List[str]`): The list of bpe decoded sentences. wp_preds
                (`List[str]`): The list of wp decoded sentences.

        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        ...

    def char_decode(self, sequences): # -> list[Any]:
        """
        Convert a list of lists of char token ids into a list of strings by calling char tokenizer.

        Args:
            sequences (`torch.Tensor`):
                List of tokenized input ids.
        Returns:
            `List[str]`: The list of char decoded sentences.
        """
        ...

    def bpe_decode(self, sequences):
        """
        Convert a list of lists of bpe token ids into a list of strings by calling bpe tokenizer.

        Args:
            sequences (`torch.Tensor`):
                List of tokenized input ids.
        Returns:
            `List[str]`: The list of bpe decoded sentences.
        """
        ...

    def wp_decode(self, sequences): # -> list[Any]:
        """
        Convert a list of lists of word piece token ids into a list of strings by calling word piece tokenizer.

        Args:
            sequences (`torch.Tensor`):
                List of tokenized input ids.
        Returns:
            `List[str]`: The list of wp decoded sentences.
        """
        ...
