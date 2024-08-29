"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional, Set, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_vit_msn import ViTMSNConfig

""" PyTorch ViT MSN (masked siamese network) model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
class ViTMSNEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """
    def __init__(self, config: ViTMSNConfig, use_mask_token: bool = ...) -> None:
        ...

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        ...

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor] = ..., interpolate_pos_encoding: bool = ...) -> torch.Tensor:
        ...



class ViTMSNPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    def __init__(self, config) -> None:
        ...

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = ...) -> torch.Tensor:
        ...



class ViTMSNSelfAttention(nn.Module):
    def __init__(self, config: ViTMSNConfig) -> None:
        ...

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, hidden_states, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        ...



class ViTMSNSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTMSNLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """
    def __init__(self, config: ViTMSNConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...



class ViTMSNAttention(nn.Module):
    def __init__(self, config: ViTMSNConfig) -> None:
        ...

    def prune_heads(self, heads: Set[int]) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        ...



class ViTMSNIntermediate(nn.Module):
    def __init__(self, config: ViTMSNConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



class ViTMSNOutput(nn.Module):
    def __init__(self, config: ViTMSNConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...



class ViTMSNLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""
    def __init__(self, config: ViTMSNConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        ...



class ViTMSNEncoder(nn.Module):
    def __init__(self, config: ViTMSNConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...) -> Union[tuple, BaseModelOutput]:
        ...



class ViTMSNPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ViTMSNConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...


VIT_MSN_START_DOCSTRING = ...
VIT_MSN_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare ViTMSN Model outputting raw hidden-states without any specific head on top.", VIT_MSN_START_DOCSTRING)
class ViTMSNModel(ViTMSNPreTrainedModel):
    def __init__(self, config: ViTMSNConfig, use_mask_token: bool = ...) -> None:
        ...

    def get_input_embeddings(self) -> ViTMSNPatchEmbeddings:
        ...

    @add_start_docstrings_to_model_forward(VIT_MSN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.Tensor] = ..., bool_masked_pos: Optional[torch.BoolTensor] = ..., head_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., interpolate_pos_encoding: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[tuple, BaseModelOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ViTMSNModel
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
        >>> model = ViTMSNModel.from_pretrained("facebook/vit-msn-small")
        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        ...



@add_start_docstrings("""
    ViTMSN Model with an image classification head on top e.g. for ImageNet.
    """, VIT_MSN_START_DOCSTRING)
class ViTMSNForImageClassification(ViTMSNPreTrainedModel):
    def __init__(self, config: ViTMSNConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(VIT_MSN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., interpolate_pos_encoding: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[tuple, ImageClassifierOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ViTMSNForImageClassification
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> torch.manual_seed(2)  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
        >>> model = ViTMSNForImageClassification.from_pretrained("facebook/vit-msn-small")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits
        >>> # model predicts one of the 1000 ImageNet classes
        >>> predicted_label = logits.argmax(-1).item()
        >>> print(model.config.id2label[predicted_label])
        tusker
        ```"""
        ...
