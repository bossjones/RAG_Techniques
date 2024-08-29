"""
This type stub file was generated by pyright.
"""

import torch
from typing import Iterable, Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_pvt import PvtConfig

""" PyTorch PVT model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...
def drop_path(input: torch.Tensor, drop_prob: float = ..., training: bool = ...) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    ...

class PvtDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = ...) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...

    def extra_repr(self) -> str:
        ...



class PvtPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    def __init__(self, config: PvtConfig, image_size: Union[int, Iterable[int]], patch_size: Union[int, Iterable[int]], stride: int, num_channels: int, hidden_size: int, cls_token: bool = ...) -> None:
        ...

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        ...

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        ...



class PvtSelfOutput(nn.Module):
    def __init__(self, config: PvtConfig, hidden_size: int) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



class PvtEfficientSelfAttention(nn.Module):
    """Efficient self-attention mechanism with reduction of the sequence [PvT paper](https://arxiv.org/abs/2102.12122)."""
    def __init__(self, config: PvtConfig, hidden_size: int, num_attention_heads: int, sequences_reduction_ratio: float) -> None:
        ...

    def transpose_for_scores(self, hidden_states: int) -> torch.Tensor:
        ...

    def forward(self, hidden_states: torch.Tensor, height: int, width: int, output_attentions: bool = ...) -> Tuple[torch.Tensor]:
        ...



class PvtAttention(nn.Module):
    def __init__(self, config: PvtConfig, hidden_size: int, num_attention_heads: int, sequences_reduction_ratio: float) -> None:
        ...

    def prune_heads(self, heads): # -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, height: int, width: int, output_attentions: bool = ...) -> Tuple[torch.Tensor]:
        ...



class PvtFFN(nn.Module):
    def __init__(self, config: PvtConfig, in_features: int, hidden_features: Optional[int] = ..., out_features: Optional[int] = ...) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



class PvtLayer(nn.Module):
    def __init__(self, config: PvtConfig, hidden_size: int, num_attention_heads: int, drop_path: float, sequences_reduction_ratio: float, mlp_ratio: float) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, height: int, width: int, output_attentions: bool = ...): # -> Any:
        ...



class PvtEncoder(nn.Module):
    def __init__(self, config: PvtConfig) -> None:
        ...

    def forward(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutput]:
        ...



class PvtPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = PvtConfig
    base_model_prefix = ...
    main_input_name = ...


PVT_START_DOCSTRING = ...
PVT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Pvt encoder outputting raw hidden-states without any specific head on top.", PVT_START_DOCSTRING)
class PvtModel(PvtPreTrainedModel):
    def __init__(self, config: PvtConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(PVT_INPUTS_DOCSTRING.format("(batch_size, channels, height, width)"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC, modality="vision", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutput]:
        ...



@add_start_docstrings("""
    Pvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """, PVT_START_DOCSTRING)
class PvtForImageClassification(PvtPreTrainedModel):
    def __init__(self, config: PvtConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(PVT_INPUTS_DOCSTRING.format("(batch_size, channels, height, width)"))
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: Optional[torch.Tensor], labels: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
