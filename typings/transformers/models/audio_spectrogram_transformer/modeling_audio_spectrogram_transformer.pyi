"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional, Set, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_audio_spectrogram_transformer import ASTConfig

""" PyTorch Audio Spectrogram Transformer (AST) model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_SEQ_CLASS_CHECKPOINT = ...
_SEQ_CLASS_EXPECTED_OUTPUT = ...
_SEQ_CLASS_EXPECTED_LOSS = ...
class ASTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """
    def __init__(self, config: ASTConfig) -> None:
        ...

    def get_shape(self, config): # -> tuple[Any, Any]:
        ...

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        ...



class ASTPatchEmbeddings(nn.Module):
    """
    This class turns `input_values` into the initial `hidden_states` (patch embeddings) of shape `(batch_size,
    seq_length, hidden_size)` to be consumed by a Transformer.
    """
    def __init__(self, config) -> None:
        ...

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        ...



class ASTSelfAttention(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        ...

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, hidden_states, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        ...



class ASTSelfOutput(nn.Module):
    """
    The residual connection is defined in ASTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """
    def __init__(self, config: ASTConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...



class ASTAttention(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        ...

    def prune_heads(self, heads: Set[int]) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        ...



class ASTIntermediate(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



class ASTOutput(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...



class ASTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""
    def __init__(self, config: ASTConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        ...



class ASTEncoder(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...) -> Union[tuple, BaseModelOutput]:
        ...



class ASTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ASTConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...


AUDIO_SPECTROGRAM_TRANSFORMER_START_DOCSTRING = ...
AUDIO_SPECTROGRAM_TRANSFORMER_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare AST Model transformer outputting raw hidden-states without any specific head on top.", AUDIO_SPECTROGRAM_TRANSFORMER_START_DOCSTRING)
class ASTModel(ASTPreTrainedModel):
    def __init__(self, config: ASTConfig) -> None:
        ...

    def get_input_embeddings(self) -> ASTPatchEmbeddings:
        ...

    @add_start_docstrings_to_model_forward(AUDIO_SPECTROGRAM_TRANSFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC, modality="audio", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, input_values: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        ...



class ASTMLPHead(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        ...

    def forward(self, hidden_state): # -> Any:
        ...



@add_start_docstrings("""
    Audio Spectrogram Transformer model with an audio classification head on top (a linear layer on top of the pooled
    output) e.g. for datasets like AudioSet, Speech Commands v2.
    """, AUDIO_SPECTROGRAM_TRANSFORMER_START_DOCSTRING)
class ASTForAudioClassification(ASTPreTrainedModel):
    def __init__(self, config: ASTConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(AUDIO_SPECTROGRAM_TRANSFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_SEQ_CLASS_CHECKPOINT, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC, modality="audio", expected_output=_SEQ_CLASS_EXPECTED_OUTPUT, expected_loss=_SEQ_CLASS_EXPECTED_LOSS)
    def forward(self, input_values: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the audio classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
