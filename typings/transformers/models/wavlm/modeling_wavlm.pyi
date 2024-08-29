"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import CausalLMOutput, SequenceClassifierOutput, TokenClassifierOutput, Wav2Vec2BaseModelOutput, XVectorOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_wavlm import WavLMConfig

""" PyTorch WavLM model."""
logger = ...
_HIDDEN_STATES_START_POSITION = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_CTC_EXPECTED_OUTPUT = ...
_CTC_EXPECTED_LOSS = ...
_FRAME_CLASS_CHECKPOINT = ...
_FRAME_EXPECTED_OUTPUT = ...
_XVECTOR_CHECKPOINT = ...
_XVECTOR_EXPECTED_OUTPUT = ...
class WavLMNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...

    def forward(self, hidden_states):
        ...



class WavLMLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...

    def forward(self, hidden_states):
        ...



class WavLMGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...

    def forward(self, hidden_states):
        ...



class WavLMPositionalConvEmbedding(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states):
        ...



class WavLMSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings) -> None:
        ...

    def forward(self, hidden_states):
        ...



class WavLMFeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""
    def __init__(self, config) -> None:
        ...

    def forward(self, input_values): # -> Any:
        ...



class WavLMFeatureExtractor(WavLMFeatureEncoder):
    def __init__(self, config) -> None:
        ...



class WavLMFeatureProjection(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> tuple[Any, Any]:
        ...



class WavLMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = ..., num_buckets: int = ..., max_distance: int = ..., has_relative_position_bias: bool = ...) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., position_bias: Optional[torch.Tensor] = ..., output_attentions: bool = ..., index=...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Attention layer with relative attention"""
        ...

    def torch_multi_head_self_attention(self, hidden_states: torch.FloatTensor, attention_mask: Union[torch.LongTensor, torch.BoolTensor], gated_position_bias: torch.FloatTensor, output_attentions: bool) -> (torch.FloatTensor, torch.FloatTensor):
        """simple wrapper around torch's multi_head_attention_forward function"""
        ...

    def compute_bias(self, query_length: int, key_length: int) -> torch.FloatTensor:
        ...



class WavLMFeedForward(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Any:
        ...



class WavLMEncoderLayer(nn.Module):
    def __init__(self, config: WavLMConfig, has_relative_position_bias: bool = ...) -> None:
        ...

    def forward(self, hidden_states, attention_mask=..., position_bias=..., output_attentions=..., index=...): # -> tuple[Any, Any, Any] | tuple[Any, Any]:
        ...



class WavLMEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config: WavLMConfig, has_relative_position_bias: bool = ...) -> None:
        ...

    def forward(self, hidden_states, attention_mask=..., position_bias=..., output_attentions=...): # -> tuple[Any, Any, Any] | tuple[Any, Any]:
        ...



class WavLMEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> tuple[Any, ...] | BaseModelOutput:
        ...



class WavLMEncoderStableLayerNorm(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> tuple[Any, ...] | BaseModelOutput:
        ...



class WavLMGumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax. See [CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
    """
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> tuple[Tensor | Any, Tensor]:
        ...



class WavLMAdapter(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Any:
        ...



class WavLMAdapterLayer(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Tensor:
        ...



class WavLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = WavLMConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...


WAVLM_START_DOCSTRING = ...
WAVLM_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare WavLM Model transformer outputting raw hidden-states without any specific head on top.", WAVLM_START_DOCSTRING)
class WavLMModel(WavLMPreTrainedModel):
    def __init__(self, config: WavLMConfig) -> None:
        ...

    def freeze_feature_extractor(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        ...

    def freeze_feature_encoder(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...

    @add_start_docstrings_to_model_forward(WAVLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=Wav2Vec2BaseModelOutput, config_class=_CONFIG_FOR_DOC, modality="audio", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = ..., mask_time_indices: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        ...



@add_start_docstrings("""WavLM Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""", WAVLM_START_DOCSTRING)
class WavLMForCTC(WavLMPreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = ...) -> None:
        ...

    def tie_weights(self): # -> None:
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """
        ...

    def freeze_feature_extractor(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...

    def freeze_feature_encoder(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...

    def freeze_base_model(self): # -> None:
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        ...

    @add_start_docstrings_to_model_forward(WAVLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC, expected_output=_CTC_EXPECTED_OUTPUT, expected_loss=_CTC_EXPECTED_LOSS)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[torch.Tensor] = ...) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        ...



@add_start_docstrings("""
    WavLM Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
    SUPERB Keyword Spotting.
    """, WAVLM_START_DOCSTRING)
class WavLMForSequenceClassification(WavLMPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    def freeze_feature_extractor(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        ...

    def freeze_feature_encoder(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...

    def freeze_base_model(self): # -> None:
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        ...

    @add_start_docstrings_to_model_forward(WAVLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC, modality="audio")
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[torch.Tensor] = ...) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...



@add_start_docstrings("""
    WavLM Model with a frame classification head on top for tasks like Speaker Diarization.
    """, WAVLM_START_DOCSTRING)
class WavLMForAudioFrameClassification(WavLMPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    def freeze_feature_extractor(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...

    def freeze_feature_encoder(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...

    def freeze_base_model(self): # -> None:
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        ...

    @add_start_docstrings_to_model_forward(WAVLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_FRAME_CLASS_CHECKPOINT, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC, modality="audio", expected_output=_FRAME_EXPECTED_OUTPUT)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...



class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=..., margin=...) -> None:
        ...

    def forward(self, hidden_states, labels): # -> Any:
        ...



class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



@add_start_docstrings("""
    WavLM Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """, WAVLM_START_DOCSTRING)
class WavLMForXVector(WavLMPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    def freeze_feature_extractor(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...

    def freeze_feature_encoder(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...

    def freeze_base_model(self): # -> None:
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        ...

    @add_start_docstrings_to_model_forward(WAVLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_XVECTOR_CHECKPOINT, output_type=XVectorOutput, config_class=_CONFIG_FOR_DOC, modality="audio", expected_output=_XVECTOR_EXPECTED_OUTPUT)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[torch.Tensor] = ...) -> Union[Tuple, XVectorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
