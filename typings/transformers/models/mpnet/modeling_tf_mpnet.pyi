"""
This type stub file was generated by pyright.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Union
from ...modeling_tf_outputs import TFBaseModelOutput, TFMaskedLMOutput, TFMultipleChoiceModelOutput, TFQuestionAnsweringModelOutput, TFSequenceClassifierOutput, TFTokenClassifierOutput
from ...modeling_tf_utils import TFMaskedLanguageModelingLoss, TFModelInputType, TFMultipleChoiceLoss, TFPreTrainedModel, TFQuestionAnsweringLoss, TFSequenceClassificationLoss, TFTokenClassificationLoss, keras, keras_serializable, unpack_inputs
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_mpnet import MPNetConfig

""" TF 2.0 MPNet model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
class TFMPNetPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MPNetConfig
    base_model_prefix = ...


class TFMPNetEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position embeddings."""
    def __init__(self, config, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def create_position_ids_from_input_ids(self, input_ids):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            input_ids: tf.Tensor
        Returns: tf.Tensor
        """
        ...

    def call(self, input_ids=..., position_ids=..., inputs_embeds=..., training=...):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        ...



class TFMPNetPooler(keras.layers.Layer):
    def __init__(self, config: MPNetConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFMPNetSelfAttention(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def transpose_for_scores(self, x, batch_size):
        ...

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, position_bias=..., training=...): # -> tuple[Any, Any] | tuple[Any]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFMPNetAttention(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def prune_heads(self, heads):
        ...

    def call(self, input_tensor, attention_mask, head_mask, output_attentions, position_bias=..., training=...):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFMPNetIntermediate(keras.layers.Layer):
    def __init__(self, config: MPNetConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFMPNetOutput(keras.layers.Layer):
    def __init__(self, config: MPNetConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFMPNetLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, position_bias=..., training=...):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFMPNetEncoder(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, output_hidden_states, return_dict, training=...): # -> tuple[Any, ...] | TFBaseModelOutput:
        ...

    def compute_position_bias(self, x, position_ids=...):
        """Compute binned relative position bias"""
        ...



@keras_serializable
class TFMPNetMainLayer(keras.layers.Layer):
    config_class = MPNetConfig
    def __init__(self, config, **kwargs) -> None:
        ...

    def get_input_embeddings(self) -> keras.layers.Layer:
        ...

    def set_input_embeddings(self, value: tf.Variable): # -> None:
        ...

    @unpack_inputs
    def call(self, input_ids=..., attention_mask=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=...): # -> TFBaseModelOutputWithPooling:
        ...

    def build(self, input_shape=...): # -> None:
        ...



MPNET_START_DOCSTRING = ...
MPNET_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare MPNet Model transformer outputting raw hidden-states without any specific head on top.", MPNET_START_DOCSTRING)
class TFMPNetModel(TFMPNetPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: Optional[Union[np.array, tf.Tensor]] = ..., position_ids: Optional[Union[np.array, tf.Tensor]] = ..., head_mask: Optional[Union[np.array, tf.Tensor]] = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFMPNetLMHead(keras.layers.Layer):
    """MPNet head for masked and permuted language modeling"""
    def __init__(self, config, input_embeddings, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def get_output_embeddings(self): # -> Any:
        ...

    def set_output_embeddings(self, value): # -> None:
        ...

    def get_bias(self): # -> dict[str, Any]:
        ...

    def set_bias(self, value): # -> None:
        ...

    def call(self, hidden_states):
        ...



@add_start_docstrings("""MPNet Model with a `language modeling` head on top.""", MPNET_START_DOCSTRING)
class TFMPNetForMaskedLM(TFMPNetPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    def get_lm_head(self): # -> TFMPNetLMHead:
        ...

    def get_prefix_bias_name(self):
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: tf.Tensor | None = ..., training: bool = ...) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFMPNetClassificationHead(keras.layers.Layer):
    """Head for sentence-level classification tasks."""
    def __init__(self, config, **kwargs) -> None:
        ...

    def call(self, features, training=...):
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    MPNet Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """, MPNET_START_DOCSTRING)
class TFMPNetForSequenceClassification(TFMPNetPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: Optional[Union[np.array, tf.Tensor]] = ..., position_ids: Optional[Union[np.array, tf.Tensor]] = ..., head_mask: Optional[Union[np.array, tf.Tensor]] = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: tf.Tensor | None = ..., training: bool = ...) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    MPNet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, MPNET_START_DOCSTRING)
class TFMPNetForMultipleChoice(TFMPNetPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: tf.Tensor | None = ..., training: bool = ...) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
       MPNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
       Named-Entity-Recognition (NER) tasks.
       """, MPNET_START_DOCSTRING)
class TFMPNetForTokenClassification(TFMPNetPreTrainedModel, TFTokenClassificationLoss):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: tf.Tensor | None = ..., training: bool = ...) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    MPNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, MPNET_START_DOCSTRING)
class TFMPNetForQuestionAnswering(TFMPNetPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: Optional[Union[np.array, tf.Tensor]] = ..., position_ids: Optional[Union[np.array, tf.Tensor]] = ..., head_mask: Optional[Union[np.array, tf.Tensor]] = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., start_positions: tf.Tensor | None = ..., end_positions: tf.Tensor | None = ..., training: bool = ..., **kwargs) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...
