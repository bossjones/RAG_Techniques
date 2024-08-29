"""
This type stub file was generated by pyright.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Union
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions, TFBaseModelOutputWithPoolingAndCrossAttentions, TFCausalLMOutputWithCrossAttentions, TFMaskedLMOutput, TFMultipleChoiceModelOutput, TFQuestionAnsweringModelOutput, TFSequenceClassifierOutput, TFTokenClassifierOutput
from ...modeling_tf_utils import TFCausalLanguageModelingLoss, TFMaskedLanguageModelingLoss, TFModelInputType, TFMultipleChoiceLoss, TFPreTrainedModel, TFQuestionAnsweringLoss, TFSequenceClassificationLoss, TFTokenClassificationLoss, keras, keras_serializable, unpack_inputs
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_xlm_roberta import XLMRobertaConfig

""" TF 2.0 XLM-RoBERTa model."""
logger = ...
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
XLM_ROBERTA_START_DOCSTRING = ...
XLM_ROBERTA_INPUTS_DOCSTRING = ...
class TFXLMRobertaEmbeddings(keras.layers.Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=...):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            input_ids: tf.Tensor
        Returns: tf.Tensor
        """
        ...

    def call(self, input_ids=..., position_ids=..., token_type_ids=..., inputs_embeds=..., past_key_values_length=..., training=...):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        ...



class TFXLMRobertaPooler(keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFXLMRobertaSelfAttention(keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs) -> None:
        ...

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        ...

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, encoder_hidden_states: tf.Tensor, encoder_attention_mask: tf.Tensor, past_key_value: Tuple[tf.Tensor], output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFXLMRobertaSelfOutput(keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFXLMRobertaAttention(keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs) -> None:
        ...

    def prune_heads(self, heads):
        ...

    def call(self, input_tensor: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, encoder_hidden_states: tf.Tensor, encoder_attention_mask: tf.Tensor, past_key_value: Tuple[tf.Tensor], output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFXLMRobertaIntermediate(keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFXLMRobertaOutput(keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFXLMRobertaLayer(keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, encoder_hidden_states: tf.Tensor | None, encoder_attention_mask: tf.Tensor | None, past_key_value: Tuple[tf.Tensor] | None, output_attentions: bool, training: bool = ...) -> Tuple[tf.Tensor]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFXLMRobertaEncoder(keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, encoder_hidden_states: tf.Tensor | None, encoder_attention_mask: tf.Tensor | None, past_key_values: Tuple[Tuple[tf.Tensor]] | None, use_cache: Optional[bool], output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool = ...) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



@keras_serializable
class TFXLMRobertaMainLayer(keras.layers.Layer):
    config_class = XLMRobertaConfig
    def __init__(self, config, add_pooling_layer=..., **kwargs) -> None:
        ...

    def get_input_embeddings(self) -> keras.layers.Layer:
        ...

    def set_input_embeddings(self, value: tf.Variable): # -> None:
        ...

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., encoder_hidden_states: np.ndarray | tf.Tensor | None = ..., encoder_attention_mask: np.ndarray | tf.Tensor | None = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ...) -> Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFXLMRobertaPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = XLMRobertaConfig
    base_model_prefix = ...


@add_start_docstrings("The bare XLM RoBERTa Model transformer outputting raw hidden-states without any specific head on top.", XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaModel(TFXLMRobertaPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., encoder_hidden_states: np.ndarray | tf.Tensor | None = ..., encoder_attention_mask: np.ndarray | tf.Tensor | None = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[Tuple, TFBaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). Set to `False` during training, `True` during generation
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFXLMRobertaLMHead(keras.layers.Layer):
    """XLMRoberta Head for masked language modeling."""
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



@add_start_docstrings("""XLM RoBERTa Model with a `language modeling` head on top.""", XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaForMaskedLM(TFXLMRobertaPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    def get_lm_head(self): # -> TFXLMRobertaLMHead:
        ...

    def get_prefix_bias_name(self):
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC, mask="<mask>", expected_output="' Paris'", expected_loss=0.1)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("XLM-RoBERTa Model with a `language modeling` head on top for CLM fine-tuning.", XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaForCausalLM(TFXLMRobertaPreTrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: XLMRobertaConfig, *inputs, **kwargs) -> None:
        ...

    def get_lm_head(self): # -> TFXLMRobertaLMHead:
        ...

    def get_prefix_bias_name(self):
        ...

    def prepare_inputs_for_generation(self, input_ids, past_key_values=..., attention_mask=..., **model_kwargs): # -> dict[str, Any]:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFCausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., encoder_hidden_states: np.ndarray | tf.Tensor | None = ..., encoder_attention_mask: np.ndarray | tf.Tensor | None = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFCausalLMOutputWithCrossAttentions, Tuple[tf.Tensor]]:
        r"""
        encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). Set to `False` during training, `True` during generation
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFXLMRobertaClassificationHead(keras.layers.Layer):
    """Head for sentence-level classification tasks."""
    def __init__(self, config, **kwargs) -> None:
        ...

    def call(self, features, training=...):
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    XLM RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaForSequenceClassification(TFXLMRobertaPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint="cardiffnlp/twitter-roberta-base-emotion", output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output="'optimism'", expected_loss=0.08)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
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
    XLM Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """, XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaForMultipleChoice(TFXLMRobertaPreTrainedModel, TFMultipleChoiceLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    XLM RoBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """, XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaForTokenClassification(TFXLMRobertaPreTrainedModel, TFTokenClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint="ydshieh/roberta-large-ner-english", output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output="['O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'LOC', 'O', 'LOC', 'LOC']", expected_loss=0.01)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    XLM RoBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaForQuestionAnswering(TFXLMRobertaPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint="ydshieh/roberta-base-squad2", output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC, expected_output="' puppet'", expected_loss=0.86)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., start_positions: np.ndarray | tf.Tensor | None = ..., end_positions: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
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
