"""
This type stub file was generated by pyright.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Union
from ...modeling_tf_outputs import TFBaseModelOutput, TFMaskedLMOutput, TFMultipleChoiceModelOutput, TFQuestionAnsweringModelOutput, TFSequenceClassifierOutput, TFTokenClassifierOutput
from ...modeling_tf_utils import TFMaskedLanguageModelingLoss, TFModelInputType, TFMultipleChoiceLoss, TFPreTrainedModel, TFQuestionAnsweringLoss, TFSequenceClassificationLoss, TFTokenClassificationLoss, keras, keras_serializable, unpack_inputs
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_distilbert import DistilBertConfig

"""
 TF 2.0 DistilBERT model
"""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
class TFEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def call(self, input_ids=..., position_ids=..., inputs_embeds=..., training=...):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        ...



class TFMultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def prune_heads(self, heads):
        ...

    def call(self, query, key, value, mask, head_mask, output_attentions, training=...): # -> tuple[Any, Any] | tuple[Any]:
        """
        Parameters:
            query: tf.Tensor(bs, seq_length, dim)
            key: tf.Tensor(bs, seq_length, dim)
            value: tf.Tensor(bs, seq_length, dim)
            mask: tf.Tensor(bs, seq_length)

        Returns:
            weights: tf.Tensor(bs, n_heads, seq_length, seq_length) Attention weights context: tf.Tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFFFN(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def call(self, input, training=...):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFTransformerBlock(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def call(self, x, attn_mask, head_mask, output_attentions, training=...): # -> tuple[Any, Any] | tuple[Any]:
        """
        Parameters:
            x: tf.Tensor(bs, seq_length, dim)
            attn_mask: tf.Tensor(bs, seq_length)

        Outputs: sa_weights: tf.Tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
        tf.Tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFTransformer(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def call(self, x, attn_mask, head_mask, output_attentions, output_hidden_states, return_dict, training=...): # -> tuple[Any, ...] | TFBaseModelOutput:
        """
        Parameters:
            x: tf.Tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: tf.Tensor(bs, seq_length) Attention mask on the sequence.

        Returns:
            hidden_state: tf.Tensor(bs, seq_length, dim)
                Sequence of hidden states in the last (top) layer
            all_hidden_states: Tuple[tf.Tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[tf.Tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@keras_serializable
class TFDistilBertMainLayer(keras.layers.Layer):
    config_class = DistilBertConfig
    def __init__(self, config, **kwargs) -> None:
        ...

    def get_input_embeddings(self): # -> TFEmbeddings:
        ...

    def set_input_embeddings(self, value): # -> None:
        ...

    @unpack_inputs
    def call(self, input_ids=..., attention_mask=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=...):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFDistilBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DistilBertConfig
    base_model_prefix = ...


DISTILBERT_START_DOCSTRING = ...
DISTILBERT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare DistilBERT encoder/transformer outputting raw hidden-states without any specific head on top.", DISTILBERT_START_DOCSTRING)
class TFDistilBertModel(TFDistilBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFDistilBertLMHead(keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs) -> None:
        ...

    def build(self, input_shape): # -> None:
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



@add_start_docstrings("""DistilBert Model with a `masked language modeling` head on top.""", DISTILBERT_START_DOCSTRING)
class TFDistilBertForMaskedLM(TFDistilBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    def get_lm_head(self): # -> TFDistilBertLMHead:
        ...

    def get_prefix_bias_name(self):
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, DISTILBERT_START_DOCSTRING)
class TFDistilBertForSequenceClassification(TFDistilBertPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
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
    DistilBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """, DISTILBERT_START_DOCSTRING)
class TFDistilBertForTokenClassification(TFDistilBertPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    DistilBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """, DISTILBERT_START_DOCSTRING)
class TFDistilBertForMultipleChoice(TFDistilBertPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, DISTILBERT_START_DOCSTRING)
class TFDistilBertForQuestionAnswering(TFDistilBertPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., start_positions: np.ndarray | tf.Tensor | None = ..., end_positions: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
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
