"""
This type stub file was generated by pyright.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Union
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPastAndCrossAttentions, TFSeq2SeqLMOutput, TFSeq2SeqModelOutput, TFSeq2SeqSequenceClassifierOutput
from ...modeling_tf_utils import TFCausalLanguageModelingLoss, TFModelInputType, TFPreTrainedModel, TFSequenceClassificationLoss, keras, keras_serializable, unpack_inputs
from ...utils import add_code_sample_docstrings, add_end_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_bart import BartConfig

""" TF 2.0 Bart model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
LARGE_NEGATIVE = ...
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    ...

class TFBartLearnedPositionalEmbedding(keras.layers.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs) -> None:
        ...

    def call(self, input_shape: Optional[tf.TensorShape] = ..., past_key_values_length: int = ..., position_ids: tf.Tensor | None = ...):
        """Input is expected to be of size [bsz x seqlen]."""
        ...



class TFBartAttention(keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = ..., is_decoder: bool = ..., bias: bool = ..., **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor, key_value_states: tf.Tensor | None = ..., past_key_value: Tuple[Tuple[tf.Tensor]] | None = ..., attention_mask: tf.Tensor | None = ..., layer_head_mask: tf.Tensor | None = ..., training: Optional[bool] = ...) -> Tuple[tf.Tensor, tf.Tensor | None]:
        """Input shape: Batch x Time x Channel"""
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFBartEncoderLayer(keras.layers.Layer):
    def __init__(self, config: BartConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor, attention_mask: np.ndarray | tf.Tensor | None, layer_head_mask: tf.Tensor | None, training: Optional[bool] = ...) -> tf.Tensor:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`tf.Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFBartDecoderLayer(keras.layers.Layer):
    def __init__(self, config: BartConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor, attention_mask: np.ndarray | tf.Tensor | None = ..., encoder_hidden_states: np.ndarray | tf.Tensor | None = ..., encoder_attention_mask: np.ndarray | tf.Tensor | None = ..., layer_head_mask: tf.Tensor | None = ..., cross_attn_layer_head_mask: tf.Tensor | None = ..., past_key_value: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., training: Optional[bool] = ...) -> Tuple[tf.Tensor, tf.Tensor, Tuple[Tuple[tf.Tensor]]]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`tf.Tensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`tf.Tensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`tf.Tensor`): mask for attention heads in a given layer of size
                `(decoder_attention_heads,)`
            cross_attn_layer_head_mask (`tf.Tensor`): mask for heads of the cross-attention module.
                `(decoder_attention_heads,)`
            past_key_value (`Tuple(tf.Tensor)`): cached past key and value projection states
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFBartClassificationHead(keras.layers.Layer):
    """Head for sentence-level classification tasks."""
    def __init__(self, inner_dim: int, num_classes: int, pooler_dropout: float, name: str, **kwargs) -> None:
        ...

    def call(self, inputs):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFBartPretrainedModel(TFPreTrainedModel):
    config_class = BartConfig
    base_model_prefix = ...
    @property
    def dummy_inputs(self): # -> Dict[str, Any]:
        ...

    def tf_to_pt_weight_rename(self, tf_weight): # -> tuple[Literal['model.shared.weight'], Literal['model.decoder.embed_tokens.weight']] | tuple[Any]:
        ...



BART_START_DOCSTRING = ...
BART_GENERATION_EXAMPLE = ...
BART_INPUTS_DOCSTRING = ...
@keras_serializable
class TFBartEncoder(keras.layers.Layer):
    config_class = BartConfig
    def __init__(self, config: BartConfig, embed_tokens: Optional[keras.layers.Embedding] = ..., **kwargs) -> None:
        ...

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        """
        Args:
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, `optional):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@keras_serializable
class TFBartDecoder(keras.layers.Layer):
    config_class = BartConfig
    def __init__(self, config: BartConfig, embed_tokens: Optional[keras.layers.Embedding] = ..., **kwargs) -> None:
        ...

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., encoder_hidden_states: np.ndarray | tf.Tensor | None = ..., encoder_attention_mask: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., cross_attn_head_mask: np.ndarray | tf.Tensor | None = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        r"""
        Args:
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
                range `[0, config.max_position_embeddings - 1]`.
            encoder_hidden_states (`tf.Tensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`tf.Tensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`tf.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`tf.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`tf.tTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@keras_serializable
class TFBartMainLayer(keras.layers.Layer):
    config_class = BartConfig
    def __init__(self, config: BartConfig, load_weight_prefix=..., **kwargs) -> None:
        ...

    def get_input_embeddings(self):
        ...

    def set_input_embeddings(self, new_embeddings): # -> None:
        ...

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., decoder_input_ids: np.ndarray | tf.Tensor | None = ..., decoder_attention_mask: np.ndarray | tf.Tensor | None = ..., decoder_position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., decoder_head_mask: np.ndarray | tf.Tensor | None = ..., cross_attn_head_mask: np.ndarray | tf.Tensor | None = ..., encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., decoder_inputs_embeds: np.ndarray | tf.Tensor | None = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFSeq2SeqModelOutput, Tuple[tf.Tensor]]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("The bare BART Model outputting raw hidden-states without any specific head on top.", BART_START_DOCSTRING)
class TFBartModel(TFBartPretrainedModel):
    _requires_load_weight_prefix = ...
    def __init__(self, config: BartConfig, load_weight_prefix=..., *inputs, **kwargs) -> None:
        ...

    def get_encoder(self): # -> TFBartEncoder:
        ...

    def get_decoder(self): # -> TFBartDecoder:
        ...

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., decoder_input_ids: np.ndarray | tf.Tensor | None = ..., decoder_attention_mask: np.ndarray | tf.Tensor | None = ..., decoder_position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., decoder_head_mask: np.ndarray | tf.Tensor | None = ..., cross_attn_head_mask: np.ndarray | tf.Tensor | None = ..., encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., decoder_inputs_embeds: np.ndarray | tf.Tensor | None = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ..., **kwargs) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...

    def serving_output(self, output): # -> TFSeq2SeqModelOutput:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class BiasLayer(keras.layers.Layer):
    """
    Bias as a layer. It is used for serialization purposes: `keras.Model.save_weights` stores on a per-layer basis,
    so all weights have to be registered in a layer.
    """
    def __init__(self, shape, initializer, trainable, name, **kwargs) -> None:
        ...

    def call(self, x):
        ...



@add_start_docstrings("The BART Model with a language modeling head. Can be used for summarization.", BART_START_DOCSTRING)
class TFBartForConditionalGeneration(TFBartPretrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_missing = ...
    _requires_load_weight_prefix = ...
    def __init__(self, config, load_weight_prefix=..., *inputs, **kwargs) -> None:
        ...

    def get_decoder(self): # -> TFBartDecoder:
        ...

    def get_encoder(self): # -> TFBartEncoder:
        ...

    def get_output_embeddings(self):
        ...

    def set_output_embeddings(self, value): # -> None:
        ...

    def get_bias(self): # -> dict[str, Any]:
        ...

    def set_bias(self, value): # -> None:
        ...

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., decoder_input_ids: np.ndarray | tf.Tensor | None = ..., decoder_attention_mask: np.ndarray | tf.Tensor | None = ..., decoder_position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., decoder_head_mask: np.ndarray | tf.Tensor | None = ..., cross_attn_head_mask: np.ndarray | tf.Tensor | None = ..., encoder_outputs: Optional[TFBaseModelOutput] = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., decoder_inputs_embeds: np.ndarray | tf.Tensor | None = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFSeq2SeqLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        ...

    def serving_output(self, output): # -> TFSeq2SeqLMOutput:
        ...

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=..., attention_mask=..., decoder_attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., use_cache=..., encoder_outputs=..., **kwargs): # -> dict[str, Any]:
        ...

    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    Bart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """, BART_START_DOCSTRING)
class TFBartForSequenceClassification(TFBartPretrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: BartConfig, load_weight_prefix=..., *inputs, **kwargs) -> None:
        ...

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., decoder_input_ids: np.ndarray | tf.Tensor | None = ..., decoder_attention_mask: np.ndarray | tf.Tensor | None = ..., decoder_position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., decoder_head_mask: np.ndarray | tf.Tensor | None = ..., cross_attn_head_mask: np.ndarray | tf.Tensor | None = ..., encoder_outputs: Optional[TFBaseModelOutput] = ..., past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., decoder_inputs_embeds: np.ndarray | tf.Tensor | None = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFSeq2SeqSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
        """
        ...

    def serving_output(self, output): # -> TFSeq2SeqSequenceClassifierOutput:
        ...

    def build(self, input_shape=...): # -> None:
        ...
