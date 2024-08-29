"""
This type stub file was generated by pyright.
"""

import random
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple
from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPastAndCrossAttentions, FlaxCausalLMOutputWithCrossAttentions
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_blenderbot import BlenderbotConfig

""" Flax Blenderbot model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
BLENDERBOT_START_DOCSTRING = ...
BLENDERBOT_INPUTS_DOCSTRING = ...
BLENDERBOT_ENCODE_INPUTS_DOCSTRING = ...
BLENDERBOT_DECODE_INPUTS_DOCSTRING = ...
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    Shift input ids one token to the right.
    """
    ...

class FlaxBlenderbotAttention(nn.Module):
    config: BlenderbotConfig
    embed_dim: int
    num_heads: int
    dropout: float = ...
    causal: bool = ...
    bias: bool = ...
    dtype: jnp.dtype = ...
    def setup(self) -> None:
        ...

    def __call__(self, hidden_states: jnp.ndarray, key_value_states: Optional[jnp.ndarray] = ..., attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., deterministic: bool = ...) -> Tuple[jnp.ndarray]:
        """Input shape: Batch x Time x Channel"""
        ...



class FlaxBlenderbotEncoderLayer(nn.Module):
    config: BlenderbotConfig
    dtype: jnp.dtype = ...
    def setup(self) -> None:
        ...

    def __call__(self, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray, output_attentions: bool = ..., deterministic: bool = ...) -> Tuple[jnp.ndarray]:
        ...



class FlaxBlenderbotEncoderLayerCollection(nn.Module):
    config: BlenderbotConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...

    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any | tuple[()], ...] | FlaxBaseModelOutput:
        ...



class FlaxBlenderbotDecoderLayer(nn.Module):
    config: BlenderbotConfig
    dtype: jnp.dtype = ...
    def setup(self) -> None:
        ...

    def __call__(self, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray, encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., output_attentions: bool = ..., deterministic: bool = ...) -> Tuple[jnp.ndarray]:
        ...



class FlaxBlenderbotDecoderLayerCollection(nn.Module):
    config: BlenderbotConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...

    def __call__(self, hidden_states, attention_mask, encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...



class FlaxBlenderbotEncoder(nn.Module):
    config: BlenderbotConfig
    embed_tokens: nn.Embed
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...

    def __call__(self, input_ids, attention_mask, position_ids, output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | FlaxBaseModelOutput:
        ...



class FlaxBlenderbotDecoder(nn.Module):
    config: BlenderbotConfig
    embed_tokens: nn.Embed
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...

    def __call__(self, input_ids, attention_mask, position_ids, encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> tuple[Any, ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...



class FlaxBlenderbotModule(nn.Module):
    config: BlenderbotConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...

    def __call__(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, position_ids, decoder_position_ids, output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> FlaxSeq2SeqModelOutput:
        ...



class FlaxBlenderbotPreTrainedModel(FlaxPreTrainedModel):
    config_class = BlenderbotConfig
    base_model_prefix: str = ...
    module_class: nn.Module = ...
    def __init__(self, config: BlenderbotConfig, input_shape: Tuple[int] = ..., seed: int = ..., dtype: jnp.dtype = ..., _do_init: bool = ..., **kwargs) -> None:
        ...

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = ...) -> FrozenDict:
        ...

    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*:
                `attentions`). `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*)
                is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
                cross-attention of the decoder.
        """
        ...

    @add_start_docstrings(BLENDERBOT_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=BlenderbotConfig)
    def encode(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = ..., position_ids: Optional[jnp.ndarray] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlaxBlenderbotForConditionalGeneration

        >>> model = FlaxBlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
        >>> encoder_outputs = model.encode(**inputs)
        ```"""
        ...

    @add_start_docstrings(BLENDERBOT_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=BlenderbotConfig)
    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_position_ids: Optional[jnp.ndarray] = ..., past_key_values: dict = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        r"""
        Returns:

        Example:

        ```python
        >>> import jax.numpy as jnp
        >>> from transformers import AutoTokenizer, FlaxBlenderbotForConditionalGeneration

        >>> model = FlaxBlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
        >>> encoder_outputs = model.encode(**inputs)

        >>> decoder_start_token_id = model.config.decoder_start_token_id
        >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> last_decoder_hidden_states = outputs.last_hidden_state
        ```"""
        ...

    @add_start_docstrings_to_model_forward(BLENDERBOT_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = ..., decoder_input_ids: Optional[jnp.ndarray] = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., position_ids: Optional[jnp.ndarray] = ..., decoder_position_ids: Optional[jnp.ndarray] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        ...



@add_start_docstrings("The bare MBart Model transformer outputting raw hidden-states without any specific head on top.", BLENDERBOT_START_DOCSTRING)
class FlaxBlenderbotModel(FlaxBlenderbotPreTrainedModel):
    config: BlenderbotConfig
    dtype: jnp.dtype = ...
    module_class = ...


class FlaxBlenderbotForConditionalGenerationModule(nn.Module):
    config: BlenderbotConfig
    dtype: jnp.dtype = ...
    bias_init: Callable[..., jnp.ndarray] = ...
    def setup(self): # -> None:
        ...

    def __call__(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, position_ids, decoder_position_ids, output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> Any | FlaxSeq2SeqLMOutput:
        ...



@add_start_docstrings("The Blenderbot Model with a language modeling head. Can be used for summarization.", BLENDERBOT_START_DOCSTRING)
class FlaxBlenderbotForConditionalGeneration(FlaxBlenderbotPreTrainedModel):
    module_class = ...
    dtype: jnp.dtype = ...
    @add_start_docstrings(BLENDERBOT_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=BlenderbotConfig)
    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_position_ids: Optional[jnp.ndarray] = ..., past_key_values: dict = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...): # -> FlaxCausalLMOutputWithCrossAttentions | Any:
        r"""
        Returns:

        Example:

        ```python
        >>> import jax.numpy as jnp
        >>> from transformers import AutoTokenizer, FlaxBlenderbotForConditionalGeneration

        >>> model = FlaxBlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
        >>> encoder_outputs = model.encode(**inputs)

        >>> decoder_start_token_id = model.config.decoder_start_token_id
        >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> logits = outputs.logits
        ```"""
        ...

    def prepare_inputs_for_generation(self, decoder_input_ids, max_length, attention_mask: Optional[jax.Array] = ..., decoder_attention_mask: Optional[jax.Array] = ..., encoder_outputs=..., **kwargs): # -> dict[str, Any]:
        ...

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        ...



FLAX_BLENDERBOT_CONDITIONAL_GENERATION_DOCSTRING = ...
