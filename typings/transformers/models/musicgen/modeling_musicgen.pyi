"""
This type stub file was generated by pyright.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Tuple, Union
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions, ModelOutput, Seq2SeqLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, is_flash_attn_2_available, replace_return_docstrings
from .configuration_musicgen import MusicgenConfig, MusicgenDecoderConfig
from ...generation.streamers import BaseStreamer

""" PyTorch Musicgen model."""
if is_flash_attn_2_available():
    ...
if TYPE_CHECKING:
    ...
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
@dataclass
class MusicgenUnconditionalInput(ModelOutput):
    """
    Args:
        encoder_outputs  (`Tuple[torch.FloatTensor]` of length 1, with tensor shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the text encoder model.
        attention_mask (`torch.LongTensor`)  of shape `(batch_size, sequence_length)`, *optional*):
            Encoder attention mask to avoid performing attention on padding token indices. Mask values selected in `[0,
            1]`: 1 for tokens that are **not masked**, 0 for tokens that are **masked**.
        guidance_scale (`float`, *optional*):
            Guidance scale for classifier free guidance, setting the balance between the conditional logits (predicted
            from the prompts) and the unconditional logits (predicted without prompts).
    """
    encoder_outputs: Tuple[torch.FloatTensor] = ...
    attention_mask: torch.LongTensor = ...
    guidance_scale: float = ...


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int): # -> Tensor:
    """
    Shift input ids one token to the right.
    """
    ...

class MusicgenSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""
    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        ...

    def make_weights(self, num_embeddings: int, embedding_dim: int): # -> None:
        ...

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int): # -> Tensor:
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        ...

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = ...): # -> Tensor:
        ...



class MusicgenAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = ..., is_decoder: bool = ..., bias: bool = ..., is_causal: bool = ..., config: Optional[MusicgenConfig] = ...) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ..., attention_mask: Optional[torch.Tensor] = ..., layer_head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        ...



class MusicgenFlashAttention2(MusicgenAttention):
    """
    Musicgen flash attention module. This module inherits from `MusicgenAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """
    def __init__(self, *args, **kwargs) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ..., attention_mask: Optional[torch.Tensor] = ..., layer_head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        ...



class MusicgenSdpaAttention(MusicgenAttention):
    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ..., attention_mask: Optional[torch.Tensor] = ..., layer_head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        ...



MUSICGEN_ATTENTION_CLASSES = ...
class MusicgenDecoderLayer(nn.Module):
    def __init__(self, config: MusicgenDecoderConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., encoder_hidden_states: Optional[torch.Tensor] = ..., encoder_attention_mask: Optional[torch.Tensor] = ..., layer_head_mask: Optional[torch.Tensor] = ..., cross_attn_layer_head_mask: Optional[torch.Tensor] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ..., output_attentions: Optional[bool] = ..., use_cache: Optional[bool] = ...) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        ...



class MusicgenPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MusicgenDecoderConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn_2 = ...
    _supports_sdpa = ...


MUSICGEN_START_DOCSTRING = ...
MUSICGEN_INPUTS_DOCSTRING = ...
MUSICGEN_DECODER_INPUTS_DOCSTRING = ...
class MusicgenDecoder(MusicgenPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MusicgenDecoderLayer`]
    """
    def __init__(self, config: MusicgenDecoderConfig) -> None:
        ...

    def get_input_embeddings(self): # -> ModuleList | Module:
        ...

    def set_input_embeddings(self, value): # -> None:
        ...

    @add_start_docstrings_to_model_forward(MUSICGEN_DECODER_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor = ..., attention_mask: Optional[torch.Tensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.Tensor] = ..., cross_attn_head_mask: Optional[torch.Tensor] = ..., past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        ...



@add_start_docstrings("The bare Musicgen decoder model outputting raw hidden-states without any specific head on top.", MUSICGEN_START_DOCSTRING)
class MusicgenModel(MusicgenPreTrainedModel):
    def __init__(self, config: MusicgenDecoderConfig) -> None:
        ...

    def get_input_embeddings(self): # -> ModuleList | Module:
        ...

    def set_input_embeddings(self, value): # -> None:
        ...

    def get_decoder(self): # -> MusicgenDecoder:
        ...

    @add_start_docstrings_to_model_forward(MUSICGEN_DECODER_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor = ..., attention_mask: Optional[torch.Tensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.Tensor] = ..., cross_attn_head_mask: Optional[torch.Tensor] = ..., past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        ...



@add_start_docstrings("The MusicGen decoder model with a language modelling head on top.", MUSICGEN_START_DOCSTRING)
class MusicgenForCausalLM(MusicgenPreTrainedModel):
    def __init__(self, config: MusicgenDecoderConfig) -> None:
        ...

    def get_input_embeddings(self): # -> ModuleList | Module:
        ...

    def set_input_embeddings(self, value): # -> None:
        ...

    def get_output_embeddings(self): # -> ModuleList:
        ...

    def set_output_embeddings(self, new_embeddings): # -> None:
        ...

    def set_decoder(self, decoder): # -> None:
        ...

    def get_decoder(self): # -> MusicgenDecoder:
        ...

    @add_start_docstrings_to_model_forward(MUSICGEN_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor = ..., attention_mask: Optional[torch.Tensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.Tensor] = ..., cross_attn_head_mask: Optional[torch.Tensor] = ..., past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
                Returns:
        """
        ...

    def prepare_inputs_for_generation(self, input_ids, attention_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., head_mask=..., cross_attn_head_mask=..., past_key_values=..., use_cache=..., delay_pattern_mask=..., guidance_scale=..., **kwargs): # -> dict[str, Any]:
        ...

    def build_delay_pattern_mask(self, input_ids: torch.LongTensor, pad_token_id: int, max_length: int = ...): # -> tuple[Tensor, Tensor] | tuple[LongTensor, Tensor]:
        """Build a delayed pattern mask to the input_ids. Each codebook is offset by the previous codebook by
        one, giving a delayed pattern mask at the start of sequence and end of sequence. Take the example where there
        are 4 codebooks and a max sequence length of 8, we have the delayed pattern mask of shape `(codebooks,
        seq_len)`:
        - [P, -1, -1, -1, -1, P, P, P]
        - [P, P, -1, -1, -1, -1, P, P]
        - [P, P, P, -1, -1, -1, -1, P]
        - [P, P, P, P, -1, -1, -1, -1]
        where P is the special padding token id and -1 indicates that the token is valid for prediction. If we include
        a prompt (decoder input ids), the -1 positions indicate where new tokens should be predicted. Otherwise, the
        mask is set to the value in the prompt:
        - [P, a, b, -1, -1, P, P, P]
        - [P, P, c, d, -1, -1, P, P]
        - [P, P, P, e, f, -1, -1, P]
        - [P, P, P, P, g, h, -1, -1]
        where a-h indicate the input prompt (decoder input ids) that are offset by 1. Now, we only override the -1
        tokens in our prediction.
        """
        ...

    @staticmethod
    def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask): # -> Tensor:
        """Apply a delay pattern mask to the decoder input ids, only preserving predictions where
        the mask is set to -1, and otherwise setting to the value detailed in the mask."""
        ...

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor] = ..., generation_config: Optional[GenerationConfig] = ..., logits_processor: Optional[LogitsProcessorList] = ..., stopping_criteria: Optional[StoppingCriteriaList] = ..., synced_gpus: Optional[bool] = ..., streamer: Optional[BaseStreamer] = ..., **kwargs): # -> GenerateNonBeamOutput | LongTensor | Tensor:
        """

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        ...



@add_start_docstrings("The composite MusicGen model with a text encoder, audio encoder and Musicgen decoder, " "for music generation tasks with one or both of text and audio prompts.", MUSICGEN_START_DOCSTRING)
class MusicgenForConditionalGeneration(PreTrainedModel):
    config_class = MusicgenConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn_2 = ...
    _supports_sdpa = ...
    def __init__(self, config: Optional[MusicgenConfig] = ..., text_encoder: Optional[PreTrainedModel] = ..., audio_encoder: Optional[PreTrainedModel] = ..., decoder: Optional[MusicgenForCausalLM] = ...) -> None:
        ...

    def tie_weights(self): # -> None:
        ...

    def get_audio_encoder(self): # -> PreTrainedModel | None:
        ...

    def get_text_encoder(self): # -> PreTrainedModel | None:
        ...

    def get_encoder(self): # -> PreTrainedModel | None:
        ...

    def get_decoder(self): # -> MusicgenForCausalLM:
        ...

    def get_input_embeddings(self): # -> Module:
        ...

    def get_output_embeddings(self): # -> ModuleList:
        ...

    def set_output_embeddings(self, new_embeddings): # -> None:
        ...

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs): # -> tuple[Any | Self, dict[str, Any] | dict[str, Any | list[Any]] | Any] | Self:
        r"""
        Example:

        ```python
        >>> from transformers import MusicgenForConditionalGeneration

        >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        ```"""
        ...

    @classmethod
    def from_sub_models_pretrained(cls, text_encoder_pretrained_model_name_or_path: str = ..., audio_encoder_pretrained_model_name_or_path: str = ..., decoder_pretrained_model_name_or_path: str = ..., *model_args, **kwargs) -> PreTrainedModel:
        r"""
        Instantiate a text encoder, an audio encoder, and a MusicGen decoder from one, two or three base classes of the
        library from pretrained model checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            text_encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the text encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            audio_encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the audio encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the decoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            model_args (remaining positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the text encoder configuration, use the prefix *text_encoder_* for each configuration
                  parameter.
                - To update the audio encoder configuration, use the prefix *audio_encoder_* for each configuration
                  parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import MusicgenForConditionalGeneration

        >>> # initialize a musicgen model from a t5 text encoder, encodec audio encoder, and musicgen decoder
        >>> model = MusicgenForConditionalGeneration.from_sub_models_pretrained(
        ...     text_encoder_pretrained_model_name_or_path="google-t5/t5-base",
        ...     audio_encoder_pretrained_model_name_or_path="facebook/encodec_24khz",
        ...     decoder_pretrained_model_name_or_path="facebook/musicgen-small",
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./musicgen-ft")
        >>> # load fine-tuned model
        >>> model = MusicgenForConditionalGeneration.from_pretrained("./musicgen-ft")
        ```"""
        ...

    @add_start_docstrings_to_model_forward(MUSICGEN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.BoolTensor] = ..., input_values: Optional[torch.FloatTensor] = ..., padding_mask: Optional[torch.BoolTensor] = ..., decoder_input_ids: Optional[torch.LongTensor] = ..., decoder_attention_mask: Optional[torch.BoolTensor] = ..., encoder_outputs: Optional[Tuple[torch.FloatTensor]] = ..., past_key_values: Tuple[Tuple[torch.FloatTensor]] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., decoder_inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., **kwargs) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        Returns:

        Examples:
        ```python
        >>> from transformers import AutoProcessor, MusicgenForConditionalGeneration
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

        >>> inputs = processor(
        ...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
        ...     padding=True,
        ...     return_tensors="pt",
        ... )

        >>> pad_token_id = model.generation_config.pad_token_id
        >>> decoder_input_ids = (
        ...     torch.ones((inputs.input_ids.shape[0] * model.decoder.num_codebooks, 1), dtype=torch.long)
        ...     * pad_token_id
        ... )

        >>> logits = model(**inputs, decoder_input_ids=decoder_input_ids).logits
        >>> logits.shape  # (bsz * num_codebooks, tgt_len, vocab_size)
        torch.Size([8, 1, 2048])
        ```"""
        ...

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=..., attention_mask=..., head_mask=..., decoder_attention_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., use_cache=..., encoder_outputs=..., decoder_delay_pattern_mask=..., guidance_scale=..., **kwargs): # -> dict[str, Any]:
        ...

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor): # -> Tensor:
        ...

    def resize_token_embeddings(self, *args, **kwargs):
        ...

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor] = ..., generation_config: Optional[GenerationConfig] = ..., logits_processor: Optional[LogitsProcessorList] = ..., stopping_criteria: Optional[StoppingCriteriaList] = ..., synced_gpus: Optional[bool] = ..., streamer: Optional[BaseStreamer] = ..., **kwargs): # -> GenerateNonBeamOutput | LongTensor | Any | Tensor:
        """

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        ...

    def get_unconditional_inputs(self, num_samples=...): # -> MusicgenUnconditionalInput:
        """
        Helper function to get null inputs for unconditional generation, enabling the model to be used without the
        feature extractor or tokenizer.

        Args:
            num_samples (int, *optional*):
                Number of audio samples to unconditionally generate.
            max_new_tokens (int, *optional*):
                Number of tokens to generate for each sample. More tokens means longer audio samples, at the expense of
                longer inference (since more audio tokens need to be generated per sample).

        Example:
        ```python
        >>> from transformers import MusicgenForConditionalGeneration

        >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

        >>> # get the unconditional (or 'null') inputs for the model
        >>> unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
        >>> audio_samples = model.generate(**unconditional_inputs, max_new_tokens=256)
        ```"""
        ...
