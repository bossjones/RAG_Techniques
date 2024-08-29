"""
This type stub file was generated by pyright.
"""

import os
from typing import List, TYPE_CHECKING, Union
from ...configuration_utils import PretrainedConfig

""" ALIGN model configuration"""
if TYPE_CHECKING:
    ...
logger = ...
class AlignTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AlignTextModel`]. It is used to instantiate a
    ALIGN text encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the text encoder of the ALIGN
    [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture. The default values here are
    copied from BERT.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the Align Text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`AlignTextModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`AlignTextModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.

    Example:

    ```python
    >>> from transformers import AlignTextConfig, AlignTextModel

    >>> # Initializing a AlignTextConfig with kakaobrain/align-base style configuration
    >>> configuration = AlignTextConfig()

    >>> # Initializing a AlignTextModel (with random weights) from the kakaobrain/align-base style configuration
    >>> model = AlignTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    def __init__(self, vocab_size=..., hidden_size=..., num_hidden_layers=..., num_attention_heads=..., intermediate_size=..., hidden_act=..., hidden_dropout_prob=..., attention_probs_dropout_prob=..., max_position_embeddings=..., type_vocab_size=..., initializer_range=..., layer_norm_eps=..., pad_token_id=..., position_embedding_type=..., use_cache=..., **kwargs) -> None:
        ...

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> PretrainedConfig:
        ...



class AlignVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AlignVisionModel`]. It is used to instantiate a
    ALIGN vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the ALIGN
    [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture. The default values are copied
    from EfficientNet (efficientnet-b7)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 600):
            The input image size.
        width_coefficient (`float`, *optional*, defaults to 2.0):
            Scaling coefficient for network width at each stage.
        depth_coefficient (`float`, *optional*, defaults to 3.1):
            Scaling coefficient for network depth at each stage.
        depth_divisor `int`, *optional*, defaults to 8):
            A unit of network width.
        kernel_sizes (`List[int]`, *optional*, defaults to `[3, 3, 5, 3, 5, 5, 3]`):
            List of kernel sizes to be used in each block.
        in_channels (`List[int]`, *optional*, defaults to `[32, 16, 24, 40, 80, 112, 192]`):
            List of input channel sizes to be used in each block for convolutional layers.
        out_channels (`List[int]`, *optional*, defaults to `[16, 24, 40, 80, 112, 192, 320]`):
            List of output channel sizes to be used in each block for convolutional layers.
        depthwise_padding (`List[int]`, *optional*, defaults to `[]`):
            List of block indices with square padding.
        strides (`List[int]`, *optional*, defaults to `[1, 2, 2, 2, 1, 2, 1]`):
            List of stride sizes to be used in each block for convolutional layers.
        num_block_repeats (`List[int]`, *optional*, defaults to `[1, 2, 2, 3, 3, 4, 1]`):
            List of the number of times each block is to repeated.
        expand_ratios (`List[int]`, *optional*, defaults to `[1, 6, 6, 6, 6, 6, 6]`):
            List of scaling coefficient of each block.
        squeeze_expansion_ratio (`float`, *optional*, defaults to 0.25):
            Squeeze expansion ratio.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in each block. If string, `"gelu"`, `"relu"`,
            `"selu", `"gelu_new"`, `"silu"` and `"mish"` are supported.
        hiddem_dim (`int`, *optional*, defaults to 1280):
            The hidden dimension of the layer before the classification head.
        pooling_type (`str` or `function`, *optional*, defaults to `"mean"`):
            Type of final pooling to be applied before the dense classification head. Available options are [`"mean"`,
            `"max"`]
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        batch_norm_eps (`float`, *optional*, defaults to 1e-3):
            The epsilon used by the batch normalization layers.
        batch_norm_momentum (`float`, *optional*, defaults to 0.99):
            The momentum used by the batch normalization layers.
        drop_connect_rate (`float`, *optional*, defaults to 0.2):
            The drop rate for skip connections.

    Example:

    ```python
    >>> from transformers import AlignVisionConfig, AlignVisionModel

    >>> # Initializing a AlignVisionConfig with kakaobrain/align-base style configuration
    >>> configuration = AlignVisionConfig()

    >>> # Initializing a AlignVisionModel (with random weights) from the kakaobrain/align-base style configuration
    >>> model = AlignVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    def __init__(self, num_channels: int = ..., image_size: int = ..., width_coefficient: float = ..., depth_coefficient: float = ..., depth_divisor: int = ..., kernel_sizes: List[int] = ..., in_channels: List[int] = ..., out_channels: List[int] = ..., depthwise_padding: List[int] = ..., strides: List[int] = ..., num_block_repeats: List[int] = ..., expand_ratios: List[int] = ..., squeeze_expansion_ratio: float = ..., hidden_act: str = ..., hidden_dim: int = ..., pooling_type: str = ..., initializer_range: float = ..., batch_norm_eps: float = ..., batch_norm_momentum: float = ..., drop_connect_rate: float = ..., **kwargs) -> None:
        ...

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> PretrainedConfig:
        ...



class AlignConfig(PretrainedConfig):
    r"""
    [`AlignConfig`] is the configuration class to store the configuration of a [`AlignModel`]. It is used to
    instantiate a ALIGN model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the ALIGN
    [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`AlignTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`AlignVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 640):
            Dimentionality of text and vision projection layers.
        temperature_init_value (`float`, *optional*, defaults to 1.0):
            The inital value of the *temperature* paramter. Default is used as per the original ALIGN implementation.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import AlignConfig, AlignModel

    >>> # Initializing a AlignConfig with kakaobrain/align-base style configuration
    >>> configuration = AlignConfig()

    >>> # Initializing a AlignModel (with random weights) from the kakaobrain/align-base style configuration
    >>> model = AlignModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a AlignConfig from a AlignTextConfig and a AlignVisionConfig
    >>> from transformers import AlignTextConfig, AlignVisionConfig

    >>> # Initializing ALIGN Text and Vision configurations
    >>> config_text = AlignTextConfig()
    >>> config_vision = AlignVisionConfig()

    >>> config = AlignConfig.from_text_vision_configs(config_text, config_vision)
    ```"""
    model_type = ...
    def __init__(self, text_config=..., vision_config=..., projection_dim=..., temperature_init_value=..., initializer_range=..., **kwargs) -> None:
        ...

    @classmethod
    def from_text_vision_configs(cls, text_config: AlignTextConfig, vision_config: AlignVisionConfig, **kwargs): # -> Self:
        r"""
        Instantiate a [`AlignConfig`] (or a derived class) from align text model configuration and align vision model
        configuration.

        Returns:
            [`AlignConfig`]: An instance of a configuration object
        """
        ...
