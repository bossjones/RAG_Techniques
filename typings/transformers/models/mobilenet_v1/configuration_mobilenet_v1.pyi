"""
This type stub file was generated by pyright.
"""

from typing import Mapping
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

""" MobileNetV1 model configuration"""
logger = ...
class MobileNetV1Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MobileNetV1Model`]. It is used to instantiate a
    MobileNetV1 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MobileNetV1
    [google/mobilenet_v1_1.0_224](https://huggingface.co/google/mobilenet_v1_1.0_224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        depth_multiplier (`float`, *optional*, defaults to 1.0):
            Shrinks or expands the number of channels in each layer. Default is 1.0, which starts the network with 32
            channels. This is sometimes also called "alpha" or "width multiplier".
        min_depth (`int`, *optional*, defaults to 8):
            All layers will have at least this many channels.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu6"`):
            The non-linear activation function (function or string) in the Transformer encoder and convolution layers.
        tf_padding (`bool`, *optional*, defaults to `True`):
            Whether to use TensorFlow padding rules on the convolution layers.
        classifier_dropout_prob (`float`, *optional*, defaults to 0.999):
            The dropout ratio for attached classifiers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 0.001):
            The epsilon used by the layer normalization layers.

    Example:

    ```python
    >>> from transformers import MobileNetV1Config, MobileNetV1Model

    >>> # Initializing a "mobilenet_v1_1.0_224" style configuration
    >>> configuration = MobileNetV1Config()

    >>> # Initializing a model from the "mobilenet_v1_1.0_224" style configuration
    >>> model = MobileNetV1Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    def __init__(self, num_channels=..., image_size=..., depth_multiplier=..., min_depth=..., hidden_act=..., tf_padding=..., classifier_dropout_prob=..., initializer_range=..., layer_norm_eps=..., **kwargs) -> None:
        ...



class MobileNetV1OnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        ...

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        ...

    @property
    def atol_for_validation(self) -> float:
        ...
