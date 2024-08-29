"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional, TYPE_CHECKING
from .base import HfQuantizer
from ..modeling_utils import PreTrainedModel
from ..utils import is_torch_available
from ..utils.quantization_config import QuantizationConfigMixin

if TYPE_CHECKING:
    ...
if is_torch_available():
    ...
logger = ...
class GptqHfQuantizer(HfQuantizer):
    """
    Quantizer of the GPTQ method - for GPTQ the quantizer support calibration of the model through
    `auto_gptq` package. Quantization is done under the hood for users if they load a non-prequantized model.
    """
    requires_calibration = ...
    required_packages = ...
    optimum_quantizer = ...
    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs) -> None:
        ...

    def validate_environment(self, *args, **kwargs): # -> None:
        ...

    def update_torch_dtype(self, torch_dtype: torch.dtype) -> torch.dtype:
        ...

    @property
    def is_trainable(self, model: Optional[PreTrainedModel] = ...): # -> Literal[True]:
        ...

    @property
    def is_serializable(self): # -> Literal[True]:
        ...
