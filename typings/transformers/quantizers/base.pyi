"""
This type stub file was generated by pyright.
"""

import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union
from ..utils import is_torch_available
from ..utils.quantization_config import QuantizationConfigMixin
from ..modeling_utils import PreTrainedModel

if TYPE_CHECKING:
    ...
if is_torch_available():
    ...
class HfQuantizer(ABC):
    """
    Abstract class of the HuggingFace quantizer. Supports for now quantizing HF transformers models for inference and/or quantization.
    This class is used only for transformers.PreTrainedModel.from_pretrained and cannot be easily used outside the scope of that method
    yet.

    Attributes
        quantization_config (`transformers.utils.quantization_config.QuantizationConfigMixin`):
            The quantization config that defines the quantization parameters of your model that you want to quantize.
        modules_to_not_convert (`List[str]`, *optional*):
            The list of module names to not convert when quantizing the model.
        required_packages (`List[str]`, *optional*):
            The list of required pip packages to install prior to using the quantizer
        requires_calibration (`bool`):
            Whether the quantization method requires to calibrate the model before using it.
        requires_parameters_quantization (`bool`):
            Whether the quantization method requires to create a new Parameter. For example, for bitsandbytes, it is
            required to create a new xxxParameter in order to properly quantize the model.
    """
    requires_calibration = ...
    required_packages = ...
    requires_parameters_quantization = ...
    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs) -> None:
        ...

    def update_torch_dtype(self, torch_dtype: torch.dtype) -> torch.dtype:
        """
        Some quantization methods require to explicitly set the dtype of the model to a
        target dtype. You need to override this method in case you want to make sure that behavior is
        preserved

        Args:
            torch_dtype (`torch.dtype`):
                The input dtype that is passed in `from_pretrained`
        """
        ...

    def update_device_map(self, device_map: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Override this method if you want to pass a override the existing device map with a new
        one. E.g. for bitsandbytes, since `accelerate` is a hard requirement, if no device_map is
        passed, the device_map is set to `"auto"``

        Args:
            device_map (`Union[dict, str]`, *optional*):
                The device_map that is passed through the `from_pretrained` method.
        """
        ...

    def adjust_target_dtype(self, torch_dtype: torch.dtype) -> torch.dtype:
        """
        Override this method if you want to adjust the `target_dtype` variable used in `from_pretrained`
        to compute the device_map in case the device_map is a `str`. E.g. for bitsandbytes we force-set `target_dtype`
        to `torch.int8` and for 4-bit we pass a custom enum `accelerate.CustomDtype.int4`.

        Args:
            torch_dtype (`torch.dtype`, *optional*):
                The torch_dtype that is used to compute the device_map.
        """
        ...

    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        """
        Override this method if you want to adjust the `missing_keys`.

        Args:
            missing_keys (`List[str]`, *optional*):
                The list of missing keys in the checkpoint compared to the state dict of the model
        """
        ...

    def get_special_dtypes_update(self, model, torch_dtype: torch.dtype) -> Dict[str, torch.dtype]:
        """
        returns dtypes for modules that are not quantized - used for the computation of the device_map in case
        one passes a str as a device_map. The method will use the `modules_to_not_convert` that is modified
        in `_process_model_before_weight_loading`.

        Args:
            model (`~transformers.PreTrainedModel`):
                The model to quantize
            torch_dtype (`torch.dtype`):
                The dtype passed in `from_pretrained` method.
        """
        ...

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        """adjust max_memory argument for infer_auto_device_map() if extra memory is needed for quantization"""
        ...

    def check_quantized_param(self, model: PreTrainedModel, param_value: torch.Tensor, param_name: str, state_dict: Dict[str, Any], **kwargs) -> bool:
        """
        checks if a loaded state_dict component is part of quantized param + some validation; only defined if
        requires_parameters_quantization == True for quantization methods that require to create a new parameters
        for quantization.
        """
        ...

    def create_quantized_param(self, *args, **kwargs) -> torch.nn.Parameter:
        """
        takes needed components from state_dict and creates quantized param; only applicable if
        requires_parameters_quantization == True
        """
        ...

    def validate_environment(self, *args, **kwargs): # -> None:
        """
        This method is used to potentially check for potential conflicts with arguments that are
        passed in `from_pretrained`. You need to define it for all future quantizers that are integrated with transformers.
        If no explicit check are needed, simply return nothing.
        """
        ...

    def preprocess_model(self, model: PreTrainedModel, **kwargs): # -> None:
        """
        Setting model attributes and/or converting model before weights loading. At this point
        the model should be initialized on the meta device so you can freely manipulate the skeleton
        of the model in order to replace modules in-place. Make sure to override the abstract method `_process_model_before_weight_loading`.

        Args:
            model (`~transformers.PreTrainedModel`):
                The model to quantize
            kwargs (`dict`, *optional*):
                The keyword arguments that are passed along `_process_model_before_weight_loading`.
        """
        ...

    def postprocess_model(self, model: PreTrainedModel, **kwargs): # -> None:
        """
        Post-process the model post weights loading.
        Make sure to override the abstract method `_process_model_after_weight_loading`.

        Args:
            model (`~transformers.PreTrainedModel`):
                The model to quantize
            kwargs (`dict`, *optional*):
                The keyword arguments that are passed along `_process_model_after_weight_loading`.
        """
        ...

    @property
    @abstractmethod
    def is_serializable(self): # -> None:
        ...

    @property
    @abstractmethod
    def is_trainable(self): # -> None:
        ...
