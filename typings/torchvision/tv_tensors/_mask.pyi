"""
This type stub file was generated by pyright.
"""

import torch
from typing import Any, Optional, Union
from ._tv_tensor import TVTensor

class Mask(TVTensor):
    """:class:`torch.Tensor` subclass for segmentation and detection masks.

    Args:
        data (tensor-like, PIL.Image.Image): Any data that can be turned into a tensor with :func:`torch.as_tensor` as
            well as PIL images.
        dtype (torch.dtype, optional): Desired data type. If omitted, will be inferred from
            ``data``.
        device (torch.device, optional): Desired device. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the mask is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """
    def __new__(cls, data: Any, *, dtype: Optional[torch.dtype] = ..., device: Optional[Union[torch.device, str, int]] = ..., requires_grad: Optional[bool] = ...) -> Mask:
        ...
