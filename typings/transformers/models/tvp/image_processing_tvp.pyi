"""
This type stub file was generated by pyright.
"""

import numpy as np
import PIL
from typing import Dict, Iterable, List, Optional, Tuple, Union
from ...image_processing_utils import BaseImageProcessor
from ...image_transforms import PaddingMode
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType, is_vision_available

"""Image processor class for TVP."""
if is_vision_available():
    ...
logger = ...
def make_batched(videos) -> List[List[ImageInput]]:
    ...

def get_resize_output_image_size(input_image: np.ndarray, max_size: int = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ...) -> Tuple[int, int]:
    ...

class TvpImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Tvp image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"longest_edge": 448}`):
            Size of the output image after resizing. The longest edge of the image will be resized to
            `size["longest_edge"]` while maintaining the aspect ratio of the original image. Can be overriden by
            `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by the `do_center_crop`
            parameter in the `preprocess` method.
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 448, "width": 448}`):
            Size of the image after applying the center crop. Can be overridden by the `crop_size` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Defines the scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter
            in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess` method.
        pad_size (`Dict[str, int]`, *optional*, defaults to `{"height": 448, "width": 448}`):
            Size of the image after applying the padding. Can be overridden by the `pad_size` parameter in the
            `preprocess` method.
        constant_values (`Union[float, Iterable[float]]`, *optional*, defaults to 0):
            The fill value to use when padding the image.
        pad_mode (`PaddingMode`, *optional*, defaults to `PaddingMode.CONSTANT`):
            Use what kind of mode in padding.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        do_flip_channel_order (`bool`, *optional*, defaults to `True`):
            Whether to flip the color channels from RGB to BGR. Can be overridden by the `do_flip_channel_order`
            parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """
    model_input_names = ...
    def __init__(self, do_resize: bool = ..., size: Dict[str, int] = ..., resample: PILImageResampling = ..., do_center_crop: bool = ..., crop_size: Dict[str, int] = ..., do_rescale: bool = ..., rescale_factor: Union[int, float] = ..., do_pad: bool = ..., pad_size: Dict[str, int] = ..., constant_values: Union[float, Iterable[float]] = ..., pad_mode: PaddingMode = ..., do_normalize: bool = ..., do_flip_channel_order: bool = ..., image_mean: Optional[Union[float, List[float]]] = ..., image_std: Optional[Union[float, List[float]]] = ..., **kwargs) -> None:
        ...

    def resize(self, image: np.ndarray, size: Dict[str, int], resample: PILImageResampling = ..., data_format: Optional[Union[str, ChannelDimension]] = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ..., **kwargs) -> np.ndarray:
        """
        Resize an image.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image. If `size` is of the form `{"height": h, "width": w}`, the output image will
                have the size `(h, w)`. If `size` is of the form `{"longest_edge": s}`, the output image will have its
                longest edge of length `s` while keeping the aspect ratio of the original image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        ...

    def pad_image(self, image: np.ndarray, pad_size: Dict[str, int] = ..., constant_values: Union[float, Iterable[float]] = ..., pad_mode: PaddingMode = ..., data_format: Optional[Union[str, ChannelDimension]] = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ..., **kwargs): # -> ndarray[Any, Any]:
        """
        Pad an image with zeros to the given size.

        Args:
            image (`np.ndarray`):
                Image to pad.
            pad_size (`Dict[str, int]`)
                Size of the output image with pad.
            constant_values (`Union[float, Iterable[float]]`)
                The fill value to use when padding the image.
            pad_mode (`PaddingMode`)
                The pad mode, default to PaddingMode.CONSTANT
            data_format (`ChannelDimension` or `str`, *optional*)
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        ...

    def preprocess(self, videos: Union[ImageInput, List[ImageInput], List[List[ImageInput]]], do_resize: bool = ..., size: Dict[str, int] = ..., resample: PILImageResampling = ..., do_center_crop: bool = ..., crop_size: Dict[str, int] = ..., do_rescale: bool = ..., rescale_factor: float = ..., do_pad: bool = ..., pad_size: Dict[str, int] = ..., constant_values: Union[float, Iterable[float]] = ..., pad_mode: PaddingMode = ..., do_normalize: bool = ..., do_flip_channel_order: bool = ..., image_mean: Optional[Union[float, List[float]]] = ..., image_std: Optional[Union[float, List[float]]] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., data_format: ChannelDimension = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ..., **kwargs) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            videos (`ImageInput` or `List[ImageInput]` or `List[List[ImageInput]]`):
                Frames to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after applying resize.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
                has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_centre_crop`):
                Whether to centre crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the image after applying the centre crop.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_pad (`bool`, *optional*, defaults to `True`):
                Whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess` method.
            pad_size (`Dict[str, int]`, *optional*, defaults to `{"height": 448, "width": 448}`):
                Size of the image after applying the padding. Can be overridden by the `pad_size` parameter in the
                `preprocess` method.
            constant_values (`Union[float, Iterable[float]]`, *optional*, defaults to 0):
                The fill value to use when padding the image.
            pad_mode (`PaddingMode`, *optional*, defaults to "PaddingMode.CONSTANT"):
                Use what kind of mode in padding.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            do_flip_channel_order (`bool`, *optional*, defaults to `self.do_flip_channel_order`):
                Whether to flip the channel order of the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                    - Unset: Use the inferred channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        ...
