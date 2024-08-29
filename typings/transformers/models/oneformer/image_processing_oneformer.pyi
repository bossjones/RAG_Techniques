"""
This type stub file was generated by pyright.
"""

import numpy as np
import torch
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType, is_torch_available

"""Image processor class for OneFormer."""
logger = ...
if is_torch_available():
    ...
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    ...

def get_max_height_width(images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = ...) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    ...

def make_pixel_mask(image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = ...) -> np.ndarray:
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`np.ndarray`):
            Image to make the pixel mask for.
        output_size (`Tuple[int, int]`):
            Output size of the mask.
    """
    ...

def binary_mask_to_rle(mask): # -> list[Any]:
    """
    Converts given binary mask of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        mask (`torch.Tensor` or `numpy.array`):
            A binary mask tensor of shape `(height, width)` where 0 denotes background and 1 denotes the target
            segment_id or class_id.
    Returns:
        `List`: Run-length encoded list of the binary mask. Refer to COCO API for more information about the RLE
        format.
    """
    ...

def convert_segmentation_to_rle(segmentation): # -> list[Any]:
    """
    Converts given segmentation map of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        segmentation (`torch.Tensor` or `numpy.array`):
            A segmentation map of shape `(height, width)` where each value denotes a segment or class id.
    Returns:
        `List[List]`: A list of lists, where each list is the run-length encoding of a segment / class id.
    """
    ...

def remove_low_and_no_objects(masks, scores, labels, object_mask_threshold, num_labels): # -> tuple[Any, Any, Any]:
    """
    Binarize the given masks using `object_mask_threshold`, it returns the associated values of `masks`, `scores` and
    `labels`.

    Args:
        masks (`torch.Tensor`):
            A tensor of shape `(num_queries, height, width)`.
        scores (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        labels (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        object_mask_threshold (`float`):
            A number between 0 and 1 used to binarize the masks.
    Raises:
        `ValueError`: Raised when the first dimension doesn't match in all input tensors.
    Returns:
        `Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: The `masks`, `scores` and `labels` without the region
        < `object_mask_threshold`.
    """
    ...

def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=..., overlap_mask_area_threshold=...): # -> tuple[Any | Literal[False], Any]:
    ...

def compute_segments(mask_probs, pred_scores, pred_labels, mask_threshold: float = ..., overlap_mask_area_threshold: float = ..., label_ids_to_fuse: Optional[Set[int]] = ..., target_size: Tuple[int, int] = ...): # -> tuple[Tensor, list[Dict[Any, Any]]]:
    ...

def convert_segmentation_map_to_binary_masks(segmentation_map: np.ndarray, instance_id_to_semantic_id: Optional[Dict[int, int]] = ..., ignore_index: Optional[int] = ..., reduce_labels: bool = ...): # -> tuple[NDArray[floating[_32Bit]], NDArray[signedinteger[_64Bit]]]:
    ...

def get_oneformer_resize_output_image_size(image: np.ndarray, size: Union[int, Tuple[int, int], List[int], Tuple[int]], max_size: Optional[int] = ..., default_to_square: bool = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ...) -> tuple:
    """
    Computes the output size given the desired size.

    Args:
        image (`np.ndarray`):
            The input image.
        size (`int` or `Tuple[int, int]` or `List[int]` or `Tuple[int]`):
            The size of the output image.
        max_size (`int`, *optional*):
            The maximum size of the output image.
        default_to_square (`bool`, *optional*, defaults to `True`):
            Whether to default to square if no size is provided.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `Tuple[int, int]`: The output size.
    """
    ...

def prepare_metadata(class_info): # -> dict[Any, Any]:
    ...

def load_metadata(repo_id, class_info_file): # -> Any:
    ...

class OneFormerImageProcessor(BaseImageProcessor):
    r"""
    Constructs a OneFormer image processor. The image processor can be used to prepare image(s), task input(s) and
    optional text inputs and targets for the model.

    This image processor inherits from [`BaseImageProcessor`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int`, *optional*, defaults to 800):
            Resize the input to the given size. Only has an effect if `do_resize` is set to `True`. If size is a
            sequence like `(width, height)`, output size will be matched to this. If size is an int, smaller edge of
            the image will be matched to this number. i.e, if `height > width`, then image will be rescaled to `(size *
            height / width, size)`.
        resample (`int`, *optional*, defaults to `Resampling.BILINEAR`):
            An optional resampling filter. This can be one of `PIL.Image.Resampling.NEAREST`,
            `PIL.Image.Resampling.BOX`, `PIL.Image.Resampling.BILINEAR`, `PIL.Image.Resampling.HAMMING`,
            `PIL.Image.Resampling.BICUBIC` or `PIL.Image.Resampling.LANCZOS`. Only has an effect if `do_resize` is set
            to `True`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the input to a certain `scale`.
        rescale_factor (`float`, *optional*, defaults to `1/ 255`):
            Rescale the input by the given factor. Only has an effect if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`int`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images. Defaults to the ImageNet mean.
        image_std (`int`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images. Defaults to the
            ImageNet std.
        ignore_index (`int`, *optional*):
            Label to be assigned to background pixels in segmentation maps. If provided, segmentation map pixels
            denoted with 0 (background) will be replaced with `ignore_index`.
        do_reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to decrement all label values of segmentation maps by 1. Usually used for datasets where 0
            is used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k).
            The background label will be replaced by `ignore_index`.
        repo_path (`str`, *optional*, defaults to `"shi-labs/oneformer_demo"`):
            Path to hub repo or local directory containing the JSON file with class information for the dataset.
            If unset, will look for `class_info_file` in the current working directory.
        class_info_file (`str`, *optional*):
            JSON file containing class information for the dataset. See `shi-labs/oneformer_demo/cityscapes_panoptic.json` for an example.
        num_text (`int`, *optional*):
            Number of text entries in the text input list.
    """
    model_input_names = ...
    def __init__(self, do_resize: bool = ..., size: Dict[str, int] = ..., resample: PILImageResampling = ..., do_rescale: bool = ..., rescale_factor: float = ..., do_normalize: bool = ..., image_mean: Union[float, List[float]] = ..., image_std: Union[float, List[float]] = ..., ignore_index: Optional[int] = ..., do_reduce_labels: bool = ..., repo_path: Optional[str] = ..., class_info_file: str = ..., num_text: Optional[int] = ..., **kwargs) -> None:
        ...

    def resize(self, image: np.ndarray, size: Dict[str, int], resample: PILImageResampling = ..., data_format=..., input_data_format: Optional[Union[str, ChannelDimension]] = ..., **kwargs) -> np.ndarray:
        """
        Resize the image to the given size. Size can be min_size (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.
        """
        ...

    def rescale(self, image: np.ndarray, rescale_factor: float, data_format: Optional[Union[str, ChannelDimension]] = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ...) -> np.ndarray:
        """
        Rescale the image by the given factor. image = image * rescale_factor.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            rescale_factor (`float`):
                The value to use for rescaling.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. If unset, is inferred from the input image. Can be
                one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        ...

    def convert_segmentation_map_to_binary_masks(self, segmentation_map: np.ndarray, instance_id_to_semantic_id: Optional[Dict[int, int]] = ..., ignore_index: Optional[int] = ..., reduce_labels: bool = ...): # -> tuple[NDArray[floating[_32Bit]], NDArray[signedinteger[_64Bit]]]:
        ...

    def __call__(self, images, task_inputs=..., segmentation_maps=..., **kwargs) -> BatchFeature:
        ...

    def preprocess(self, images: ImageInput, task_inputs: Optional[List[str]] = ..., segmentation_maps: Optional[ImageInput] = ..., instance_id_to_semantic_id: Optional[Dict[int, int]] = ..., do_resize: Optional[bool] = ..., size: Optional[Dict[str, int]] = ..., resample: PILImageResampling = ..., do_rescale: Optional[bool] = ..., rescale_factor: Optional[float] = ..., do_normalize: Optional[bool] = ..., image_mean: Optional[Union[float, List[float]]] = ..., image_std: Optional[Union[float, List[float]]] = ..., ignore_index: Optional[int] = ..., do_reduce_labels: Optional[bool] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., data_format: Union[str, ChannelDimension] = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ..., **kwargs) -> BatchFeature:
        ...

    def pad(self, images: List[np.ndarray], constant_values: Union[float, Iterable[float]] = ..., return_pixel_mask: bool = ..., return_tensors: Optional[Union[str, TensorType]] = ..., data_format: Optional[ChannelDimension] = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ...) -> BatchFeature:
        """
        Pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width
        in the batch and optionally returns their corresponding pixel mask.

        Args:
            image (`np.ndarray`):
                Image to pad.
            constant_values (`float` or `Iterable[float]`, *optional*):
                The value to use for the padding if `mode` is `"constant"`.
            return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether to return a pixel mask.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        ...

    def get_semantic_annotations(self, label, num_class_obj): # -> tuple[NDArray[Any], NDArray[Any], Any]:
        ...

    def get_instance_annotations(self, label, num_class_obj): # -> tuple[NDArray[Any], NDArray[Any], Any]:
        ...

    def get_panoptic_annotations(self, label, num_class_obj): # -> tuple[NDArray[Any], NDArray[Any], Any]:
        ...

    def encode_inputs(self, pixel_values_list: List[ImageInput], task_inputs: List[str], segmentation_maps: ImageInput = ..., instance_id_to_semantic_id: Optional[Union[List[Dict[int, int]], Dict[int, int]]] = ..., ignore_index: Optional[int] = ..., reduce_labels: bool = ..., return_tensors: Optional[Union[str, TensorType]] = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ...): # -> BatchFeature:
        """
        Pad images up to the largest image in a batch and create a corresponding `pixel_mask`.

        OneFormer addresses semantic segmentation with a mask classification paradigm, thus input segmentation maps
        will be converted to lists of binary masks and their respective labels. Let's see an example, assuming
        `segmentation_maps = [[2,6,7,9]]`, the output will contain `mask_labels =
        [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]` (four binary masks) and `class_labels = [2,6,7,9]`, the labels for
        each mask.

        Args:
            pixel_values_list (`List[ImageInput]`):
                List of images (pixel values) to be padded. Each image should be a tensor of shape `(channels, height,
                width)`.

            task_inputs (`List[str]`):
                List of task values.

            segmentation_maps (`ImageInput`, *optional*):
                The corresponding semantic segmentation maps with the pixel-wise annotations.

             (`bool`, *optional*, defaults to `True`):
                Whether or not to pad images up to the largest image in a batch and create a pixel mask.

                If left to the default, will return a pixel mask that is:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            instance_id_to_semantic_id (`List[Dict[int, int]]` or `Dict[int, int]`, *optional*):
                A mapping between object instance ids and class ids. If passed, `segmentation_maps` is treated as an
                instance segmentation map where each pixel represents an instance id. Can be provided as a single
                dictionary with a global/dataset-level mapping or as a list of dictionaries (one per image), to map
                instance ids in each image separately.

            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):
                If set, will return tensors instead of NumPy arrays. If set to `'pt'`, return PyTorch `torch.Tensor`
                objects.

            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred from the input
                image.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
            - **pixel_mask** -- Pixel mask to be fed to a model (when `=True` or if `pixel_mask` is in
              `self.model_input_names`).
            - **mask_labels** -- Optional list of mask labels of shape `(labels, height, width)` to be fed to a model
              (when `annotations` are provided).
            - **class_labels** -- Optional list of class labels of shape `(labels)` to be fed to a model (when
              `annotations` are provided). They identify the labels of `mask_labels`, e.g. the label of
              `mask_labels[i][j]` if `class_labels[i][j]`.
            - **text_inputs** -- Optional list of text string entries to be fed to a model (when `annotations` are
              provided). They identify the binary masks present in the image.
        """
        ...

    def post_process_semantic_segmentation(self, outputs, target_sizes: Optional[List[Tuple[int, int]]] = ...) -> torch.Tensor:
        """
        Converts the output of [`MaskFormerForInstanceSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`MaskFormerForInstanceSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
        Returns:
            `List[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        ...

    def post_process_instance_segmentation(self, outputs, task_type: str = ..., is_demo: bool = ..., threshold: float = ..., mask_threshold: float = ..., overlap_mask_area_threshold: float = ..., target_sizes: Optional[List[Tuple[int, int]]] = ..., return_coco_annotation: Optional[bool] = ...): # -> list[Dict[str, Tensor]]:
        """
        Converts the output of [`OneFormerForUniversalSegmentationOutput`] into image instance segmentation
        predictions. Only supports PyTorch.

        Args:
            outputs ([`OneFormerForUniversalSegmentationOutput`]):
                The outputs from [`OneFormerForUniversalSegmentationOutput`].
            task_type (`str`, *optional)*, defaults to "instance"):
                The post processing depends on the task token input. If the `task_type` is "panoptic", we need to
                ignore the stuff predictions.
            is_demo (`bool`, *optional)*, defaults to `True`):
                Whether the model is in demo mode. If true, use threshold to predict final masks.
            threshold (`float`, *optional*, defaults to 0.5):
                The probability score threshold to keep predicted instance masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
                The overlap mask area threshold to merge or discard small disconnected parts within each binary
                instance mask.
            target_sizes (`List[Tuple]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction in batch. If left to None, predictions will not be
                resized.
            return_coco_annotation (`bool`, *optional)*, defaults to `False`):
                Whether to return predictions in COCO format.

        Returns:
            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- a tensor of shape `(height, width)` where each pixel represents a `segment_id`, set
              to `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized
              to the corresponding `target_sizes` entry.
            - **segments_info** -- A dictionary that contains additional information on each segment.
                - **id** -- an integer representing the `segment_id`.
                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
                - **was_fused** -- a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
                  Multiple instances of the same class / label were fused and assigned a single `segment_id`.
                - **score** -- Prediction score of segment with `segment_id`.
        """
        ...

    def post_process_panoptic_segmentation(self, outputs, threshold: float = ..., mask_threshold: float = ..., overlap_mask_area_threshold: float = ..., label_ids_to_fuse: Optional[Set[int]] = ..., target_sizes: Optional[List[Tuple[int, int]]] = ...) -> List[Dict]:
        """
        Converts the output of [`MaskFormerForInstanceSegmentationOutput`] into image panoptic segmentation
        predictions. Only supports PyTorch.

        Args:
            outputs ([`MaskFormerForInstanceSegmentationOutput`]):
                The outputs from [`MaskFormerForInstanceSegmentation`].
            threshold (`float`, *optional*, defaults to 0.5):
                The probability score threshold to keep predicted instance masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
                The overlap mask area threshold to merge or discard small disconnected parts within each binary
                instance mask.
            label_ids_to_fuse (`Set[int]`, *optional*):
                The labels in this state will have all their instances be fused together. For instance we could say
                there can only be one sky in an image, but several persons, so the label ID for sky would be in that
                set, but not the one for person.
            target_sizes (`List[Tuple]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction in batch. If left to None, predictions will not be
                resized.

        Returns:
            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- a tensor of shape `(height, width)` where each pixel represents a `segment_id`, set
              to `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized
              to the corresponding `target_sizes` entry.
            - **segments_info** -- A dictionary that contains additional information on each segment.
                - **id** -- an integer representing the `segment_id`.
                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
                - **was_fused** -- a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
                  Multiple instances of the same class / label were fused and assigned a single `segment_id`.
                - **score** -- Prediction score of segment with `segment_id`.
        """
        ...
