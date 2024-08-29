"""
This type stub file was generated by pyright.
"""

import torch
from typing import List, Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_glpn import GLPNConfig

""" PyTorch GLPN model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
def drop_path(input: torch.Tensor, drop_prob: float = ..., training: bool = ...) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    ...

class GLPNDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = ...) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...

    def extra_repr(self) -> str:
        ...



class GLPNOverlapPatchEmbeddings(nn.Module):
    """Construct the overlapping patch embeddings."""
    def __init__(self, patch_size, stride, num_channels, hidden_size) -> None:
        ...

    def forward(self, pixel_values): # -> tuple[Any, Any, Any]:
        ...



class GLPNEfficientSelfAttention(nn.Module):
    """SegFormer's efficient self-attention mechanism. Employs the sequence reduction process introduced in the [PvT
    paper](https://arxiv.org/abs/2102.12122)."""
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio) -> None:
        ...

    def transpose_for_scores(self, hidden_states):
        ...

    def forward(self, hidden_states, height, width, output_attentions=...): # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...



class GLPNSelfOutput(nn.Module):
    def __init__(self, config, hidden_size) -> None:
        ...

    def forward(self, hidden_states, input_tensor): # -> Any:
        ...



class GLPNAttention(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio) -> None:
        ...

    def prune_heads(self, heads): # -> None:
        ...

    def forward(self, hidden_states, height, width, output_attentions=...): # -> Any:
        ...



class GLPNDWConv(nn.Module):
    def __init__(self, dim=...) -> None:
        ...

    def forward(self, hidden_states, height, width): # -> Any:
        ...



class GLPNMixFFN(nn.Module):
    def __init__(self, config, in_features, hidden_features=..., out_features=...) -> None:
        ...

    def forward(self, hidden_states, height, width): # -> Any:
        ...



class GLPNLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""
    def __init__(self, config, hidden_size, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio) -> None:
        ...

    def forward(self, hidden_states, height, width, output_attentions=...): # -> Any:
        ...



class GLPNEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, pixel_values, output_attentions=..., output_hidden_states=..., return_dict=...): # -> tuple[Any, ...] | BaseModelOutput:
        ...



class GLPNPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GLPNConfig
    base_model_prefix = ...
    main_input_name = ...


GLPN_START_DOCSTRING = ...
GLPN_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare GLPN encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top.", GLPN_START_DOCSTRING)
class GLPNModel(GLPNPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(GLPN_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC, modality="vision", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutput]:
        ...



class GLPNSelectiveFeatureFusion(nn.Module):
    """
    Selective Feature Fusion module, as explained in the [paper](https://arxiv.org/abs/2201.07436) (section 3.4). This
    module adaptively selects and integrates local and global features by attaining an attention map for each feature.
    """
    def __init__(self, in_channel=...) -> None:
        ...

    def forward(self, local_features, global_features):
        ...



class GLPNDecoderStage(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        ...

    def forward(self, hidden_state, residual=...): # -> Any:
        ...



class GLPNDecoder(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        ...



class SiLogLoss(nn.Module):
    r"""
    Implements the Scale-invariant log scale loss [Eigen et al., 2014](https://arxiv.org/abs/1406.2283).

    $$L=\frac{1}{n} \sum_{i} d_{i}^{2}-\frac{1}{2 n^{2}}\left(\sum_{i} d_{i}^{2}\right)$$ where $d_{i}=\log y_{i}-\log
    y_{i}^{*}$.

    """
    def __init__(self, lambd=...) -> None:
        ...

    def forward(self, pred, target): # -> Tensor:
        ...



class GLPNDepthEstimationHead(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        ...



@add_start_docstrings("""GLPN Model transformer with a lightweight depth estimation head on top e.g. for KITTI, NYUv2.""", GLPN_START_DOCSTRING)
class GLPNForDepthEstimation(GLPNPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(GLPN_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, labels: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], DepthEstimatorOutput]:
        r"""
        labels (`torch.FloatTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, GLPNForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("vinvino02/glpn-kitti")
        >>> model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ...     predicted_depth = outputs.predicted_depth

        >>> # interpolate to original size
        >>> prediction = torch.nn.functional.interpolate(
        ...     predicted_depth.unsqueeze(1),
        ...     size=image.size[::-1],
        ...     mode="bicubic",
        ...     align_corners=False,
        ... )

        >>> # visualize the prediction
        >>> output = prediction.squeeze().cpu().numpy()
        >>> formatted = (output * 255 / np.max(output)).astype("uint8")
        >>> depth = Image.fromarray(formatted)
        ```"""
        ...
