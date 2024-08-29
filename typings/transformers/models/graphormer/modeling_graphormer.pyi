"""
This type stub file was generated by pyright.
"""

import torch
import torch.nn as nn
from typing import Iterable, Iterator, List, Optional, Tuple, Union
from ...modeling_outputs import BaseModelOutputWithNoAttention, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from .configuration_graphormer import GraphormerConfig

""" PyTorch Graphormer model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
def quant_noise(module: nn.Module, p: float, block_size: int): # -> Module | Linear | Embedding | Conv2d:
    """
    From:
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/quant_noise.py

    Wraps modules and applies quantization noise to the weights for subsequent quantization with Iterative Product
    Quantization as described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights, see "And the Bit Goes Down:
          Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper which consists in randomly dropping
          blocks
    """
    ...

class LayerDropModuleList(nn.ModuleList):
    """
    From:
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/layer_drop.py
    A LayerDrop implementation based on [`torch.nn.ModuleList`]. LayerDrop as described in
    https://arxiv.org/abs/1909.11556.

    We refresh the choice of which layers to drop every time we iterate over the LayerDropModuleList instance. During
    evaluation we always iterate over all layers.

    Usage:

    ```python
    layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
    for layer in layers:  # this might iterate over layers 1 and 3
        x = layer(x)
    for layer in layers:  # this might iterate over all layers
        x = layer(x)
    for layer in layers:  # this might not iterate over any layers
        x = layer(x)
    ```

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """
    def __init__(self, p: float, modules: Optional[Iterable[nn.Module]] = ...) -> None:
        ...

    def __iter__(self) -> Iterator[nn.Module]:
        ...



class GraphormerGraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """
    def __init__(self, config: GraphormerConfig) -> None:
        ...

    def forward(self, input_nodes: torch.LongTensor, in_degree: torch.LongTensor, out_degree: torch.LongTensor) -> torch.Tensor:
        ...



class GraphormerGraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """
    def __init__(self, config: GraphormerConfig) -> None:
        ...

    def forward(self, input_nodes: torch.LongTensor, attn_bias: torch.Tensor, spatial_pos: torch.LongTensor, input_edges: torch.LongTensor, attn_edge_type: torch.LongTensor) -> torch.Tensor:
        ...



class GraphormerMultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """
    def __init__(self, config: GraphormerConfig) -> None:
        ...

    def reset_parameters(self): # -> None:
        ...

    def forward(self, query: torch.LongTensor, key: Optional[torch.Tensor], value: Optional[torch.Tensor], attn_bias: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor] = ..., need_weights: bool = ..., attn_mask: Optional[torch.Tensor] = ..., before_softmax: bool = ..., need_head_weights: bool = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            key_padding_mask (Bytetorch.Tensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (Bytetorch.Tensor, optional): typically used to
                implement causal attention, where the mask prevents the attention from looking forward in time
                (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default: return the average attention weights over all
                heads.
        """
        ...

    def apply_sparse_mask(self, attn_weights: torch.Tensor, tgt_len: int, src_len: int, bsz: int) -> torch.Tensor:
        ...



class GraphormerGraphEncoderLayer(nn.Module):
    def __init__(self, config: GraphormerConfig) -> None:
        ...

    def build_fc(self, input_dim: int, output_dim: int, q_noise: float, qn_block_size: int) -> Union[nn.Module, nn.Linear, nn.Embedding, nn.Conv2d]:
        ...

    def forward(self, input_nodes: torch.Tensor, self_attn_bias: Optional[torch.Tensor] = ..., self_attn_mask: Optional[torch.Tensor] = ..., self_attn_padding_mask: Optional[torch.Tensor] = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        nn.LayerNorm is applied either before or after the self-attention/ffn modules similar to the original
        Transformer implementation.
        """
        ...



class GraphormerGraphEncoder(nn.Module):
    def __init__(self, config: GraphormerConfig) -> None:
        ...

    def forward(self, input_nodes: torch.LongTensor, input_edges: torch.LongTensor, attn_bias: torch.Tensor, in_degree: torch.LongTensor, out_degree: torch.LongTensor, spatial_pos: torch.LongTensor, attn_edge_type: torch.LongTensor, perturb=..., last_state_only: bool = ..., token_embeddings: Optional[torch.Tensor] = ..., attn_mask: Optional[torch.Tensor] = ...) -> Tuple[Union[torch.Tensor, List[torch.LongTensor]], torch.Tensor]:
        ...



class GraphormerDecoderHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        ...

    def forward(self, input_nodes: torch.Tensor, **unused) -> torch.Tensor:
        ...



class GraphormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GraphormerConfig
    base_model_prefix = ...
    main_input_name_nodes = ...
    main_input_name_edges = ...
    def normal_(self, data: torch.Tensor): # -> None:
        ...

    def init_graphormer_params(self, module: Union[nn.Linear, nn.Embedding, GraphormerMultiheadAttention]): # -> None:
        """
        Initialize the weights specific to the Graphormer Model.
        """
        ...



class GraphormerModel(GraphormerPreTrainedModel):
    """The Graphormer model is a graph-encoder model.

    It goes from a graph to its representation. If you want to use the model for a downstream classification task, use
    GraphormerForGraphClassification instead. For any other downstream task, feel free to add a new class, or combine
    this model with a downstream model of your choice, following the example in GraphormerForGraphClassification.
    """
    def __init__(self, config: GraphormerConfig) -> None:
        ...

    def reset_output_layer_parameters(self): # -> None:
        ...

    def forward(self, input_nodes: torch.LongTensor, input_edges: torch.LongTensor, attn_bias: torch.Tensor, in_degree: torch.LongTensor, out_degree: torch.LongTensor, spatial_pos: torch.LongTensor, attn_edge_type: torch.LongTensor, perturb: Optional[torch.FloatTensor] = ..., masked_tokens: None = ..., return_dict: Optional[bool] = ..., **unused) -> Union[Tuple[torch.LongTensor], BaseModelOutputWithNoAttention]:
        ...

    def max_nodes(self): # -> Callable[[], ...]:
        """Maximum output length supported by the encoder."""
        ...



class GraphormerForGraphClassification(GraphormerPreTrainedModel):
    """
    This model can be used for graph-level classification or regression tasks.

    It can be trained on
    - regression (by setting config.num_classes to 1); there should be one float-type label per graph
    - one task classification (by setting config.num_classes to the number of classes); there should be one integer
      label per graph
    - binary multi-task classification (by setting config.num_classes to the number of labels); there should be a list
      of integer labels for each graph.
    """
    def __init__(self, config: GraphormerConfig) -> None:
        ...

    def forward(self, input_nodes: torch.LongTensor, input_edges: torch.LongTensor, attn_bias: torch.Tensor, in_degree: torch.LongTensor, out_degree: torch.LongTensor, spatial_pos: torch.LongTensor, attn_edge_type: torch.LongTensor, labels: Optional[torch.LongTensor] = ..., return_dict: Optional[bool] = ..., **unused) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        ...
