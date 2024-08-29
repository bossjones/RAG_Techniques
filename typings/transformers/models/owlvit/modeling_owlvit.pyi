"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Union
from torch import Tensor, nn
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, is_vision_available, replace_return_docstrings
from .configuration_owlvit import OwlViTConfig, OwlViTTextConfig, OwlViTVisionConfig

""" PyTorch OWL-ViT model."""
if is_vision_available():
    ...
logger = ...
_CHECKPOINT_FOR_DOC = ...
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    ...

def owlvit_loss(similarity: torch.Tensor) -> torch.Tensor:
    ...

@dataclass
class OwlViTOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds (`torch.FloatTensor` of shape `(batch_size * num_max_text_queries, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`OwlViTTextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`OwlViTVisionModel`].
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`OwlViTTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`OwlViTVisionModel`].
    """
    loss: Optional[torch.FloatTensor] = ...
    logits_per_image: torch.FloatTensor = ...
    logits_per_text: torch.FloatTensor = ...
    text_embeds: torch.FloatTensor = ...
    image_embeds: torch.FloatTensor = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> Tuple[Any]:
        ...



def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    ...

def box_iou(boxes1, boxes2): # -> tuple[Any, Any]:
    ...

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    ...

@dataclass
class OwlViTObjectDetectionOutput(ModelOutput):
    """
    Output type of [`OwlViTForObjectDetection`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~OwlViTImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, num_max_text_queries, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`OwlViTTextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`OwlViTVisionModel`]. OWL-ViT represents images as a set of image patches and computes
            image embeddings for each patch.
        class_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            Class embeddings of all image patches. OWL-ViT represents images as a set of image patches where the total
            number of patches is (image_size / patch_size)**2.
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`OwlViTTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`OwlViTVisionModel`].
    """
    loss: Optional[torch.FloatTensor] = ...
    loss_dict: Optional[Dict] = ...
    logits: torch.FloatTensor = ...
    pred_boxes: torch.FloatTensor = ...
    text_embeds: torch.FloatTensor = ...
    image_embeds: torch.FloatTensor = ...
    class_embeds: torch.FloatTensor = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> Tuple[Any]:
        ...



@dataclass
class OwlViTImageGuidedObjectDetectionOutput(ModelOutput):
    """
    Output type of [`OwlViTForObjectDetection.image_guided_detection`].

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`):
            Classification logits (including no-object) for all queries.
        target_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual target image in the batch
            (disregarding possible padding). You can use [`~OwlViTImageProcessor.post_process_object_detection`] to
            retrieve the unnormalized bounding boxes.
        query_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual query image in the batch
            (disregarding possible padding). You can use [`~OwlViTImageProcessor.post_process_object_detection`] to
            retrieve the unnormalized bounding boxes.
        image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`OwlViTVisionModel`]. OWL-ViT represents images as a set of image patches and computes
            image embeddings for each patch.
        query_image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`OwlViTVisionModel`]. OWL-ViT represents images as a set of image patches and computes
            image embeddings for each patch.
        class_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            Class embeddings of all image patches. OWL-ViT represents images as a set of image patches where the total
            number of patches is (image_size / patch_size)**2.
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`OwlViTTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`OwlViTVisionModel`].
    """
    logits: torch.FloatTensor = ...
    image_embeds: torch.FloatTensor = ...
    query_image_embeds: torch.FloatTensor = ...
    target_pred_boxes: torch.FloatTensor = ...
    query_pred_boxes: torch.FloatTensor = ...
    class_embeds: torch.FloatTensor = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> Tuple[Any]:
        ...



class OwlViTVisionEmbeddings(nn.Module):
    def __init__(self, config: OwlViTVisionConfig) -> None:
        ...

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        ...



class OwlViTTextEmbeddings(nn.Module):
    def __init__(self, config: OwlViTTextConfig) -> None:
        ...

    def forward(self, input_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ...) -> torch.Tensor:
        ...



class OwlViTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., causal_attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        ...



class OwlViTMLP(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



class OwlViTEncoderLayer(nn.Module):
    def __init__(self, config: OwlViTConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal_attention_mask: torch.Tensor, output_attentions: Optional[bool] = ...) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        ...



class OwlViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = OwlViTConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...


OWLVIT_START_DOCSTRING = ...
OWLVIT_TEXT_INPUTS_DOCSTRING = ...
OWLVIT_VISION_INPUTS_DOCSTRING = ...
OWLVIT_INPUTS_DOCSTRING = ...
OWLVIT_OBJECT_DETECTION_INPUTS_DOCSTRING = ...
OWLVIT_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING = ...
class OwlViTEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`OwlViTEncoderLayer`].

    Args:
        config: OwlViTConfig
    """
    def __init__(self, config: OwlViTConfig) -> None:
        ...

    def forward(self, inputs_embeds, attention_mask: Optional[torch.Tensor] = ..., causal_attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`).
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        ...



class OwlViTTextTransformer(nn.Module):
    def __init__(self, config: OwlViTTextConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTTextConfig)
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        ...



class OwlViTTextModel(OwlViTPreTrainedModel):
    config_class = OwlViTTextConfig
    def __init__(self, config: OwlViTTextConfig) -> None:
        ...

    def get_input_embeddings(self) -> nn.Module:
        ...

    def set_input_embeddings(self, value): # -> None:
        ...

    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTTextConfig)
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:
        ```python
        >>> from transformers import AutoProcessor, OwlViTTextModel

        >>> model = OwlViTTextModel.from_pretrained("google/owlvit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> inputs = processor(
        ...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt"
        ... )
        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        ...



class OwlViTVisionTransformer(nn.Module):
    def __init__(self, config: OwlViTVisionConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(OWLVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTVisionConfig)
    def forward(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        ...



class OwlViTVisionModel(OwlViTPreTrainedModel):
    config_class = OwlViTVisionConfig
    main_input_name = ...
    def __init__(self, config: OwlViTVisionConfig) -> None:
        ...

    def get_input_embeddings(self) -> nn.Module:
        ...

    @add_start_docstrings_to_model_forward(OWLVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTVisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, OwlViTVisionModel

        >>> model = OwlViTVisionModel.from_pretrained("google/owlvit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        ...



@add_start_docstrings(OWLVIT_START_DOCSTRING)
class OwlViTModel(OwlViTPreTrainedModel):
    config_class = OwlViTConfig
    def __init__(self, config: OwlViTConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`OwlViTTextModel`].

        Examples:
        ```python
        >>> from transformers import AutoProcessor, OwlViTModel

        >>> model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> inputs = processor(
        ...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt"
        ... )
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        ...

    @add_start_docstrings_to_model_forward(OWLVIT_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`OwlViTVisionModel`].

        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, OwlViTModel

        >>> model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> image_features = model.get_image_features(**inputs)
        ```"""
        ...

    @add_start_docstrings_to_model_forward(OWLVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OwlViTOutput, config_class=OwlViTConfig)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., pixel_values: Optional[torch.FloatTensor] = ..., attention_mask: Optional[torch.Tensor] = ..., return_loss: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_base_image_embeds: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, OwlViTOutput]:
        r"""
        Returns:

        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, OwlViTModel

        >>> model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(text=[["a photo of a cat", "a photo of a dog"]], images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        ...



class OwlViTBoxPredictionHead(nn.Module):
    def __init__(self, config: OwlViTConfig, out_dim: int = ...) -> None:
        ...

    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor:
        ...



class OwlViTClassPredictionHead(nn.Module):
    def __init__(self, config: OwlViTConfig) -> None:
        ...

    def forward(self, image_embeds: torch.FloatTensor, query_embeds: Optional[torch.FloatTensor], query_mask: Optional[torch.Tensor]) -> Tuple[torch.FloatTensor]:
        ...



class OwlViTForObjectDetection(OwlViTPreTrainedModel):
    config_class = OwlViTConfig
    def __init__(self, config: OwlViTConfig) -> None:
        ...

    @staticmethod
    def normalize_grid_corner_coordinates(num_patches: int) -> torch.Tensor:
        ...

    @lru_cache(maxsize=2)
    def compute_box_bias(self, num_patches: int, feature_map: Optional[torch.FloatTensor] = ...) -> torch.Tensor:
        ...

    def box_predictor(self, image_feats: torch.FloatTensor, feature_map: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            image_feats:
                Features extracted from the image, returned by the `image_text_embedder` method.
            feature_map:
                A spatial re-arrangement of image_features, also returned by the `image_text_embedder` method.
        Returns:
            pred_boxes:
                List of predicted boxes (cxcywh normalized to 0, 1) nested within a dictionary.
        """
        ...

    def class_predictor(self, image_feats: torch.FloatTensor, query_embeds: Optional[torch.FloatTensor] = ..., query_mask: Optional[torch.Tensor] = ...) -> Tuple[torch.FloatTensor]:
        """
        Args:
            image_feats:
                Features extracted from the `image_text_embedder`.
            query_embeds:
                Text query embeddings.
            query_mask:
                Must be provided with query_embeddings. A mask indicating which query embeddings are valid.
        """
        ...

    def image_text_embedder(self, input_ids: torch.Tensor, pixel_values: torch.FloatTensor, attention_mask: torch.Tensor, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ...) -> Tuple[torch.FloatTensor]:
        ...

    def image_embedder(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ...) -> Tuple[torch.FloatTensor]:
        ...

    def embed_image_query(self, query_image_features: torch.FloatTensor, query_feature_map: torch.FloatTensor) -> torch.FloatTensor:
        ...

    @add_start_docstrings_to_model_forward(OWLVIT_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OwlViTImageGuidedObjectDetectionOutput, config_class=OwlViTConfig)
    def image_guided_detection(self, pixel_values: torch.FloatTensor, query_pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> OwlViTImageGuidedObjectDetectionOutput:
        r"""
        Returns:

        Examples:
        ```python
        >>> import requests
        >>> from PIL import Image
        >>> import torch
        >>> from transformers import AutoProcessor, OwlViTForObjectDetection

        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch16")
        >>> model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> query_url = "http://images.cocodataset.org/val2017/000000001675.jpg"
        >>> query_image = Image.open(requests.get(query_url, stream=True).raw)
        >>> inputs = processor(images=image, query_images=query_image, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model.image_guided_detection(**inputs)
        >>> # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        >>> target_sizes = torch.Tensor([image.size[::-1]])
        >>> # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> results = processor.post_process_image_guided_detection(
        ...     outputs=outputs, threshold=0.6, nms_threshold=0.3, target_sizes=target_sizes
        ... )
        >>> i = 0  # Retrieve predictions for the first image
        >>> boxes, scores = results[i]["boxes"], results[i]["scores"]
        >>> for box, score in zip(boxes, scores):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(f"Detected similar object with confidence {round(score.item(), 3)} at location {box}")
        Detected similar object with confidence 0.856 at location [10.94, 50.4, 315.8, 471.39]
        Detected similar object with confidence 1.0 at location [334.84, 25.33, 636.16, 374.71]
        ```"""
        ...

    @add_start_docstrings_to_model_forward(OWLVIT_OBJECT_DETECTION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OwlViTObjectDetectionOutput, config_class=OwlViTConfig)
    def forward(self, input_ids: torch.Tensor, pixel_values: torch.FloatTensor, attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> OwlViTObjectDetectionOutput:
        r"""
        Returns:

        Examples:
        ```python
        >>> import requests
        >>> from PIL import Image
        >>> import torch
        >>> from transformers import AutoProcessor, OwlViTForObjectDetection

        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = [["a photo of a cat", "a photo of a dog"]]
        >>> inputs = processor(text=texts, images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        >>> target_sizes = torch.Tensor([image.size[::-1]])
        >>> # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
        >>> results = processor.post_process_object_detection(
        ...     outputs=outputs, threshold=0.1, target_sizes=target_sizes
        ... )

        >>> i = 0  # Retrieve predictions for the first image for the corresponding text queries
        >>> text = texts[i]
        >>> boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        >>> for box, score, label in zip(boxes, scores, labels):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
        Detected a photo of a cat with confidence 0.707 at location [324.97, 20.44, 640.58, 373.29]
        Detected a photo of a cat with confidence 0.717 at location [1.46, 55.26, 315.55, 472.17]
        ```"""
        ...
