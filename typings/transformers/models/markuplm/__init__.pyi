"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available
from .configuration_markuplm import MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP, MarkupLMConfig
from .feature_extraction_markuplm import MarkupLMFeatureExtractor
from .processing_markuplm import MarkupLMProcessor
from .tokenization_markuplm import MarkupLMTokenizer

_import_structure = ...
if not is_tokenizers_available():
    ...
if not is_torch_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
