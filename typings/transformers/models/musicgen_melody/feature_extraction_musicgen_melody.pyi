"""
This type stub file was generated by pyright.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, is_torch_available, is_torchaudio_available

"""
Feature extractor class for Musicgen Melody
"""
if is_torch_available():
    ...
if is_torchaudio_available():
    ...
logger = ...
class MusicgenMelodyFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a MusicgenMelody feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts chroma features from audio processed by [Demucs](https://github.com/adefossez/demucs/tree/main) or
    directly from raw audio waveform.

    Args:
        feature_size (`int`, *optional*, defaults to 12):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 32000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        hop_length (`int`, *optional*, defaults to 4096):
            Length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, *optional*, defaults to 30):
            The maximum number of chunks of `sampling_rate` samples used to trim and pad longer or shorter audio
            sequences.
        n_fft (`int`, *optional*, defaults to 16384):
            Size of the Fourier transform.
        num_chroma (`int`, *optional*, defaults to 12):
            Number of chroma bins to use.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether to return the attention mask. Can be overwritten when calling the feature extractor.

            [What are attention masks?](../glossary#attention-mask)

            <Tip>

            For Whisper models, `attention_mask` should always be passed for batched inference, to avoid subtle
            bugs.

            </Tip>
        stem_indices (`List[int]`, *optional*, defaults to `[3, 2]`):
            Stem channels to extract if demucs outputs are passed.
    """
    model_input_names = ...
    def __init__(self, feature_size=..., sampling_rate=..., hop_length=..., chunk_length=..., n_fft=..., num_chroma=..., padding_value=..., return_attention_mask=..., stem_indices=..., **kwargs) -> None:
        ...

    def __call__(self, audio: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], truncation: bool = ..., pad_to_multiple_of: Optional[int] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., return_attention_mask: Optional[bool] = ..., padding: Optional[str] = ..., max_length: Optional[int] = ..., sampling_rate: Optional[int] = ..., **kwargs) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            audio (`torch.Tensor`, `np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[torch.Tensor]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a torch tensor, a numpy array, a list of float
                values, a list of numpy arrays, a list of torch tensors, or a list of list of float values.
                If `audio` is the output of Demucs, it has to be a torch tensor of shape `(batch_size, num_stems, channel_size, audio_length)`.
                Otherwise, it must be mono or stereo channel audio.
            truncation (`bool`, *optional*, default to `True`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*, defaults to None):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

                <Tip>
                For Musicgen Melody models, audio `attention_mask` is not necessary.
                </Tip>

            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
                Note that if `audio` is the output of Demucs, `sampling_rate` must be the sampling rate at which Demucs operates.
        """
        ...

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        ...
