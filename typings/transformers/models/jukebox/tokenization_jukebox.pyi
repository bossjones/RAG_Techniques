"""
This type stub file was generated by pyright.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType

"""Tokenization classes for OpenAI Jukebox."""
logger = ...
VOCAB_FILES_NAMES = ...
class JukeboxTokenizer(PreTrainedTokenizer):
    """
    Constructs a Jukebox tokenizer. Jukebox can be conditioned on 3 different inputs :
        - Artists, unique ids are associated to each artist from the provided dictionary.
        - Genres, unique ids are associated to each genre from the provided dictionary.
        - Lyrics, character based tokenization. Must be initialized with the list of characters that are inside the
        vocabulary.

    This tokenizer does not require training. It should be able to process a different number of inputs:
    as the conditioning of the model can be done on the three different queries. If None is provided, defaults values will be used.:

    Depending on the number of genres on which the model should be conditioned (`n_genres`).
    ```python
    >>> from transformers import JukeboxTokenizer

    >>> tokenizer = JukeboxTokenizer.from_pretrained("openai/jukebox-1b-lyrics")
    >>> tokenizer("Alan Jackson", "Country Rock", "old town road")["input_ids"]
    [tensor([[   0,    0,    0, 6785,  546,   41,   38,   30,   76,   46,   41,   49,
               40,   76,   44,   41,   27,   30]]), tensor([[  0,   0,   0, 145,   0]]), tensor([[  0,   0,   0, 145,   0]])]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    If nothing is provided, the genres and the artist will either be selected randomly or set to None

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to:
    this superclass for more information regarding those methods.

    However the code does not allow that and only supports composing from various genres.

    Args:
        artists_file (`str`):
            Path to the vocabulary file which contains a mapping between artists and ids. The default file supports
            both "v2" and "v3"
        genres_file (`str`):
            Path to the vocabulary file which contain a mapping between genres and ids.
        lyrics_file (`str`):
            Path to the vocabulary file which contains the accepted characters for the lyrics tokenization.
        version (`List[str]`, `optional`, default to `["v3", "v2", "v2"]`) :
            List of the tokenizer versions. The `5b-lyrics`'s top level prior model was trained using `v3` instead of
            `v2`.
        n_genres (`int`, `optional`, defaults to 1):
            Maximum number of genres to use for composition.
        max_n_lyric_tokens (`int`, `optional`, defaults to 512):
            Maximum number of lyric tokens to keep.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    """
    vocab_files_names = ...
    model_input_names = ...
    def __init__(self, artists_file, genres_file, lyrics_file, version=..., max_n_lyric_tokens=..., n_genres=..., unk_token=..., **kwargs) -> None:
        ...

    @property
    def vocab_size(self): # -> int:
        ...

    def get_vocab(self): # -> dict[str, Any]:
        ...

    def tokenize(self, artist, genre, lyrics, **kwargs): # -> tuple[str, str, list[Any]]:
        """
        Converts three strings in a 3 sequence of tokens using the tokenizer
        """
        ...

    def prepare_for_tokenization(self, artists: str, genres: str, lyrics: str, is_split_into_words: bool = ...) -> Tuple[str, str, str, Dict[str, Any]]:
        """
        Performs any necessary transformations before tokenization.

        Args:
            artist (`str`):
                The artist name to prepare. This will mostly lower the string
            genres (`str`):
                The genre name to prepare. This will mostly lower the string.
            lyrics (`str`):
                The lyrics to prepare.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
        """
        ...

    def convert_lyric_tokens_to_string(self, lyrics: List[str]) -> str:
        ...

    def convert_to_tensors(self, inputs, tensor_type: Optional[Union[str, TensorType]] = ..., prepend_batch_axis: bool = ...): # -> NDArray[Any] | Tensor | list[Any]:
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                unset, no modification is done.
            prepend_batch_axis (`int`, *optional*, defaults to `False`):
                Whether or not to add the batch dimension during the conversion.
        """
        ...

    def __call__(self, artist, genres, lyrics=..., return_tensors=...) -> BatchEncoding:
        """Convert the raw string to a list of token ids

        Args:
            artist (`str`):
                Name of the artist.
            genres (`str`):
                List of genres that will be mixed to condition the audio
            lyrics (`str`, *optional*, defaults to `""`):
                Lyrics used to condition the generation
        """
        ...

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        """
        Saves the tokenizer's vocabulary dictionary to the provided save_directory.

        Args:
            save_directory (`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.

            filename_prefix (`Optional[str]`, *optional*):
                A prefix to add to the names of the files saved by the tokenizer.

        """
        ...
