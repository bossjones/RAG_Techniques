"""
This type stub file was generated by pyright.
"""

from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union
from .vision import VisionDataset

class Flowers102(VisionDataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
    flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of
    between 40 and 258 images.

    The images have large scale, pose and light variations. In addition, there are categories that
    have large variations within the category, and several very similar categories.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    _download_url_prefix = ...
    _file_dict = ...
    _splits_map = ...
    def __init__(self, root: Union[str, Path], split: str = ..., transform: Optional[Callable] = ..., target_transform: Optional[Callable] = ..., download: bool = ...) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        ...

    def extra_repr(self) -> str:
        ...

    def download(self): # -> None:
        ...
