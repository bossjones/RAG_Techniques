"""
This type stub file was generated by pyright.
"""

from typing import Dict
from pinecone.config import Config
from pinecone.core.client.api.manage_indexes_api import ManageIndexesApi as IndexOperationsApi

class SingletonMeta(type):
    _instances: Dict[str, str] = ...
    def __call__(cls, *args, **kwargs): # -> str:
        ...



class IndexHostStore(metaclass=SingletonMeta):
    def __init__(self) -> None:
        ...

    def delete_host(self, config: Config, index_name: str): # -> None:
        ...

    def key_exists(self, key: str) -> bool:
        ...

    def set_host(self, config: Config, index_name: str, host: str): # -> None:
        ...

    def get_host(self, api: IndexOperationsApi, config: Config, index_name: str) -> str:
        ...
