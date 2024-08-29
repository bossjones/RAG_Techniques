"""
This type stub file was generated by pyright.
"""

import enum

MAX_MSG_SIZE = ...
MAX_ID_LENGTH = ...
REQUEST_ID: str = ...
CLIENT_VERSION_HEADER = ...
class NodeType(str, enum.Enum):
    STANDARD = ...
    COMPUTE = ...
    MEMORY = ...
    STANDARD2X = ...
    COMPUTE2X = ...
    MEMORY2X = ...
    STANDARD4X = ...
    COMPUTE4X = ...
    MEMORY4X = ...


CLIENT_VERSION = ...
CLIENT_ID = ...
REQUIRED_VECTOR_FIELDS = ...
OPTIONAL_VECTOR_FIELDS = ...
SOURCE_TAG = ...
