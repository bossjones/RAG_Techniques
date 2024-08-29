"""
This type stub file was generated by pyright.
"""

from typing import Dict, List, Optional, TypedDict, Union
from pinecone.core.client.models import DescribeIndexStatsResponse, FetchResponse, QueryResponse
from pinecone.models.list_response import ListResponse as SimpleListResponse
from pinecone.core.grpc.protos.vector_service_pb2 import DeleteResponse, SparseValues as GRPCSparseValues, UpdateResponse, UpsertResponse, Vector as GRPCVector
from pinecone import Vector as NonGRPCVector
from .base import GRPCIndexBase
from .future import PineconeGrpcFuture

__all__ = ["GRPCIndex", "GRPCVector", "GRPCQueryVector", "GRPCSparseValues"]
_logger = ...
class SparseVectorTypedDict(TypedDict):
    indices: List[int]
    values: List[float]
    ...


class GRPCIndex(GRPCIndexBase):
    """A client for interacting with a Pinecone index via GRPC API."""
    @property
    def stub_class(self): # -> type[VectorServiceStub]:
        ...

    def upsert(self, vectors: Union[List[GRPCVector], List[NonGRPCVector], List[tuple], List[dict]], async_req: bool = ..., namespace: Optional[str] = ..., batch_size: Optional[int] = ..., show_progress: bool = ..., **kwargs) -> Union[UpsertResponse, PineconeGrpcFuture]:
        """
        The upsert operation writes vectors into a namespace.
        If a new value is upserted for an existing vector id, it will overwrite the previous value.

        Examples:
            >>> index.upsert([('id1', [1.0, 2.0, 3.0], {'key': 'value'}),
                              ('id2', [1.0, 2.0, 3.0])
                              ],
                              namespace='ns1', async_req=True)
            >>> index.upsert([{'id': 'id1', 'values': [1.0, 2.0, 3.0], 'metadata': {'key': 'value'}},
                              {'id': 'id2',
                                        'values': [1.0, 2.0, 3.0],
                                        'sparse_values': {'indices': [1, 8], 'values': [0.2, 0.4]},
                              ])
            >>> index.upsert([GRPCVector(id='id1', values=[1.0, 2.0, 3.0], metadata={'key': 'value'}),
                              GRPCVector(id='id2', values=[1.0, 2.0, 3.0]),
                              GRPCVector(id='id3',
                                         values=[1.0, 2.0, 3.0],
                                         sparse_values=GRPCSparseValues(indices=[1, 2], values=[0.2, 0.4]))])

        Args:
            vectors (Union[List[Vector], List[Tuple]]): A list of vectors to upsert.

                     A vector can be represented by a 1) GRPCVector object, a 2) tuple or 3) a dictionary
                     1) if a tuple is used, it must be of the form (id, values, metadata) or (id, values).
                        where id is a string, vector is a list of floats, and metadata is a dict.
                        Examples: ('id1', [1.0, 2.0, 3.0], {'key': 'value'}), ('id2', [1.0, 2.0, 3.0])

                    2) if a GRPCVector object is used, a GRPCVector object must be of the form
                        GRPCVector(id, values, metadata), where metadata is an optional argument of type
                        Dict[str, Union[str, float, int, bool, List[int], List[float], List[str]]]
                       Examples: GRPCVector(id='id1', values=[1.0, 2.0, 3.0], metadata={'key': 'value'}),
                                 GRPCVector(id='id2', values=[1.0, 2.0, 3.0]),
                                 GRPCVector(id='id3',
                                            values=[1.0, 2.0, 3.0],
                                            sparse_values=GRPCSparseValues(indices=[1, 2], values=[0.2, 0.4]))

                    3) if a dictionary is used, it must be in the form
                       {'id': str, 'values': List[float], 'sparse_values': {'indices': List[int], 'values': List[float]},
                        'metadata': dict}

                    Note: the dimension of each vector must match the dimension of the index.
            async_req (bool): If True, the upsert operation will be performed asynchronously.
                              Cannot be used with batch_size.
                              Defaults to False. See: https://docs.pinecone.io/docs/performance-tuning [optional]
            namespace (str): The namespace to write to. If not specified, the default namespace is used. [optional]
            batch_size (int): The number of vectors to upsert in each batch.
                                Cannot be used with async_req=True.
                               If not specified, all vectors will be upserted in a single batch. [optional]
            show_progress (bool): Whether to show a progress bar using tqdm.
                                  Applied only if batch_size is provided. Default is True.

        Returns: UpsertResponse, contains the number of vectors upserted
        """
        ...

    def upsert_from_dataframe(self, df, namespace: str = ..., batch_size: int = ..., use_async_requests: bool = ..., show_progress: bool = ...) -> UpsertResponse:
        """Upserts a dataframe into the index.

        Args:
            df: A pandas dataframe with the following columns: id, vector, and metadata.
            namespace: The namespace to upsert into.
            batch_size: The number of rows to upsert in a single batch.
            use_async_requests: Whether to upsert multiple requests at the same time using asynchronous request mechanism.
                                Set to `False`
            show_progress: Whether to show a progress bar.
        """
        ...

    def delete(self, ids: Optional[List[str]] = ..., delete_all: Optional[bool] = ..., namespace: Optional[str] = ..., filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = ..., async_req: bool = ..., **kwargs) -> Union[DeleteResponse, PineconeGrpcFuture]:
        """
        The Delete operation deletes vectors from the index, from a single namespace.
        No error raised if the vector id does not exist.
        Note: for any delete call, if namespace is not specified, the default namespace is used.

        Delete can occur in the following mutual exclusive ways:
        1. Delete by ids from a single namespace
        2. Delete all vectors from a single namespace by setting delete_all to True
        3. Delete all vectors from a single namespace by specifying a metadata filter
           (note that for this option delete all must be set to False)

        Examples:
            >>> index.delete(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.delete(delete_all=True, namespace='my_namespace')
            >>> index.delete(filter={'key': 'value'}, namespace='my_namespace', async_req=True)

        Args:
            ids (List[str]): Vector ids to delete [optional]
            delete_all (bool): This indicates that all vectors in the index namespace should be deleted.. [optional]
                               Default is False.
            namespace (str): The namespace to delete vectors from [optional]
                             If not specified, the default namespace is used.
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
                    If specified, the metadata filter here will be used to select the vectors to delete.
                    This is mutually exclusive with specifying ids to delete in the ids param or using delete_all=True.
                     See https://www.pinecone.io/docs/metadata-filtering/.. [optional]
            async_req (bool): If True, the delete operation will be performed asynchronously.
                              Defaults to False. [optional]

        Returns: DeleteResponse (contains no data) or a PineconeGrpcFuture object if async_req is True.
        """
        ...

    def fetch(self, ids: Optional[List[str]], namespace: Optional[str] = ..., **kwargs) -> FetchResponse:
        """
        The fetch operation looks up and returns vectors, by ID, from a single namespace.
        The returned vectors include the vector data and/or metadata.

        Examples:
            >>> index.fetch(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.fetch(ids=['id1', 'id2'])

        Args:
            ids (List[str]): The vector IDs to fetch.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]

        Returns: FetchResponse object which contains the list of Vector objects, and namespace name.
        """
        ...

    def query(self, vector: Optional[List[float]] = ..., id: Optional[str] = ..., namespace: Optional[str] = ..., top_k: Optional[int] = ..., filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = ..., include_values: Optional[bool] = ..., include_metadata: Optional[bool] = ..., sparse_vector: Optional[Union[GRPCSparseValues, SparseVectorTypedDict]] = ..., **kwargs) -> QueryResponse:
        """
        The Query operation searches a namespace, using a query vector.
        It retrieves the ids of the most similar items in a namespace, along with their similarity scores.

        Examples:
            >>> index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace')
            >>> index.query(id='id1', top_k=10, namespace='my_namespace')
            >>> index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace', filter={'key': 'value'})
            >>> index.query(id='id1', top_k=10, namespace='my_namespace', include_metadata=True, include_values=True)
            >>> index.query(vector=[1, 2, 3], sparse_vector={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>             top_k=10, namespace='my_namespace')
            >>> index.query(vector=[1, 2, 3], sparse_vector=GRPCSparseValues([1, 2], [0.2, 0.4]),
            >>>             top_k=10, namespace='my_namespace')

        Args:
            vector (List[float]): The query vector. This should be the same length as the dimension of the index
                                  being queried. Each `query()` request can contain only one of the parameters
                                  `id` or `vector`.. [optional]
            id (str): The unique ID of the vector to be used as a query vector.
                      Each `query()` request can contain only one of the parameters
                      `vector` or  `id`.. [optional]
            top_k (int): The number of results to return for each query. Must be an integer greater than 1.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
                    The filter to apply. You can use vector metadata to limit your search.
                    See https://www.pinecone.io/docs/metadata-filtering/.. [optional]
            include_values (bool): Indicates whether vector values are included in the response.
                                   If omitted the server will use the default value of False [optional]
            include_metadata (bool): Indicates whether metadata is included in the response as well as the ids.
                                     If omitted the server will use the default value of False  [optional]
            sparse_vector: (Union[SparseValues, Dict[str, Union[List[float], List[int]]]]): sparse values of the query vector.
                            Expected to be either a GRPCSparseValues object or a dict of the form:
                             {'indices': List[int], 'values': List[float]}, where the lists each have the same length.

        Returns: QueryResponse object which contains the list of the closest vectors as ScoredVector objects,
                 and namespace name.
        """
        ...

    def update(self, id: str, async_req: bool = ..., values: Optional[List[float]] = ..., set_metadata: Optional[Dict[str, Union[str, float, int, bool, List[int], List[float], List[str]]]] = ..., namespace: Optional[str] = ..., sparse_values: Optional[Union[GRPCSparseValues, SparseVectorTypedDict]] = ..., **kwargs) -> Union[UpdateResponse, PineconeGrpcFuture]:
        """
        The Update operation updates vector in a namespace.
        If a value is included, it will overwrite the previous value.
        If a set_metadata is included,
        the values of the fields specified in it will be added or overwrite the previous value.

        Examples:
            >>> index.update(id='id1', values=[1, 2, 3], namespace='my_namespace')
            >>> index.update(id='id1', set_metadata={'key': 'value'}, namespace='my_namespace', async_req=True)
            >>> index.update(id='id1', values=[1, 2, 3], sparse_values={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>              namespace='my_namespace')
            >>> index.update(id='id1', values=[1, 2, 3], sparse_values=GRPCSparseValues(indices=[1, 2], values=[0.2, 0.4]),
            >>>              namespace='my_namespace')

        Args:
            id (str): Vector's unique id.
            async_req (bool): If True, the update operation will be performed asynchronously.
                              Defaults to False. [optional]
            values (List[float]): vector values to set. [optional]
            set_metadata (Dict[str, Union[str, float, int, bool, List[int], List[float], List[str]]]]):
                metadata to set for vector. [optional]
            namespace (str): Namespace name where to update the vector.. [optional]
            sparse_values: (Dict[str, Union[List[float], List[int]]]): sparse values to update for the vector.
                           Expected to be either a GRPCSparseValues object or a dict of the form:
                           {'indices': List[int], 'values': List[float]} where the lists each have the same length.


        Returns: UpdateResponse (contains no data) or a PineconeGrpcFuture object if async_req is True.
        """
        ...

    def list_paginated(self, prefix: Optional[str] = ..., limit: Optional[int] = ..., pagination_token: Optional[str] = ..., namespace: Optional[str] = ..., **kwargs) -> SimpleListResponse:
        """
        The list_paginated operation finds vectors based on an id prefix within a single namespace.
        It returns matching ids in a paginated form, with a pagination token to fetch the next page of results.
        This id list can then be passed to fetch or delete operations, depending on your use case.

        Consider using the `list` method to avoid having to handle pagination tokens manually.

        Examples:
            >>> results = index.list_paginated(prefix='99', limit=5, namespace='my_namespace')
            >>> [v.id for v in results.vectors]
            ['99', '990', '991', '992', '993']
            >>> results.pagination.next
            eyJza2lwX3Bhc3QiOiI5OTMiLCJwcmVmaXgiOiI5OSJ9
            >>> next_results = index.list_paginated(prefix='99', limit=5, namespace='my_namespace', pagination_token=results.pagination.next)

        Args:
            prefix (Optional[str]): The id prefix to match. If unspecified, an empty string prefix will
                                    be used with the effect of listing all ids in a namespace [optional]
            limit (Optional[int]): The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]
            namespace (Optional[str]): The namespace to fetch vectors from. If not specified, the default namespace is used. [optional]

        Returns: SimpleListResponse object which contains the list of ids, the namespace name, pagination information, and usage showing the number of read_units consumed.
        """
        ...

    def list(self, **kwargs): # -> Generator[list[Any], Any, None]:
        """
        The list operation accepts all of the same arguments as list_paginated, and returns a generator that yields
        a list of the matching vector ids in each page of results. It automatically handles pagination tokens on your
        behalf.

        Examples:
            >>> for ids in index.list(prefix='99', limit=5, namespace='my_namespace'):
            >>>     print(ids)
            ['99', '990', '991', '992', '993']
            ['994', '995', '996', '997', '998']
            ['999']

        Args:
            prefix (Optional[str]): The id prefix to match. If unspecified, an empty string prefix will
                                    be used with the effect of listing all ids in a namespace [optional]
            limit (Optional[int]): The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]
            namespace (Optional[str]): The namespace to fetch vectors from. If not specified, the default namespace is used. [optional]
        """
        ...

    def describe_index_stats(self, filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = ..., **kwargs) -> DescribeIndexStatsResponse:
        """
        The DescribeIndexStats operation returns statistics about the index's contents.
        For example: The vector count per namespace and the number of dimensions.

        Examples:
            >>> index.describe_index_stats()
            >>> index.describe_index_stats(filter={'key': 'value'})

        Args:
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
            If this parameter is present, the operation only returns statistics for vectors that satisfy the filter.
            See https://www.pinecone.io/docs/metadata-filtering/.. [optional]

        Returns: DescribeIndexStatsResponse object which contains stats about the index.
        """
        ...
