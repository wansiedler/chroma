from dataclasses import dataclass
from typing import (
    Optional,
    Self,
)
from chromadb.api.types import EmbeddingFunction, Documents
import chromadb.utils.embedding_functions as ef


@dataclass
class RuntimeConfig:
    num_threads: int = 4
    resize_factor: float = 1.3
    batch_size: int = 100
    sync_threshold: int = 1000


class RuntimeConfigBuilder:
    def __init__(self) -> None:
        self._num_threads: Optional[int] = None
        self._resize_factor: Optional[float] = None
        self._batch_size: Optional[int] = None
        self._sync_threshold: Optional[int] = None

    def with_num_threads(self, num_threads: int) -> Self:
        self._num_threads = num_threads
        return self

    def with_resize_factor(self, resize_factor: float) -> Self:
        self._resize_factor = resize_factor
        return self

    def with_batch_size(self, batch_size: int) -> Self:
        self._batch_size = batch_size
        return self

    def with_sync_threshold(self, sync_threshold: int) -> Self:
        self._sync_threshold = sync_threshold
        return self

    def build(self) -> RuntimeConfig:
        return RuntimeConfig(
            num_threads=self._num_threads
            if self._num_threads is not None
            else RuntimeConfig.num_threads,
            resize_factor=self._resize_factor
            if self._resize_factor is not None
            else RuntimeConfig.resize_factor,
            batch_size=self._batch_size
            if self._batch_size is not None
            else RuntimeConfig.batch_size,
            sync_threshold=self._sync_threshold
            if self._sync_threshold is not None
            else RuntimeConfig.sync_threshold,
        )


@dataclass
class HNSWConfig:
    ef_search: int = 100


class HNSWConfigBuilder:
    def __init__(self) -> None:
        self._ef_search: Optional[int] = None

    def with_ef_search(self, ef_search: int) -> Self:
        self._ef_search = ef_search
        return self

    def build(self) -> HNSWConfig:
        return HNSWConfig(
            ef_search=self._ef_search
            if self._ef_search is not None
            else HNSWConfig.ef_search,
        )


@dataclass
class HNSWCreateConfig(HNSWConfig):
    distance_metric: str = "l2"
    ef_construction: int = 80
    max_neighbors: int = 50


class HNSWCreateConfigBuilder:
    def __init__(self) -> None:
        self._ef_search: Optional[int] = None
        self._distance_metric: Optional[str] = None
        self._ef_construction: Optional[int] = None
        self._max_neighbors: Optional[int] = None

    def with_ef_search(self, ef_search: int) -> Self:
        self._ef_search = ef_search
        return self

    def with_distance_metric(self, distance_metric: str) -> Self:
        self._distance_metric = distance_metric
        return self

    def with_ef_construction(self, ef_construction: int) -> Self:
        self._ef_construction = ef_construction
        return self

    def with_max_neighbors(self, max_neighbors: int) -> Self:
        self._max_neighbors = max_neighbors
        return self

    def build(self) -> HNSWCreateConfig:
        return HNSWCreateConfig(
            ef_search=self._ef_search
            if self._ef_search is not None
            else HNSWCreateConfig.ef_search,
            distance_metric=self._distance_metric
            if self._distance_metric is not None
            else HNSWCreateConfig.distance_metric,
            ef_construction=self._ef_construction
            if self._ef_construction is not None
            else HNSWCreateConfig.ef_construction,
            max_neighbors=self._max_neighbors
            if self._max_neighbors is not None
            else HNSWCreateConfig.max_neighbors,
        )


@dataclass
class CreateCollectionConfig:
    hnsw: HNSWCreateConfig = HNSWCreateConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    embedding_function: Optional[
        EmbeddingFunction[Documents]
    ] = ef.DefaultEmbeddingFunction()


class CreateCollectionConfigBuilder:
    def __init__(self) -> None:
        self._hnsw_builder = HNSWCreateConfigBuilder()
        self._runtime_builder = RuntimeConfigBuilder()
        self._embedding_function: Optional[
            EmbeddingFunction[Documents]
        ] = ef.DefaultEmbeddingFunction()

    def with_hnsw(
        self,
        ef_search: Optional[int] = None,
        distance_metric: Optional[str] = None,
        ef_construction: Optional[int] = None,
        max_neighbors: Optional[int] = None,
    ) -> Self:
        if ef_search is not None:
            self._hnsw_builder.with_ef_search(ef_search)
        if distance_metric is not None:
            self._hnsw_builder.with_distance_metric(distance_metric)
        if ef_construction is not None:
            self._hnsw_builder.with_ef_construction(ef_construction)
        if max_neighbors is not None:
            self._hnsw_builder.with_max_neighbors(max_neighbors)
        return self

    def with_runtime(
        self,
        num_threads: Optional[int] = None,
        resize_factor: Optional[float] = None,
        batch_size: Optional[int] = None,
        sync_threshold: Optional[int] = None,
    ) -> Self:
        if num_threads is not None:
            self._runtime_builder.with_num_threads(num_threads)
        if resize_factor is not None:
            self._runtime_builder.with_resize_factor(resize_factor)
        if batch_size is not None:
            self._runtime_builder.with_batch_size(batch_size)
        if sync_threshold is not None:
            self._runtime_builder.with_sync_threshold(sync_threshold)
        return self

    def with_embedding_function(
        self, embedding_function: Optional[EmbeddingFunction[Documents]]
    ) -> Self:
        self._embedding_function = embedding_function
        return self

    def build(self) -> CreateCollectionConfig:
        return CreateCollectionConfig(
            hnsw=self._hnsw_builder.build(),
            runtime=self._runtime_builder.build(),
            embedding_function=self._embedding_function,
        )


@dataclass
class UpdateCollectionConfig:
    hnsw: HNSWConfig = HNSWConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    embedding_function: Optional[
        EmbeddingFunction[Documents]
    ] = ef.DefaultEmbeddingFunction()


class UpdateCollectionConfigBuilder:
    def __init__(self) -> None:
        self._hnsw_builder = HNSWConfigBuilder()
        self._runtime_builder = RuntimeConfigBuilder()
        self._embedding_function: Optional[
            EmbeddingFunction[Documents]
        ] = ef.DefaultEmbeddingFunction()

    def with_hnsw(
        self,
        ef_search: Optional[int] = None,
    ) -> Self:
        if ef_search is not None:
            self._hnsw_builder.with_ef_search(ef_search)
        return self

    def with_runtime(
        self,
        num_threads: Optional[int] = None,
        resize_factor: Optional[float] = None,
        batch_size: Optional[int] = None,
        sync_threshold: Optional[int] = None,
    ) -> Self:
        if num_threads is not None:
            self._runtime_builder.with_num_threads(num_threads)
        if resize_factor is not None:
            self._runtime_builder.with_resize_factor(resize_factor)
        if batch_size is not None:
            self._runtime_builder.with_batch_size(batch_size)
        if sync_threshold is not None:
            self._runtime_builder.with_sync_threshold(sync_threshold)
        return self

    def with_embedding_function(
        self, embedding_function: Optional[EmbeddingFunction[Documents]]
    ) -> Self:
        self._embedding_function = embedding_function
        return self

    def build(self) -> UpdateCollectionConfig:
        return UpdateCollectionConfig(
            hnsw=self._hnsw_builder.build(),
            runtime=self._runtime_builder.build(),
            embedding_function=self._embedding_function,
        )


@dataclass
class QueryCollectionConfig:
    hnsw: HNSWConfig = HNSWConfig()
    embedding_function: Optional[
        EmbeddingFunction[Documents]
    ] = ef.DefaultEmbeddingFunction()


class QueryCollectionConfigBuilder:
    def __init__(self) -> None:
        self._hnsw_builder = HNSWConfigBuilder()
        self._embedding_function: Optional[
            EmbeddingFunction[Documents]
        ] = ef.DefaultEmbeddingFunction()

    def with_hnsw(
        self,
        ef_search: Optional[int] = None,
    ) -> Self:
        if ef_search is not None:
            self._hnsw_builder.with_ef_search(ef_search)
        return self

    def with_embedding_function(
        self, embedding_function: Optional[EmbeddingFunction[Documents]]
    ) -> Self:
        self._embedding_function = embedding_function
        return self

    def build(self) -> QueryCollectionConfig:
        return QueryCollectionConfig(
            hnsw=self._hnsw_builder.build(),
            embedding_function=self._embedding_function,
        )


# Factory functions for better ergonomics
def NewCreateCollectionConfig() -> CreateCollectionConfigBuilder:
    return CreateCollectionConfigBuilder()


def NewUpdateCollectionConfig() -> UpdateCollectionConfigBuilder:
    return UpdateCollectionConfigBuilder()


def NewQueryCollectionConfig() -> QueryCollectionConfigBuilder:
    return QueryCollectionConfigBuilder()


# Example usage
create_config = (
    NewCreateCollectionConfig()
    .with_hnsw(ef_search=100, ef_construction=100, max_neighbors=100)
    .with_runtime(num_threads=4)
    .with_embedding_function(embedding_function=ef.DefaultEmbeddingFunction())
    .build()
)

update_config = (
    NewUpdateCollectionConfig()
    .with_hnsw(ef_search=100)
    .with_runtime(num_threads=4)
    .with_embedding_function(embedding_function=ef.DefaultEmbeddingFunction())
    .build()
)

query_config = (
    NewQueryCollectionConfig()
    .with_hnsw(ef_search=100)
    .with_embedding_function(embedding_function=ef.DefaultEmbeddingFunction())
    .build()
)
