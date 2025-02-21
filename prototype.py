# import chromadb

from enum import Enum
from typing import Protocol, List, Dict, Any, Self, TypedDict, Optional
from abc import abstractmethod
import numpy as np
from chromadb.api.types import D, Embeddings, Embeddable


class DistanceMetric(Enum):
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"


supported_embedding_functions = {}


class EmbeddingFunction(Protocol[D]):
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def generate_embeddings(self, input: D) -> Embeddings:
        pass

    @abstractmethod
    def default_metric(self) -> DistanceMetric:
        pass

    @abstractmethod
    def build_from_config(self, config: Dict[str, Any]) -> Self:
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def modifiable_variables(self) -> List[str]:
        pass

    @abstractmethod
    def register(self) -> None:
        pass


class CohereEmbeddingFunction(EmbeddingFunction[Embeddable]):
    def __init__(self, model_name: Optional[str], api_key_env_var: Optional[str]):
        self._model_name = model_name
        self._api_key_env_var = api_key_env_var

    def name(self) -> str:
        return "cohere"

    def generate_embeddings(self, input: Embeddable) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        return [np.random.rand(1024).astype(np.float32) for _ in input]

    def default_metric(self) -> DistanceMetric:
        if self._model_name == "large":
            return DistanceMetric.COSINE
        elif self._model_name == "small":
            return DistanceMetric.L2
        else:
            raise ValueError(f"Unsupported model name: {self._model_name}")

    def build_from_config(self, config: Dict[str, Any]) -> Self:
        if "model_name" in config:
            self._model_name = config["model_name"]
        if "api_key_env_var" in config:
            self._api_key_env_var = config["api_key_env_var"]
        return self

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self._model_name,
            "api_key_env_var": self._api_key_env_var,
        }

    def modifiable_variables(self) -> List[str]:
        return ["api_key_env_var"]

    def register(self) -> None:
        supported_embedding_functions[self.name()] = self


class HNSWConfig(TypedDict, total=False):
    ef_search: int


class HNSWCreateConfig(TypedDict, total=False):
    ef_construction: int
    max_neighbors: int
    ef_search: int


class RuntimeConfig(TypedDict, total=False):
    num_threads: int
    batch_size: int
    sync_threshold: int
    resize_factor: float


class CreateCollectionConfig(TypedDict, total=False):
    hnsw: HNSWCreateConfig
    runtime: RuntimeConfig
    embedding_function: EmbeddingFunction[Embeddable]


class UpdateCollectionConfig(TypedDict, total=False):
    hnsw: HNSWConfig
    runtime: RuntimeConfig
    embedding_function: EmbeddingFunction[Embeddable]


# example usage


def create_collection(name: str, config: CreateCollectionConfig) -> None:
    pass


def update_collection(name: str, config: UpdateCollectionConfig) -> None:
    pass


cef = CohereEmbeddingFunction(model_name="large", api_key_env_var="COHERE_API_KEY")

create_collection(
    name="my_collection",
    config={
        "hnsw": {"max_neighbors": 100, "ef_search": 100},
        "runtime": {
            "num_threads": 10,
            "batch_size": 10,
            "sync_threshold": 0,
            "resize_factor": 1.0,
        },
        "embedding_function": cef,
    },
)


update_collection(
    "my_collection",
    {
        "hnsw": {"ef_search": 100},
        "runtime": {
            "num_threads": 10,
            "batch_size": 10,
            "sync_threshold": 0,
            "resize_factor": 1.0,
        },
        "embedding_function": cef,
    },
)

cef.generate_embeddings(["Hello, world!"])
