
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

class EncodingStrategy(ABC):
    def __init__(self, params_signature: str, experiment_graph_dir: Path, embedding_dimension: int = 128, embedding_params=None):
        self.params_signature = params_signature
        self.experiment_graph_dir = experiment_graph_dir
        self.embedding_dimension = embedding_dimension
        if embedding_params is None:
            embedding_params = {}
        if not isinstance(embedding_params, dict):
            raise TypeError("embedding_params must be a dict")
        self.embedding_params = embedding_params
        

    @staticmethod
    @abstractmethod
    def fit(self, walks: List[str]):
        pass

    @staticmethod
    @abstractmethod
    def embedding_to_file(self, model):
        pass


