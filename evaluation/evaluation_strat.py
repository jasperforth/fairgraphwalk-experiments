from __future__ import annotations

from abc import ABC, abstractmethod
from data_utils.graph.graph import Graph
from pathlib import Path



class EvaluationStrategy(ABC):
    def __init__(self,  result_dir: Path, params_signature: str, sensitive_attribute_name: str, other_attribute_name: str, train_size: float ):
        self.result_dir = result_dir
        self.params_signature = params_signature
        self.sensitive_attribute_name = sensitive_attribute_name
        self.other_attribute_name = other_attribute_name
        self.train_size = train_size
        

    @abstractmethod
    def evaluate(self, graph: Graph, embedding_path: str):
        # train / test split could go here
        pass

    