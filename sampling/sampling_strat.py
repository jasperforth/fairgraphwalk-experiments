from abc import ABC, abstractmethod
from keyword import kwlist
from data_utils.graph.graph import Graph
from typing import List, Final
from pathlib import Path

import networkx as nx


class SamplingStrategy(ABC):
    def __init__(self, p: float, q: float, walk_length: int = 80, 
                    num_walks: int = 10,quiet: bool = False) -> None:
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.quiet = quiet
        

    @abstractmethod
    def generate_walks(self, graph: Graph) -> List[str]:
        # not sure what they node ids will be, but should be a list of node ids
        pass

    @staticmethod
    @abstractmethod
    def precompute_transition_probabilities() -> dict:
        pass


