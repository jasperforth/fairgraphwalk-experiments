from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd

class Graph(ABC):
    @abstractmethod
    def __init__(self, graph: Graph, attributes: pd.DataFrame):
        self.graph = graph
        self.attributes = attributes
    
    @abstractmethod
    def graph_from_nxgraph(self, g: Graph, df_attributes: pd.DataFrame):
        return Graph(g, df_attributes)

    @staticmethod
    @abstractmethod
    def graph_from_edgelist(self, edgelist_path, attributes_path) -> Graph:
        pass
