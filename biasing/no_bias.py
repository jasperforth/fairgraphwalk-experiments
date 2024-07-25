from .bias_strat import BiasStrategy
from data_utils.graph.graph import Graph
from pathlib import Path


class NoBias(BiasStrategy):
    def __init__(self, graph: Graph, experiment_graph_dir: Path = None,
                    sensitive_attribute_name: str = None, 
                    alpha: float=None, exponent: float = None, 
                    prewalk_length: int = 6,   
                    quiet: bool = False) -> None:
        super().__init__(graph, experiment_graph_dir, 
                         sensitive_attribute_name, 
                         alpha, exponent, prewalk_length, quiet)
    


    def adapt_weights(self) -> Graph:
        return self.graph
