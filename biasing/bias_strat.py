from abc import ABC, abstractmethod
from data_utils.graph.graph import Graph
from pathlib import Path

#TODO implement quiet global


# different ways of biasing / adapting the original graph's edge weights
class BiasStrategy(ABC):
    @abstractmethod
    def __init__(self, graph: Graph, experiment_graph_dir: Path,
                    sensitive_attribute_name: str = None, 
                    alpha: float=None, exponent: float = None, 
                    prewalk_length: int = 6,
                    quiet: bool = False) -> None:
        self.graph = graph
        self.experiment_graph_dir = experiment_graph_dir
        self.sensitive_attribute_name = sensitive_attribute_name
        self.alpha = alpha
        self.exponent = exponent
        self.prewalk_length = prewalk_length
        self.quiet = quiet
    
        
    @abstractmethod 
    def adapt_weights(self) -> Graph:
        pass
