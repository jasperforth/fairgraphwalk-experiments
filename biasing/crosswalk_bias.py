import os
import logging
import pandas as pd

from pathlib import Path

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Set the number of CPUs to run on and environment variables for parallel processing, to be set before importing numpy
# https://rcpedia.stanford.edu/topicGuides/parallelProcessingPython.html
# Note: parallelization is managed on experiment_run level, to avoid subparallelization set ncore to 1!
ncore = "1"
os.environ["OMP_NUM_THREADS"] = ncore
os.environ["OPENBLAS_NUM_THREADS"] = ncore
os.environ["MKL_NUM_THREADS"] = ncore
os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
os.environ["NUMEXPR_NUM_THREADS"] = ncore

import numpy as np

from data_utils.graph.graph import Graph
from biasing.bias_strat import BiasStrategy
from experiment_utils.logging_utils import setup_worker_logging
from experiment_utils.config import DATA_DIR

# Configure logging
log_dir = DATA_DIR
setup_worker_logging("crosswalk bias", log_dir)
logger = logging.getLogger(__name__)

# TODO cite: based on crosswalk from github
# TODO movedicrected into graph!
class CrossWalkBias(BiasStrategy):
    """
    CrossWalk biasing strategy for adapting graph edge weights based on node attributes.
    """
    def __init__(self, graph: Graph, experiment_graph_dir: Path,
                    sensitive_attribute_name: str = None, 
                    alpha: float=None, exponent: float = None, 
                    graph_name: str = None,
                    prewalk_length: int = 6, quiet: bool = False) -> None:
        """
        Initialize the CrossWalkBias.
        
        Args:
            graph (Graph): Input graph.
            experiment_graph_dir (Path): Directory for the experiment graph.
            sensitive_attribute_name (str, optional): Name of the sensitive attribute. Defaults to None.
            alpha (float, optional): Alpha parameter for biasing. Defaults to None.
            exponent (float, optional): Exponent parameter for biasing. Defaults to None.
            graph_name (str, optional): Name of the graph. Defaults to None.
            prewalk_length (int, optional): Length of the prewalk. Defaults to 6.
            quiet (bool, optional): Flag to control verbosity. Defaults to False.
        """
        super().__init__(graph, experiment_graph_dir, 
                         sensitive_attribute_name, 
                         alpha, exponent, prewalk_length, quiet)
        self.g = graph.graph
        self.df_attributes = graph.attributes
        self.graph_name = graph_name
        if sensitive_attribute_name:
            logger.info(f"Using {sensitive_attribute_name} as sensitive attribute")
            self.df_sens = self.df_attributes.set_index('user_id')
            self.df_sens = self.df_sens[self.sensitive_attribute_name]

    def adapt_weights(self) -> Graph:
        """
        Adapt graph edge weights based on sensitive attributes.
        
        Returns:
            Graph: Adapted graph.
        """
        cfn_path = self.experiment_graph_dir / f'colorfulness_sens_{self.sensitive_attribute_name}_prewalklength_{self.prewalk_length}.csv'

        if not cfn_path.exists():
            self.pre_compute_biasing_params(self)
        else:
            logger.info(f'Loading colorfulness from file for {self.sensitive_attribute_name} \
                        and {self.graph_name} with prewalk length: {self.prewalk_length}')
        df_cfn = pd.read_csv(cfn_path).set_index('user_id')['colorfulness']

        crosswalk_graph = self.compute_crosswalk_graph(self.g, self.alpha, self.exponent, self.graph_name,
                                                          self.df_sens, df_cfn, self.quiet)

        self.graph = self.graph.graph_from_nxgraph(crosswalk_graph, self.df_attributes)
        #print("ADAPTWEIGHTSCROSSWALK", self.graph.graph, "ADAPTWEIHTSCROSSWALKAttributes", self.graph.attributes)
        
        return self.graph

    def pre_compute_biasing_params(self):
        """
        Precompute neighborhood colorfulness parameters.
        """
        cfn_path = self.experiment_graph_dir / f'colorfulness_sens_{self.sensitive_attribute_name}_prewalklength_{self.prewalk_length}.csv'

        if not cfn_path.exists():
            with logging_redirect_tqdm():
                colorfulness = self.colorfulness(self.prewalk_length, self.g, self.df_sens, self.graph_name, self.quiet)
                df_cfn = pd.DataFrame.from_dict(colorfulness, orient='index', columns=['colorfulness'])
                df_cfn.index.name = 'user_id'
                df_cfn.to_csv(cfn_path)

    @staticmethod
    def colorfulness(l: int, graph: Graph, df_sens: pd.DataFrame, graph_name: str, quiet: bool) -> dict:
        """
        Compute the colorfulness for each node in the graph.
        
        Args:
            l (int): Length of the prewalk.
            graph (Graph): Input graph.
            df_sens (pd.DataFrame): DataFrame containing sensitive attribute information.
            graph_name (str): Name of the graph.
            quiet (bool): Flag to control verbosity.
        
        Returns:
            dict: Dictionary with colorfulness values for each node.
        """
        xtqdm_graph = graph if quiet else tqdm(graph, desc=f"Computing boundary proximites for graph {graph_name} prewalk_len {l}")
        map_results = [CrossWalkBias.node_colorfulness(source, l, graph, df_sens) for source in xtqdm_graph]
        cfn = {k: source for k, source in map_results}
        return cfn

    @staticmethod
    def node_colorfulness(source: int, l: int, graph: Graph, df_sens: pd.DataFrame):
        """
        Compute the colorfulness for a single node.
        
        Args:
            source (int): Source node.
            l (int): Length of the prewalk.
            graph (Graph): Input graph.
            df_sens (pd.DataFrame): DataFrame containing sensitive attribute information.
        
        Returns:
            tuple: Node and its colorfulness value.
        """
        n_prewalks = 1000 # # r=1000 in the paper
        res = 0.001 + np.mean([CrossWalkBias.randomwalk_colorfulness
                                (source, l, graph, df_sens) for _ in range(n_prewalks)])  
        return (source, res)


    @staticmethod
    def randomwalk_colorfulness(source: int, l: int, graph: Graph, df_sens: pd.DataFrame):
        """
        Perform a random walk and compute the colorfulness.
        
        Args:
            source (int): Source node.
            l (int): Length of the random walk.
            graph (Graph): Input graph.
            df_sens (pd.DataFrame): DataFrame containing sensitive attribute information.
        
        Returns:
            float: Colorfulness value.
        """
        source_color = df_sens[source]
        current = source
        res = 0
        for i in range(l):
            current = np.random.choice(graph[current])
            if df_sens[current] != source_color:
                res += 1
        return res / l


    @staticmethod
    def compute_crosswalk_graph(graph: Graph, alpha: float, exp: float, graph_name: str, df_sens: pd.DataFrame, df_cfn: pd.DataFrame, quiet: bool):
        """
        Compute the crosswalk graph by adapting edge weights.
        
        Args:
            graph (Graph): Input graph.
            alpha (float): Alpha parameter for biasing.
            exp (float): Exponent parameter for biasing.
            graph_name (str): Name of the graph.
            df_sens (pd.DataFrame): DataFrame containing sensitive attribute information.
            df_cfn (pd.DataFrame): DataFrame containing colorfulness information.
            quiet (bool): Flag to control verbosity.
        
        Returns:
            Graph: Adapted graph with new edge weights.
        """
        if not graph.is_directed():
            graph = graph.to_directed()
            logger.info("Graph is not directed, converting to directed graph")

        nodes_generator = graph.nodes() if quiet \
                                        else tqdm(graph.nodes(), 
                                        desc=f'Computing crosswalk graph {graph_name} for alpha: {alpha} and exp: {exp}')
        # Implementation based on Crosswalkpaper pseudocode and Repository..
        for source in nodes_generator:
            nei_colors = np.unique([df_sens[destination] for destination in graph[source]])
            if nei_colors.size == 0:
                continue
            w_n = [df_cfn[destination] ** exp for destination in graph[source]]

            if nei_colors.size == 1 and nei_colors[0] == df_sens[source]:
                w_n = [alpha * w for w in w_n]
                _sum = sum(w_n)

                for i, destination in enumerate(graph.neighbors(source)):
                    weight = w_n[i] / _sum
                    graph.add_edge(int(source), int(destination), weight=weight)
                continue

            for color in nei_colors:    
                ind_color = [i for i, destination in enumerate(
                    graph.neighbors(source)) if df_sens[destination] == color]  
                w_n_color = [w_n[i] for i in ind_color] 
                _sum_color = sum(w_n_color)  
                if color == df_sens[source]:  
                    coef = (1 - alpha)
                else: 
                    if df_sens[source] in nei_colors: 
                        coef = alpha / (nei_colors.size - 1)
                    else: 
                        coef = 1 / nei_colors.size

                for i in ind_color:
                    neighbors = [n for n in graph.neighbors(source)]
                    weight = coef * w_n[i] / _sum_color
                    graph.add_edge(int(source), int(
                        neighbors[i]), weight=weight)
        
        return graph