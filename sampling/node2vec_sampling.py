from __future__ import annotations
import logging
import os

from collections import defaultdict
from typing import List
from pathlib import Path

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from data_utils.graph.graph import Graph
from .sampling_strat import SamplingStrategy
from experiment_utils.logging_utils import setup_worker_logging
from experiment_utils.config import DATA_DIR

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
import random

# Configure logging
log_dir = DATA_DIR
setup_worker_logging("node2vec_sampling", log_dir)
logger = logging.getLogger(__name__)

class Node2VecSampling(SamplingStrategy):
    """
    Node2Vec sampling strategy for generating random walks on a graph.

    This implementation is based on the node2vec algorithm described in:

    Grover, A., & Leskovec, J. (2016). 
    node2vec: Scalable Feature Learning for Networks. 
    Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD). 
    https://dl.acm.org/doi/10.1145/2939672.2939754
    Python implementation available at: https://github.com/eliorc/node2vec

    Args:
        p (float): Return parameter for node2vec.
        q (float): In-out parameter for node2vec.
        graph_name (str): Name of the graph being processed.
        walk_length (int, optional): Length of each random walk. Defaults to 80.
        num_walks (int, optional): Number of walks per node. Defaults to 10.
        quiet (bool, optional): Flag to control verbosity. Defaults to False.
    """
    def __init__(self, p: float, q: float, graph_name: str, walk_length: int = 80, 
                    num_walks: int = 10, quiet: bool = False) -> None:
        super().__init__(p, q, walk_length, num_walks, quiet)
        self.d_graph = defaultdict(dict)
        self.graph_name = graph_name

    def generate_walks(self, graph: Graph) -> List[int]:
        """
        Generate random walks on the given graph.
        
        Args:
            graph (Graph): The input graph for generating walks.
        
        Returns:
            List[int]: A list of generated random walks.
        """
        first_travel_key = 'first_travel_key'
        neighbors_key = 'neighbors'
        prob_key = 'probabilities'
        weight_key = 'weight'

        g = graph.graph.to_directed()
        walks = []
        d_graph = self.precompute_transition_probabilities(g, self.d_graph, self.p, self.q, self.quiet,
                                                        first_travel_key, neighbors_key, prob_key, weight_key)
        # how does the defaultdict look like:                                                
        #first_20 = list(d_graph.items())[:5]; print("DGRAPH", first_20, "DGRAPH")
        
        if not self.quiet:
            with logging_redirect_tqdm():
                pbar = tqdm(total=self.num_walks, desc=f'Generating walks for {self.graph_name} params p={self.p}, q={self.q}') 

        for n_walk in range(self.num_walks):
            if not self.quiet:
                pbar.update(1)
            
            shuffled_nodes = list(d_graph.keys())
            random.shuffle(shuffled_nodes)
            #print('length of shuffled nodes', len(shuffled_nodes), 'shuffled nodes', shuffled_nodes[:10], 'shuffled nodes')     
           
            for source in shuffled_nodes:
                walk = [source]
                # Perform walk
                while len(walk) < self.walk_length:
                    walk_options = self.d_graph[walk[-1]].get(neighbors_key, None)
                    # Skip dead ends
                    if not walk_options:
                        break
                    if len(walk) == 1:
                        probabilities = self.d_graph[walk[-1]][first_travel_key]
                        walk_to = random.choices(
                        walk_options, weights=probabilities)[0]
                    else:
                        probabilities = self.d_graph[walk[-1]][prob_key][walk[-2]]
                        walk_to = random.choices(walk_options,weights=probabilities)[0]
                    
                    walk.append(walk_to)
                # Convert all to strings for NLP-style Encoding  
                walk = list(map(str, walk))
                walks.append(walk)

        if not self.quiet:
            pbar.close()    

        sampled_walks = [_ for _ in walks]
        return sampled_walks

    @staticmethod
    def precompute_transition_probabilities(graph: Graph, d_graph: defaultdict(dict), 
                                p: float, q: float, quiet: bool, first_travel_key: str, 
                                neighbors_key: str, prob_key: str, weight_key: str) -> defaultdict(dict):
        """
        Precompute transition probabilities for node2vec walks.
        
        Args:
            graph (Graph): The input graph for generating walks.
            d_graph (defaultdict): Data structure to store the transition probabilities.
            p (float): Return parameter.
            q (float): In-out parameter.
            quiet (bool): Flag to control verbosity.
            first_travel_key (str): Key for first travel probabilities.
            neighbors_key (str): Key for neighbors.
            prob_key (str): Key for probabilities.
            weight_key (str): Key for edge weights.
        
        Returns:
            defaultdict(dict): Data structure with precomputed transition probabilities.
        """
        nodes_generator = graph.nodes() if quiet \
                                        else tqdm(graph.nodes(), 
                                        desc='Computing node2vec graph') 
        
        for source in nodes_generator:
            if prob_key not in d_graph[source]:
                d_graph[source][prob_key] = dict()
            
            for current in graph.neighbors(source):
                if prob_key not in d_graph[current]:
                    d_graph[current][prob_key] = dict()
                unnormalized_weights = list()
                d_neighbors = list()
                # 2-hop neighborhood
                for destination in graph.neighbors(current):
                    weight = graph[current][destination].get('weight')
                    #print(weigth)

                    if destination == source:
                        unnormalized_weight = weight * 1 / p
                    elif destination in graph[source]:
                        unnormalized_weight = weight 
                    else:
                        unnormalized_weight = weight * 1 / q

                    unnormalized_weights.append(unnormalized_weight)
                    d_neighbors.append(destination)
                
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current][prob_key][source] = unnormalized_weights / unnormalized_weights.sum()
                    
            first_travel_weights = []
            for destination in graph.neighbors(source):
                first_travel_weights.append(graph[source][destination].get(weight_key, 1))
            
            first_travel_weights = np.array(first_travel_weights)
            d_graph[source][first_travel_key] = first_travel_weights / first_travel_weights.sum()
     
            d_graph[source][neighbors_key] = list(graph.neighbors(source))
        
        return d_graph