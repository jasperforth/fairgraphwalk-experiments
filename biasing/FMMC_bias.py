import os
import logging
import math
import statistics
import sys

import pandas as pd
import csv

from pathlib import Path

# from fmmc.fmmc_reweighting import fmmc_reweighting
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
import scipy as sp
import networkx as nx
import cvxpy as cp

from data_utils.graph.graph import Graph
from biasing.bias_strat import BiasStrategy
from experiment_utils.logging_utils import setup_worker_logging
from experiment_utils.config import DATA_DIR

# Configure logging
log_dir = DATA_DIR
setup_worker_logging("fmmc bias", log_dir)
logger = logging.getLogger(__name__)

class FMMCBias(BiasStrategy):
    """
    FMMC biasing strategy for adapting graph edge weights based on node attributes.
    """

    def __init__(self, graph: Graph, experiment_graph_dir: Path,
                 sensitive_attribute_name: str = None,
                 alpha: float = None, exponent: float = None,
                 graph_name: str = None,
                 prewalk_length: int = 6, 
                 quiet: bool = False, 
                 selfloops: bool = False) -> None:
        """
        Initialize the FMMCBias.

        Args:
            graph (Graph): Input graph.
            experiment_graph_dir (Path): Directory for the experiment graph.
            sensitive_attribute_name (str, optional): Name of the sensitive attribute. Defaults to None.
            alpha (float, optional): Alpha parameter for biasing. Defaults to None.
            exponent (float, optional): Exponent parameter for biasing. Defaults to None.
            graph_name (str, optional): Name of the graph. Defaults to None.
            prewalk_length (int, optional): Length of the prewalk. Defaults to 6.
            quiet (bool, optional): Flag to control verbosity. Defaults to False.
            selfloops (bool): Flag to control self loops. Defaults to False.
        """
        super().__init__(graph, experiment_graph_dir,
                         sensitive_attribute_name,
                         alpha, exponent, prewalk_length, quiet)

        # Determine experiment graph directory based on selfloops flag
        if selfloops:
            selfloops_directory_path = self.experiment_graph_dir = experiment_graph_dir / "selfloops"
        else:
            selfloops_directory_path = self.experiment_graph_dir = experiment_graph_dir / "NOselfloops"
        if not selfloops_directory_path.exists():
            self.create_directory(selfloops_directory_path)

        self.g = graph.graph
        self.df_attributes = graph.attributes
        self.graph_name = graph_name
        self.selfloops = selfloops
        if sensitive_attribute_name:
            logger.info(f"Using {sensitive_attribute_name} as sensitive attribute")
            self.df_sens = self.df_attributes.set_index('user_id')
            self.df_sens = self.df_sens[self.sensitive_attribute_name]

    def adapt_weights(self) -> Graph:
        """
        Adapt graph edge weights based on FMMC

        Returns:
            Graph: Adapted graph.
        """
        csv_path_weighted_edgelist = self.experiment_graph_dir / f'weighted_edgelist_selfloops_{self.selfloops}.csv'
        #checkpoint
        if csv_path_weighted_edgelist.exists():
            logger.info(f"FMMC edge weights already exist for {self.graph_name} with selfloops {self.selfloops}. Reading {csv_path_weighted_edgelist}.")
            weighted_edgelist: list[tuple[int,int,float]] = []

            with open(csv_path_weighted_edgelist, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    edge_tuple = (int(row[0]), int(row[1]), float(row[2]))
                    weighted_edgelist.append(edge_tuple)

            temp_returngraph: nx.Graph = self.build_nxgraph_from_weighted_edgelist(weighted_edgelist)
            if not self.selfloops:
                temp_returngraph = self.remove_selfloops_from_nxgraph(temp_returngraph)
            self.graph = self.graph.graph_from_nxgraph(temp_returngraph, self.df_attributes)
            logger.info(f'Finished fmmc edge reweighting for {self.graph_name}')
            return self.graph

        # runtime becomes unreasonable for large graphs
        if self.g.number_of_edges() > 200000:
            raise ValueError("Inputgraph has too many edges (more than 200k). Runtime would be very long. Change threshhold in the code if you are sure")

        #save unencoded edge list of original graph (all edge weights are 1)
        original_edgelist: list[tuple[int,int,float]] = []
        for u, v in self.g.edges():
            original_edgelist.append((u, v, 1.0))
        original_weights_csv_path = self.experiment_graph_dir / f'original_weighted_edgelist.csv'
        with open(original_weights_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for u, v, weight in original_edgelist:
                writer.writerow([str(u), str(v), str(weight)])

        #one hot encode the input graph and build temp graph
        node_to_id, id_to_node = self.create_mappings(self.g)
        one_hot_edges = self.one_hot_encode_edges(self.g, node_to_id)
        temp_g = nx.Graph()
        temp_g.add_edges_from(one_hot_edges)

        #save one hot encoding to csvs
        node_to_id_csv_path = self.experiment_graph_dir / f'node_to_id.csv'
        id_to_node_csv_path = self.experiment_graph_dir / f'id_to_node.csv'
        one_hot_edges_csv_path = self.experiment_graph_dir / f'one_hot_edges.csv'
        with open(node_to_id_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for node, idx in node_to_id.items():
                writer.writerow([node, idx])
        with open(id_to_node_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for idx, node in id_to_node.items():
                writer.writerow([idx, node])
        with open(one_hot_edges_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for u_id, v_id in one_hot_edges:
                writer.writerow([u_id, v_id])

        # initial setup starting from networkx input graph
        transitionmatrix: np.ndarray = self.build_nptransitionmatrix_from_nxgraph(temp_g)
        graph_info: dict[str, any] = self.get_graph_info(transitionmatrix)
        transition_probs: dict[str, np.ndarray] = {}
        SLEM_map: dict[str, float] = {}

        # build and save Maximum-Degree and Metropolis-Hastings chains
        fmmc_md_csv_path = self.experiment_graph_dir / f'encoded_fmmc_edgelist_md.csv'
        if fmmc_md_csv_path.exists():
            logger.info(f'{fmmc_md_csv_path} exists. Loading checkpoint.')
            transition_probs["md"] = self.get_transitionmatrix_from_weighted_edgelist_checkpoint(fmmc_md_csv_path, graph_info["n"])
        else:
            transition_probs["md"] = self.get_maximum_degree_p(graph_info)
            self.save_transitionmatrix_as_weighted_edgelist_in_csv(fmmc_md_csv_path, transition_probs["md"])
            logger.info("md done")
        fmmc_mh_csv_path = self.experiment_graph_dir / f'encoded_fmmc_edgelist_mh.csv'
        if fmmc_mh_csv_path.exists():
            logger.info(f'{fmmc_mh_csv_path} exists. Loading checkpoint.')
            transition_probs["mh"] = self.get_transitionmatrix_from_weighted_edgelist_checkpoint(fmmc_mh_csv_path, graph_info["n"])
        else:
            transition_probs["mh"] = self.get_metropolis_hastings_p(transitionmatrix, graph_info)
            self.save_transitionmatrix_as_weighted_edgelist_in_csv(fmmc_mh_csv_path, transition_probs["mh"])
            logger.info("mh done")

        for key, value in transition_probs.items():
            SLEM_map[key] = abs(self.get_SLEM_signed(value))

        # for graphs with less than 10k edges use a cvxpy solver. Can be extended with other solvers following the same pattern.
        # cvxpy solvers can cause errors for memory or convergece issues so in that case the SLEM gets set to 999 ensuring at least mh or md are lower.
        if graph_info["edgecount"] <= 10000:
            fmmc_SCS_csv_path = self.experiment_graph_dir / f'encoded_fmmc_edgelist_fmmc_SCS.csv'
            if fmmc_SCS_csv_path.exists():
                logger.info(f'{fmmc_SCS_csv_path} exists. Loading checkpoint.')
                transition_probs["fmmc_SCS"] = self.get_transitionmatrix_from_weighted_edgelist_checkpoint(fmmc_SCS_csv_path, graph_info["n"])
                try:
                    SLEM_map["fmmc_SCS"] = abs(np.linalg.eigvals(transition_probs["fmmc_SCS"])[-2])
                    transition_probs["fmmc_SCS"] = self.clean_transitionmatrix(transition_probs["fmmc_SCS"])
                except Exception as e:
                    logger.info(f'SLEM for SCS could not be calculated: {e}.')
                    logger.info(f'Setting SLEM for SCS to 999 for {self.graph_name}')
                    SLEM_map["fmmc_SCS"] = 999
                    transition_probs["fmmc_SCS"] = self.clean_transitionmatrix(transition_probs["fmmc_SCS"])
            else:
                transition_probs["fmmc_SCS"] = self.get_fmmc_transitionmatrix(graph_info, cp.SCS)
                self.save_transitionmatrix_as_weighted_edgelist_in_csv(fmmc_SCS_csv_path, transition_probs["fmmc_SCS"])
                logger.info("fmmc_SCS done")
                try:
                    SLEM_map["fmmc_SCS"] = abs(np.linalg.eigvals(transition_probs["fmmc_SCS"])[-2])
                    transition_probs["fmmc_SCS"] = self.clean_transitionmatrix(transition_probs["fmmc_SCS"])
                except Exception as e:
                    logger.info(f'SLEM for SCS could not be calculated: {e}.')
                    logger.info(f'Setting SLEM for SCS to 999 for {self.graph_name}')
                    SLEM_map["fmmc_SCS"] = 999
                    transition_probs["fmmc_SCS"] = self.clean_transitionmatrix(transition_probs["fmmc_SCS"])

        # for graphs with more than 1k edges use subgradient method
        if graph_info["edgecount"] > 1000:
            fmmc_subgradient_csv_path = self.experiment_graph_dir / f'encoded_fmmc_edgelist_subgradient.csv'
            if fmmc_subgradient_csv_path.exists():
                logger.info(f'{fmmc_subgradient_csv_path} exists. Loading checkpoint.')
                transition_probs["subgradient"] = self.get_transitionmatrix_from_weighted_edgelist_checkpoint(fmmc_subgradient_csv_path, graph_info["n"])
                logger.info(f'Loading SLEM_per_k checkpoint.')
                SLEM_per_k_checkpoint_path = self.experiment_graph_dir / f'SLEM_per_k.csv'
                SLEM_per_k_from_checkpoint: dict[int, float] = {}
                with open(SLEM_per_k_checkpoint_path, 'r') as SLEM_checkpoint_file:
                    csv_reader = csv.reader(SLEM_checkpoint_file)
                    for row in csv_reader:
                        SLEM_per_k_from_checkpoint[int(row[0])] = float(row[1])
                logger.info(f'Getting SLEM for loaded subgradient.')
                SLEM_map["subgradient"] = list(SLEM_per_k_from_checkpoint.values())[-1]
            else:
                logger.info(f'Starting subgradient computation for {self.graph_name}')
                checkpoint_dir = self.experiment_graph_dir / f'subgradient_checkpoints'
                if not checkpoint_dir.exists():
                    self.create_directory(checkpoint_dir)
                if SLEM_map["mh"] <= SLEM_map["md"]:
                    subgradient_SLEM_per_k, subgradient_transitionmatrix = self.get_subgradient_transitionmatrix(transition_probs["mh"], graph_info, self.experiment_graph_dir)
                else:
                    subgradient_SLEM_per_k, subgradient_transitionmatrix = self.get_subgradient_transitionmatrix(transition_probs["md"], graph_info, self.experiment_graph_dir)
                graph_info["subgradient_SLEM_per_k"] = subgradient_SLEM_per_k
                SLEM_map["subgradient"] = list(subgradient_SLEM_per_k.values())[-1]
                print(SLEM_map["subgradient"])
                logger.info(f'Finished subgradient computation for {self.graph_name}')

                #saving SLEM_per_k to a csv
                csv_path_SLEM_per_k = self.experiment_graph_dir / f'SLEM_per_k.csv'
                with open(csv_path_SLEM_per_k, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if SLEM_map["mh"] <= SLEM_map["md"]:
                        writer.writerow(["0", str(SLEM_map["mh"])])
                    else:
                        writer.writerow(["0", str(SLEM_map["md"])])
                    for key, value in subgradient_SLEM_per_k.items():
                        writer.writerow([str(key), str(value)])

                self.save_transitionmatrix_as_weighted_edgelist_in_csv(fmmc_subgradient_csv_path, subgradient_transitionmatrix)
                transition_probs["subgradient"] = self.clean_transitionmatrix(subgradient_transitionmatrix)

        # for key, value in transition_probs.items():
        #     if not key in SLEM_map:
        #         SLEM_map[key] = abs(self.get_SLEM_signed(value))
        graph_info["min_SLEM"] = min(SLEM_map, key=SLEM_map.get)

        #decode one hot encoding
        encoded_weighted_edgelist = self.get_weighted_edgelist_from_transitionmatrix(transition_probs[graph_info["min_SLEM"]])
        decoded_weighted_edgelist = self.get_weighted_decoded_edgelist(encoded_weighted_edgelist, id_to_node)
        temp_returngraph : nx.Graph = self.build_nxgraph_from_weighted_edgelist(decoded_weighted_edgelist)

        if not self.selfloops:
            temp_returngraph = self.remove_selfloops_from_nxgraph(temp_returngraph)

        # finished graph after FMMC reweighting
        self.graph = self.graph.graph_from_nxgraph(temp_returngraph, self.df_attributes)
        logger.info(f'Finished fmmc edge reweighting for {self.graph_name}')

        #saving FMMC reporting to csvs
        csv_path_info = self.experiment_graph_dir / f'graphinfo_selfloops_{self.selfloops}.csv'

        header = ['n','edgecount','d_max','d_avg','d_median','min_SLEM','SLEM_mh','SLEM_md','SLEM_SCS',
                  # 'SLEM_CVXOPT',
                  # 'SLEM_CLARABEL'
                  ]
        data = [str(graph_info.get("n")),str(graph_info.get("edgecount")),str(graph_info.get("d_max")),str(graph_info.get("d_avg")),str(graph_info.get("d_median")),str(graph_info.get("min_SLEM")),str(SLEM_map.get("mh")),
                str(SLEM_map.get("md")),str(SLEM_map.get("fmmc_SCS")),
                # str(SLEM_map.get("fmmc_CVXOPT")),
                # str(SLEM_map.get("fmmc_CLARABEL"))
                ]

        if graph_info["edgecount"] > 1000:
            header.append('SLEM_subgradient')
            data.append(str(SLEM_map.get("subgradient")))

        with open(csv_path_info, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(data)

        #save decoded_weighted_edgelist in csv
        with open(csv_path_weighted_edgelist, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for u, v, weight in decoded_weighted_edgelist:
                writer.writerow([str(u), str(v), str(weight)])

        #save all fmmc matrices as weighted edge lists in csvs
        for key, transitionmatrix in transition_probs.items():
            fmmc_csv_path = self.experiment_graph_dir / f'encoded_fmmc_edgelist_{key}.csv'
            if not fmmc_csv_path.exists():
                self.save_transitionmatrix_as_weighted_edgelist_in_csv(fmmc_csv_path, transitionmatrix)

        return self.graph

    @staticmethod
    def build_nptransitionmatrix_from_nxgraph(graph: nx.Graph) -> np.ndarray:
        """
        builds a numpy transitionmatrix from a networkx graph

        Args:
            graph (nx.Graph): networkx graph

        Returns:
            np.ndarray: numpy transitionmatrix
        """
        n: int = graph.number_of_nodes()
        edges: list[tuple[int, int]] = graph.edges()
        transitionmatrix = np.zeros((n, n))

        for edge in edges:
            transitionmatrix[edge[0]][edge[1]] = 1
            transitionmatrix[edge[1]][edge[0]] = 1

        return transitionmatrix

    @staticmethod
    def get_graph_info(transitionmatrix: np.ndarray) -> dict:
        """
        gets relevant graph info of input graph pre biasing used in multiple following functions

        Args:
            transitionmatrix (np.ndarray): transition matrix of input graph

        Returns:
            dict[str, any]: graph info
        """
        graph_info: dict[str, any] = {}
        n: int = transitionmatrix.shape[0]
        d_max: int = 0
        degrees: list[tuple[int, int]] = []
        zeroes: list[list[int]] = []
        edgecount: float = np.count_nonzero(transitionmatrix) / 2  # rename to edgecount

        for i in range(n):
            d: int = 0
            zeroes.append([])

            for j in range(n):
                if i == j:
                    continue
                if transitionmatrix[i][j] != 0:
                    d += 1
                else:
                    zeroes[i].append(j)

            degrees.append((i, d))
            d_max = max(d_max, d)

        d_per_node: list[int] = []
        for k in range(len(degrees)):
            d_per_node.append(degrees[k][1])

        d_avg: float = sum(d_per_node) / len(d_per_node)
        d_median: float = statistics.median(d_per_node)

        graph_info["n"] = n
        graph_info["edgecount"] = edgecount
        graph_info["d_max"] = d_max
        graph_info["d_avg"] = d_avg
        graph_info["d_median"] = d_median
        graph_info["degrees"] = degrees
        graph_info["zeroes"] = zeroes

        # if edgecount > 20000:
        #     #add stuff for subgradient

        return graph_info

    @staticmethod
    def get_maximum_degree_p(graph_info: dict) -> np.ndarray:
        """
        implementation of the maximum degree chain

        Args:
            graph_info (dict): graph info

        Returns:
            np.ndarray: maximum degree chain transition matrix
        """
        n: int = graph_info["n"]
        d_max: int = graph_info["d_max"]
        degrees: list[tuple[int, int]] = graph_info["degrees"]
        zeroes: list[list[int]] = graph_info["zeroes"]

        p_md: np.ndarray = np.zeros((n, n))

        for i in range(n):
            d: int = degrees[i][1]

            for j in range(n):
                if i == j:
                    p_md[i, j] = 1 - d / d_max
                elif j in zeroes[i]:
                    p_md[i, j] = 0
                else:
                    p_md[i][j] = 1 / d_max

        return p_md

    @staticmethod
    def get_metropolis_hastings_p(transitionmatrix: np.ndarray, graph_info: dict) -> np.ndarray:
        """
        implementation of the metropolis hastings chain

        Args:
            transitionmatrix (np.ndarray): transition matrix of original graph pre biasing
            graph_info (dict): graph info

        Returns:
            np.ndarray: metropolis hastings chain transition matrix
        """
        n: int = graph_info["n"]
        degrees: list[tuple[int, int]] = graph_info["degrees"]
        zeroes: list[list[int]] = graph_info["zeroes"]

        p_mh: np.ndarray = np.zeros((n, n))

        for i in range(n):
            d_i: int = degrees[i][1]
            d_k_list: list[int] = []

            for j in range(n):
                if transitionmatrix[i][j] != 0 and i != j:
                    d_k_list.append(degrees[j][1])
            for j in range(n):
                d_j: int = degrees[j][1]
                if i == j:
                    sum_max: float = 0
                    for d_k in d_k_list:
                        sum_max += max(0.0, 1 / d_i - 1 / d_k)
                    p_mh[i, j] = sum_max
                elif j in zeroes[i]:
                    p_mh[i, j] = 0
                else:
                    p_mh[i, j] = min(1 / d_i, 1 / d_j)

        return p_mh

    @staticmethod
    def clean_transitionmatrix(transitionmatrix: np.ndarray) -> np.ndarray:
        """
        sets all values in transitionmatrix to 0 that are too close to 0 (typically because of rounding errors from cvxpy solvers)

        Args:
            transitionmatrix (np.ndarray): transition matrix

        Returns:
            np.ndarray: cleaned transition matrix
        """
        n: int = transitionmatrix.shape[0]

        for i in range(n):
            for j in range(n):
                if transitionmatrix[i][j] < 0.0001:
                    transitionmatrix[i][j] = 0

        return transitionmatrix

    @staticmethod
    def get_fmmc_transitionmatrix(graph_info: dict, solver: cp.settings) -> np.ndarray:
        """
        gets FMMC transitionmatrix using one of the installed cvxpy solvers

        Args:
            graph_info (dict): graph info
            solver (cp.settings): cvxpy solver

        Returns:
            np.ndarray: FMMC transitionmatrix
        """
        graph_n: int = graph_info["n"]
        zeroes: list[list[int]] = graph_info["zeroes"]

        OneVector = np.ones((graph_n, 1))
        I = np.identity(graph_n)
        P = cp.Variable((graph_n, graph_n), symmetric=True)
        s = cp.Variable()
        n = P.size

        i: int = 0
        P_zeros = np.ones((graph_n, graph_n))
        for zero in zeroes:
            for z in zero:
                P_zeros[i][z] = 0
            i += 1

        constraints = [P @ OneVector == OneVector]
        constraints += [P == P.T]
        constraints += [P >= 0]
        constraints += [cp.multiply(P, P_zeros) == P]
        constraints += [-s * I << P - (1 / n) * OneVector @ np.transpose(OneVector)]
        constraints += [P - (1 / n) * OneVector @ np.transpose(OneVector) << s * I]

        prob = cp.Problem(cp.Minimize(s), constraints)

        prob.solve(verbose=True, solver=solver)
        
        logger.info(f"type of P.value {type(P.value)}")
        print(type(P.value))

        return P.value

    @staticmethod
    def is_MOSEK_installed() -> bool:
        """
        checks if MOSEK is in the cvxpy installed solvers.
        This method needs to be extended to check the users OS and then check the relevant filepath for a MOSEK license.

        Returns:
            bool: True if MOSEK is installed, False otherwise.
        """
        return 'MOSEK' in cp.installed_solvers()

    @staticmethod
    def get_SLEM_signed(transitionmatrix: np.ndarray) -> float:
        """
        gets the signed SLEM with the scipy method

        Args:
            transitionmatrix (np.ndarray): the transition matrix

        Returns:
            float: the signed SLEM
        """
        return sp.sparse.linalg.eigsh(transitionmatrix, k=2, which='LM')[0][0]

    @staticmethod
    def get_subgradient_transitionmatrix(transitionmatrix: np.ndarray, graph_info: dict, experiment_graph_dir: Path) -> tuple[dict, np.ndarray]:
        """
        implementation of the FMMC subgradient method. All variable names correspond to the names in the paper:
        https://web.stanford.edu/~boyd/papers/pdf/fmmc.pdf

        works for now but convoluted and hard to read, improvements to do:
        - restructure rebuilding block p, P_p, B, p_info

        Args:
            transitionmatrix (np.ndarray): transition matrix
            graph_info (dict): graph info
            experiment_graph_dir (Path): experiment graph directory

        Returns:
            tuple[dict, np.ndarray]: (SLEM_per_k, subgradient transition matrix)
        """
        def _p_feasibility_on_step_k(step_k: int = 0):
            """
            check if p is feasible and remove illegal values if there are any

            Args:
                step_k (int): current step, defaults to 0 for initial cleanup before the first subgradient step
            """
            if (p >= 0).all():
                print("p>=0:")
                print(p >= 0)
            else:
                logger.info(f'making p feasible (non-negative) on step {step_k}')
                for i in range(len(p)):
                    if p[i] < 0:
                        p[i] = 0
                print("p>=0:")
                print(p >= 0)
            if (B @ p <= 1).all():
                print("Bp<=1:")
                print(B @ p <= 1)
            else:
                print(f'making p feasible (smaller or equal 1) on step {step_k}')
                logger.info(f'making p feasible (smaller or equal 1) on step {step_k}')
                for i in range(n):
                    if B[i] @ p > 1:
                        emergency = 0
                        while B[i] @ p > 1:
                            nonzeroes = np.nonzero(B[i])
                            p_nonzeroes = np.zeros(len(nonzeroes[0]))
                            for g in range(len(nonzeroes[0])):
                                p_nonzeroes[g] = p[nonzeroes[0][g]]
                            while np.sum(p_nonzeroes) > 1:
                                p_nonzeroes -= 0.000001
                                if (p_nonzeroes < 0).any():
                                    for h in range(len(p_nonzeroes)):
                                        if p_nonzeroes[h] < 0:
                                            p_nonzeroes[h] = 0
                            for q in range(len(nonzeroes[0])):
                                p[nonzeroes[0][q]] = p_nonzeroes[q]
                            if emergency > 5000:
                                break
                            emergency += 1
            if (p >= 0).all():
                print("p>=0 still holds:")
                print(p >= 0)
            else:
                logger.info(f'making p feasible again (non-negative) on step {step_k}')
                for i in range(len(p)):
                    if p[i] < 0:
                        p[i] = 0
                print("p>=0:")
                print(p >= 0)

        n: int = graph_info["n"]
        SLEM_per_k: dict[int, float] = {}

        # build initial p and P_p
        p = []
        p_info = []
        z = 0
        for i in range(n):
            for j in range(z, n):
                if i == j:
                    continue
                if transitionmatrix[i, j] != 0:
                    p.append(transitionmatrix[i, j])
                    p_info.append((i, j))
            z += 1

        p = np.array(p)
        p_info = np.array(p_info)

        B = np.zeros((n, len(p_info)))
        l_count = 1
        for l in p_info:
            B[l[0]][l_count - 1] = 1
            B[l[1]][l_count - 1] = 1
            l_count += 1

        _p_feasibility_on_step_k()

        l = 1
        E_sum = np.zeros((n, n))
        for p_l in p:
            E = np.zeros((n, n))
            E[p_info[l - 1][0]][p_info[l - 1][1]] = 1
            E[p_info[l - 1][1]][p_info[l - 1][0]] = 1
            E[p_info[l - 1][0]][p_info[l - 1][0]] = -1
            E[p_info[l - 1][1]][p_info[l - 1][1]] = -1
            l += 1
            E_sum = E_sum + (p_l * E)

        P_p = np.identity(n) + E_sum

        for i in range(n):
            for j in range(n):
                if P_p[i][j] < 0:
                    P_p[i][j] = 0

        ###########################################################################################################
        # subgradient
        eigen_data = {}
        k = 1
        k_max = 200
        while k <= k_max:
            #checkpoint
            checkpoint_path = experiment_graph_dir / f'subgradient_checkpoints/P_p_weighted_edgelist_step_{k}.csv'
            if checkpoint_path.exists():
                while checkpoint_path.exists():
                    k += 1
                    checkpoint_path = experiment_graph_dir / f'subgradient_checkpoints/P_p_weighted_edgelist_step_{k}.csv'
                k -= 1
                checkpoint_path = experiment_graph_dir / f'subgradient_checkpoints/P_p_weighted_edgelist_step_{k}.csv'
                logger.info(f'subgradient checkpoint for step {k} used')
                #P_p checkpoint
                checkpoint_weighted_edgelist = []
                with open(checkpoint_path, 'r') as checkpoint_file:
                    csv_reader = csv.reader(checkpoint_file)
                    for row in csv_reader:
                        edge_tuple = (int(row[0]), int(row[1]), float(row[2]))
                        checkpoint_weighted_edgelist.append(edge_tuple)
                P_p = np.zeros((n,n))
                for u,v,w in checkpoint_weighted_edgelist:
                    P_p[u][v] = w
                    P_p[v][u] = w
                #SLEM_per_k checkpoint
                SLEM_per_k_checkpoint_path = experiment_graph_dir / f'SLEM_per_k.csv'
                with open(SLEM_per_k_checkpoint_path, 'r') as SLEM_checkpoint_file:
                    csv_reader = csv.reader(SLEM_checkpoint_file)
                    for row in csv_reader:
                        SLEM_per_k[int(row[0])] = float(row[1])

                if k == k_max:
                    return SLEM_per_k, P_p

                k += 1

            print(f'Starting subgradient computation step {k}')
            logger.info(f'Starting subgradient computation step {k}')
            # subgradient step #################################################################################
            g_p = []

            if k == 1 or checkpoint_path.exists():
                eigen_k_1 = sp.sparse.linalg.eigsh(P_p, 2, which='LM')
                SLEM_signed = eigen_k_1[0][0]
                eigenvectors = eigen_k_1[1]
            else:
                eigenvectors = eigen_data[k - 1][1]
                SLEM_signed = eigen_data[k-1][0][0]
                SLEM_per_k[k-1] = abs(SLEM_signed)

            # builds eigenvector associated with second largest eigenvalue
            u = np.array([item[0] for item in eigenvectors])

            for i in range(len(p_info)):
                E = np.zeros((n, n))
                E[p_info[i][0]][p_info[i][1]] = 1
                E[p_info[i][1]][p_info[i][0]] = 1
                E[p_info[i][0]][p_info[i][0]] = -1
                E[p_info[i][1]][p_info[i][1]] = -1
                if SLEM_signed < 0:
                    g_p.append(-1 * np.transpose(u) @ E @ u)
                else:
                    g_p.append(np.transpose(u) @ E @ u)

            g_p = np.array(g_p)

            alpha_k = 1 / math.sqrt(k)
            p = p - (alpha_k * g_p / np.linalg.norm(g_p))

            # sequential projection step #######################################################################
            # a
            for i in range(len(p)):
                p[i] = max(p[i], 0)
            # b
            for i in range(n):
                I_i = B[i]
                emergency_exit = 0

                while I_i @ p > 1:
                    for l in range(len(I_i)):
                        if p[l] <= 0:
                            I_i[l] = 0

                    nonzeroes = np.nonzero(I_i)
                    p_nonzeroes = np.zeros(len(nonzeroes[0]))
                    for j in range(len(nonzeroes[0])):
                        p_nonzeroes[j] = p[nonzeroes[0][j]]
                    min_p_l = np.min(p_nonzeroes)

                    help_sum = 0
                    for l in range(len(I_i)):
                        if I_i[l] == 1:
                            help_sum += p[l]

                    delta = min(min_p_l, (help_sum - 1) / np.count_nonzero(I_i))
                    # if delta is 0 due to rounding errors it remains 0 until emergency exit is reached so set it to small value
                    if delta == 0:
                        delta = 0.00001

                    for p_l in range(len(p)):
                        if I_i[p_l] == 1:
                            p[p_l] = p[p_l] - delta

                    # to avoid infinite loop. if a row of B can't become feasible in 30k tries move on, will be cleaned
                    # in later block.
                    if emergency_exit == 30000:
                        logger.info(f'subgradient emergency exit used on step {k}')
                        break

                    emergency_exit += 1

                B[i] = I_i

            _p_feasibility_on_step_k(k)

            # for the new p: rebuild P_p and rebuild p, B and p_info for next step ###########################
            l = 1
            E_sum = np.zeros((n, n))
            for p_l in p:
                E = np.zeros((n, n))
                E[p_info[l - 1][0]][p_info[l - 1][1]] = 1
                E[p_info[l - 1][1]][p_info[l - 1][0]] = 1
                E[p_info[l - 1][0]][p_info[l - 1][0]] = -1
                E[p_info[l - 1][1]][p_info[l - 1][1]] = -1
                l += 1
                E_sum = E_sum + (p_l * E)

            P_p = np.identity(n) + E_sum

            for i in range(n):
                for j in range(n):
                    if P_p[i][j] < 0:
                        P_p[i][j] = 0

            p = []
            p_info = []
            z = 0
            for i in range(n):
                for j in range(z, n):
                    if i == j:
                        continue
                    if P_p[i, j] != 0:
                        p.append(P_p[i, j])
                        p_info.append((i, j))
                z += 1

            p = np.array(p)
            p_info = np.array(p_info)

            B = np.zeros((n, len(p_info)))
            l_count = 1
            for l in p_info:
                B[l[0]][l_count - 1] = 1
                B[l[1]][l_count - 1] = 1
                l_count += 1
            # end of rebuilding block #####################################################################

            _p_feasibility_on_step_k(k)

            eigen_data[k] = sp.sparse.linalg.eigsh(P_p, 2, which='LM')
            SLEM_per_k[k] = eigen_data[k][0][0]
            print(SLEM_per_k[k])

            #saving SLEM_per_k every step in case its needed for checkpoint to a csv
            csv_path_SLEM_per_k = experiment_graph_dir / f'SLEM_per_k.csv'
            with open(csv_path_SLEM_per_k, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for key, value in SLEM_per_k.items():
                    writer.writerow([str(key), str(abs(value))])

            #checkpoint: save P_p as weighted_edgelist every few steps (make sure it is a multiple of while loop condition)
            if k % 1 == 0:
                P_p_csv_path = experiment_graph_dir / f'subgradient_checkpoints/P_p_weighted_edgelist_step_{k}.csv'
                FMMCBias.save_transitionmatrix_as_weighted_edgelist_in_csv(P_p_csv_path,P_p)

            print(f"Step {k} done")
            logger.info(f'Finished subgradient computation step {k}')
            k += 1

        #get absolute of last value because that would usually happen at start of while loop
        SLEM_per_k[k-1] = abs(SLEM_per_k[k-1])

        return SLEM_per_k, P_p

    @staticmethod
    def build_nxgraph_from_nptransitionmatrix(transitionmatrix: np.ndarray) -> nx.Graph:
        """
        builds networkx graph from transitionmatrix

        Args:
            transitionmatrix (np.ndarray): transition matrix

        Returns:
            nx.Graph
        """
        n: int = transitionmatrix.shape[0]
        G: nx.Graph = nx.Graph()

        for i in range(n):
            for j in range(n):
                if transitionmatrix[i][j] != 0:
                    G.add_edge(i, j, weight=transitionmatrix[i][j])

        return G

    @staticmethod
    def remove_selfloops_from_nxgraph(graph: nx.Graph) -> nx.Graph:
        """
        removes selfloops from a networkx graph

        Args:
            graph (nx.Graph): networkx graph

        Returns:
            nx.Graph: networkx graph with selfloops removed
        """
        G: nx.Graph = graph
        edges: list[tuple[int, int]] = graph.edges()

        for edge in edges:
            if edge[0] == edge[1]:
                G.remove_edge(edge[0], edge[1])

        return G

    @staticmethod
    def get_weighted_edgelist_from_transitionmatrix(transitionmatrix: np.ndarray) -> list[tuple[int, int, float]]:
        """
        gets weighted edgelist from transitionmatrix

        Args:
            transitionmatrix (np.ndarray): transition matrix

        Returns:
            list[tuple[int, int, float]]: weighted edgelist
        """
        n: int = transitionmatrix.shape[0]
        weighted_edgelist: list[tuple[int, int, float]] = []

        for i in range(n):
            for j in range(n):
                if transitionmatrix[i][j] != 0:
                    weighted_edgelist.append((i, j, transitionmatrix[i][j]))

        return weighted_edgelist

    @staticmethod
    def get_weighted_decoded_edgelist(encoded_edgelist: list, id_to_node: dict) -> list[tuple[int, int, float]]:
        """
        decode one hot encoded edgelist

        Args:
            encoded_edgelist (list[tuple[int, int, float]]): edgelist
            id_to_node (dict[str, int]): id_to_node from one hot encoding

        Returns:
            list[tuple[int, int, float]]: weighted decoded edgelist
        """
        decoded_weighted_edgelist: list[tuple[int, int, float]] = []

        for u_id, v_id, w in encoded_edgelist:
            u = id_to_node[u_id]
            v = id_to_node[v_id]
            decoded_weighted_edgelist.append((u, v, w))

        return decoded_weighted_edgelist

    @staticmethod
    def build_nxgraph_from_weighted_edgelist(weighted_edgelist: list[tuple[int, int, float]]) -> nx.Graph:
        """
        build a networkx graph from a weighted edgelist

        Args:
            weighted_edgelist (list[tuple[int, int, float]]): weighted edgelist

        Returns:
            nx.Graph: networkx graph
        """
        G = nx.Graph()

        for u, v, w in weighted_edgelist:
            G.add_edge(u, v, weight=w)

        return G

    @staticmethod
    def save_transitionmatrix_as_weighted_edgelist_in_csv(path: Path, transitionmatrix: np.ndarray) -> None:
        """
        save the transition matrix as weighted edgelist in a csv file

        Args:
            path (Path): path to save the transition matrix
            transitionmatrix (np.ndarray): transition matrix to save

        Returns:
            None
        """
        weighted_edgelist: list[tuple[int, int, float]] = FMMCBias.get_weighted_edgelist_from_transitionmatrix(transitionmatrix)

        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for u, v, weight in weighted_edgelist:
                writer.writerow([str(u), str(v), str(weight)])

    @staticmethod
    def get_transitionmatrix_from_weighted_edgelist_checkpoint(path: Path, n: int) -> np.ndarray:
        """
        create a transition matrix from a weighted edgelist

        Args:
            path (Path): path to the csv with the weighted edgelist

        Returns:
            np.ndarray: transition matrix
        """
        transitionmatrix: np.ndarray = np.zeros((n,n))

        with open(path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                transitionmatrix[int(row[0])][int(row[1])] = float(row[2])
                transitionmatrix[int(row[1])][int(row[0])] = float(row[2])

        return transitionmatrix

    """
        # Create mappings
        node_to_id, id_to_node = create_mappings(g)
        # One-hot encode the edges
        one_hot_edges = one_hot_encode_edges(g, node_to_id)
        # Decode the one-hot encoded edges back to the original form
        decoded_edges = decode_edges(one_hot_edges, id_to_node)
    """
    @staticmethod
    def create_mappings(graph: nx.Graph):
        """
        Create forward (node_to_id) and backward (id_to_node) mappings for one-hot encoding of nodes in a graph.

        Args:
            graph (nx.Graph): The input NetworkX graph.

        Returns:
            tuple: (node_to_id, id_to_node) mappings.
        """
        # Get all unique nodes in the graph and sort them for consistency
        nodes = list(graph.nodes())
        nodes_sorted = sorted(nodes)

        # Create the forward mapping (node -> index)
        node_to_id = {node: idx for idx, node in enumerate(nodes_sorted)}

        # Create the backward mapping (index -> node)
        id_to_node = {idx: node for idx, node in enumerate(nodes_sorted)}

        # Debug prints to check the mappings
        print("Forward mapping (node to id):")
        i = 0
        for node, idx in node_to_id.items():
            i += 1
            if i < 20:
                print(f"Node {node} -> ID {idx}")

        print("\nBackward mapping (id to node):")
        i = 0
        for idx, node in id_to_node.items():
            i += 1
            if i < 20:
                print(f"ID {idx} -> Node {node}")

        return node_to_id, id_to_node

    @staticmethod
    def one_hot_encode_edges(graph: nx.Graph, node_to_id: dict):
        """
        One-hot encode the edges of the graph using the forward mapping.

        Args:
            graph (nx.Graph): The input NetworkX graph.
            node_to_id (dict): The forward mapping (node to index).

        Returns:
            list: One-hot encoded edge list.
        """
        one_hot_edges = []
        i = 0
        for u, v in graph.edges():
            u_id = node_to_id[u]
            v_id = node_to_id[v]
            one_hot_edges.append((u_id, v_id))

            i += 1
            if i < 20:
                # Debug print to check the encoding
                print(f"Edge ({u}, {v}) -> ({u_id}, {v_id})")

        return one_hot_edges

    @staticmethod
    def decode_edges(one_hot_edges: list, id_to_node: dict):
        """
        Decode one-hot encoded edges back to their original node pairs.

        Args:
            one_hot_edges (list): List of one-hot encoded edges (pairs of node indices).
            id_to_node (dict): The backward mapping (index to node).

        Returns:
            list: Original edge list with node pairs.
        """
        decoded_edges = []
        i = 0
        for u_id, v_id in one_hot_edges:
            u = id_to_node[u_id]
            v = id_to_node[v_id]
            decoded_edges.append((u, v))

            i += 1
            if i < 20:
                # Debug print to check the decoding
                print(f"One-hot edge ({u_id}, {v_id}) -> ({u}, {v})")

        return decoded_edges

    @staticmethod
    def create_directory(path):
        """
        create directory at path

        Args:
            path (Path): Path where to create directory.

        Returns:
            None
        """
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            raise

    @staticmethod
    def smooth_matrix(transitionmatrix: np.ndarray) -> np.ndarray:
        """
        smooth a matrix like: https://de.mathworks.com/matlabcentral/answers/172633-eig-doesn-t-converge-can-you-explain-why

        Args:
            transitionmatrix (np.ndarray): Matrix to smooth.

        Returns:
            np.ndarray: Smooth matrix.
        """
        n = transitionmatrix.shape[0]
        nA = np.linalg.norm(transitionmatrix, 'fro')
        nA = nA / nA.size
        nA = 0.00001 * nA

        As = transitionmatrix
        As_sq = np.square(As)

        for i in range(n):
            for j in range(n):
                if As_sq[i][j] < nA:
                    As[i][j] = 0

        return As
