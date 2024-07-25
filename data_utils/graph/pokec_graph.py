from __future__ import annotations
from typing import List, Final
import logging

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib

from pathlib import Path
from data_utils.graph.graph import Graph

# Configure logging
logger = logging.getLogger(__name__)

class PokecGraph(Graph):
    """
    A class to handle Pokec-specific graph operations.
    """
    def __init__(self, graph: Graph, attributes: pd.DataFrame):
        super().__init__(graph, attributes)

    @staticmethod
    def graph_from_nxgraph(g: nx.graph, df_attributes: pd.DataFrame):
        """
        Create a PokecGraph object from a NetworkX graph and attributes dataframe.
        
        Args:
            g (nx.Graph): NetworkX graph object.
            df_attributes (pd.DataFrame): DataFrame containing node attributes.
        
        Returns:
            PokecGraph: A new PokecGraph object.
        """
        return PokecGraph(g, df_attributes)
    

    @staticmethod
    def graph_from_edgelist(edgelist_path: str, 
                            attributes_path:str) -> PokecGraph:
        """
        Create a PokecGraph object from edgelist and attributes file paths.
        
        Args:
            edgelist_path (str): Path to the edgelist file.
            attributes_path (str): Path to the attributes file.
        
        Returns:
            PokecGraph: A new PokecGraph object.
        """
        logger.info(f"Loading edgelist from {edgelist_path}")
        logger.info(f"Loading attributes from {attributes_path}")
        
        df_edgelist = pd.read_csv(edgelist_path, sep=" ", header=None)

        logger.info(f"Loading attributes from {attributes_path}")
        df_attributes = pd.read_csv(attributes_path)
        g = nx.Graph()
        for _, row in df_edgelist.iterrows():
            g.add_edge(row.iloc[0], row.iloc[1])
            g.add_edge(row.iloc[1], row.iloc[0])
        
        nx.set_edge_attributes(g, values=1, name='weight')
        g = g.to_directed()

        return PokecGraph(g, df_attributes)

    @staticmethod
    def graph_visual_analyze(df_el: pd.DataFrame, 
                             df_attr: pd.DataFrame,
                             report_dir: Path, 
                             attributes: List[str],
                             exp_graph_name: str) -> ig.Graph:
        """
        Perform visual analysis of the graph and save plots and specifications.
        
        Args:
            df_el (pd.DataFrame): DataFrame containing the edgelist.
            df_attr (pd.DataFrame): DataFrame containing node attributes.
            report_dir (Path): Directory to save reports.
            attributes (List[str]): List of attributes for visualization.
            exp_graph_name (str): Name of the experiment graph.
        
        Returns:
            ig.Graph: An igraph Graph object.
        """
        graph_plot_dir = report_dir / 'graph_plots'
        graph_specs_dir = report_dir / 'graph_specs'
        graph_plot_dir.mkdir(parents=True, exist_ok=True)
        graph_specs_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating igraph from edgelist")
        ig_graph = PokecGraph.generate_igraph(df_el=df_el)

        logger.info(f"Visualizing graph and saving plots")
        PokecGraph.visualize(graph=ig_graph, df_attr=df_attr, plot_dir = graph_plot_dir, attributes=attributes, exp_graph_name=exp_graph_name)

        logger.info(f"Counting group connections")
        df_count = PokecGraph.count_group_connections(graph=ig_graph, df_attr=df_attr, attributes=attributes)

        logger.info(f"Graph stats: Number of nodes: {ig_graph.vcount()}, Number of edges: {ig_graph.ecount()}")

        df_count.to_csv(graph_specs_dir / f'{exp_graph_name.replace(" ", "_")}_group_connect.csv')
        
        return ig_graph

    @staticmethod
    def generate_igraph(df_el):
        """
        Generate an igraph Graph object from an edgelist DataFrame.
        
        Args:
            df_el (pd.DataFrame): DataFrame containing the edgelist.
        
        Returns:
            ig.Graph: An igraph Graph object.
        """
        edges = [tuple(row) for _, row in df_el.iterrows()]
        weight = [1]
        graph = ig.Graph.TupleList(edges, directed=False, weights=weight)
        graph.simplify(multiple=True, loops=False, combine_edges=None)

        return graph

    @staticmethod
    def visualize(graph: ig.Graph, 
                  df_attr: pd.DataFrame, 
                  plot_dir: Path, 
                  attributes: List[str], 
                  exp_graph_name: str) -> None:
        """
        Visualize the graph with different attribute colorings and save the plots.
        
        Args:
            graph (ig.Graph): igraph Graph object.
            df_attr (pd.DataFrame): DataFrame containing node attributes.
            plot_dir (Path): Directory to save plots.
            attributes (List[str]): List of attributes for visualization.
            exp_graph_name (str): Name of the experiment graph.
        """
        visual_style = {
            "bbox": (8000, 4500),
            "margin": 100,
            "vertex_size": 30,
            "vertex_label_size": 0,
            "edge_curved": False,
            "edge_width": 0.5
        }

        layout = graph.layout_drl()
        alpha = 0.5  # Adjust this value for desired opacity

        # 4 COLORS
        # color_names =['#d7191c','#fdae61','#abd9e9','#2c7bb6']
        # color_names = ['#e66101','#fdb863','#b2abd2','#5e3c99']
        color_names = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta',
                       'ivory', 'indigo', 'salmon', 'plum', 'orchid', 'brown', 'grey', 'violet',
                       'tan', 'olive', 'maroon', 'lime', 'lavender', 'gold', 'fuchsia', 'crimson',
                       'coral', 'chocolate', 'chartreuse', 'azure', 'aquamarine', 'aqua', 'white']
        
        # Convert color names to RGBA
        color_rgba = [(*matplotlib.colors.to_rgb(name), alpha) for name in color_names]

        # Now use color_rgba in your plotting code
        for attribute in attributes:
            for v in graph.vs:
                node = v['name']
                if node in df_attr.index:
                    v['color'] = color_rgba[df_attr.loc[node, 'label_' + attribute]]
                

            visual_style["layout"] = layout
            
            i=0
            plot_path_color = plot_dir / f'{exp_graph_name.replace(" ", "_")}_graphplot_{i}_color_{attribute}.png'
            # Create new files to handle multiple plots, they are non deterministic
            while plot_path_color.exists():
                i += 1
                plot_path_color = plot_dir / f'{exp_graph_name.replace(" ", "_")}_graphplot_{i}_color_{attribute}.png'

            ig.plot(graph, plot_path_color, **visual_style)
            logger.info(f"Graph plot saved to {plot_path_color}")

    def count_group_connections(graph: ig.Graph, df_attr: pd.DataFrame, attributes:List[str]) -> pd.DataFrame:
        """
        Count the group connections in the graph and generate a DataFrame with the results.
        
        Args:
            graph (ig.Graph): igraph Graph object.
            df_attr (pd.DataFrame): DataFrame containing node attributes.
            attributes (List[str]): List of attributes for counting group connections.
        
        Returns:
            pd.DataFrame: DataFrame with group connection statistics.
        """
        labels = [f'label_{attribute}' for attribute in attributes]
        max_n_classes = max(len(df_attr[labels[0]].unique()),
                            len(df_attr[labels[1]].unique()))
        density = 2 * graph.ecount() / (graph.vcount() * (graph.vcount() - 1))

        cnt_result_dict = {}


        for col in labels:
            n_classes = len(df_attr[col].unique())
            cnt_list = [[0] for cnt in range(n_classes)]
            for v in graph.vs:
                node = v['name']
                if node not in df_attr.index:
                    #print('node not in df_attr.index:', node)
                    continue
                nei = graph.neighborhood(v)
                neis = np.array([df_attr.loc[u, col] for u in graph.vs[nei]['name'] if u in df_attr.index])
                if neis.size == 0:
                    raise Exception('solitary node:', v)
                if np.all(neis == df_attr.loc[node, col]):
                    cnt_list[0][0] += 1
                else:
                    for c in range(n_classes):
                        if df_attr.loc[node, col] == c:
                            cnt_list[c][0] += 1
                            break
            n_groups = 25# number of possible connections
            desc_dict = {i: f'connected innergroup and to {i} other groups' for i in range(n_groups)}
            cnt_result_dict[f'{col}_group_connections'] = [f'{cnt[0]} ({desc_dict[i]})' for i, cnt in enumerate(cnt_list)]
            cnt_result_dict[f'{col}_abs_connect_to_groups'] = [_[0] for _ in cnt_list]
            cnt_result_dict[f'{col}_rel_connect_to_groups'] = [_[0]/graph.vcount() for _ in cnt_list]

        # Some Metrics, add more if needed
        cnt_result_dict['density'] = density
        cnt_result_dict['n_nodes'] = graph.vcount()
        cnt_result_dict['m_edges'] = graph.ecount()
        cnt_result_dict['n_classes_attr'] = len(df_attr[labels[0]].unique())
        cnt_result_dict["nodes per attr abs"] = [len(df_attr[df_attr[labels[0]] == i]) for i in range(max_n_classes)]
        cnt_result_dict["nodes per attr rel"] = [len(df_attr[df_attr[labels[0]] == i])/graph.vcount() for i in range(max_n_classes)]
        cnt_result_dict['n_classes_sens'] = len(df_attr[labels[1]].unique())
        cnt_result_dict["nodes per sens abs"] = [len(df_attr[df_attr[labels[1]] == i]) for i in range(max_n_classes)]
        cnt_result_dict["nodes per sens rel"] = [len(df_attr[df_attr[labels[1]] == i])/graph.vcount() for i in range(max_n_classes)]
        cnt_result_dict['diameter'] = graph.diameter()
        cnt_result_dict['radius'] = graph.radius()
        cnt_result_dict['avg_path_length'] = graph.average_path_length()
        cnt_result_dict['clique_number'] = graph.clique_number()

        for key, value in cnt_result_dict.items():
            if type(value) == list and len(value) < max_n_classes:
                    diff = max_n_classes - len(value)
                    if isinstance(value[0], str):
                        cnt_result_dict[key] +=[""] * diff
                    else:  
                        cnt_result_dict[key] +=[np.NaN] * diff

        df_cnt = pd.DataFrame.from_dict(cnt_result_dict)
    
        return df_cnt