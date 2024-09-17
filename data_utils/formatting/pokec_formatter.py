import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple
from data_utils.formatting.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class PokecFormatter(DataFormatter):
    """
    Formatter class for handling Pokec dataset-specific formatting tasks.
    Inherits from the DataFormatter base class.
    """
    @staticmethod
    def pre_formatting(attribute_filepath: str, 
                       edgelist_filepath: str,
                       formatted_data_dir: Path, 
                       filter_category: str, 
                       column_names: List[str], 
                       attributes: List[str],
                       include_edge_counts: bool = False,
        ) -> pd.DataFrame:
        """
        Pre-formats the raw attribute data by filtering and saving to CSV files.

        Args:
            attribute_filepath (str): Path to the raw attribute file.
            edgelist_filepath (str): Path to the edgelist file.
            formatted_data_dir (Path): Directory to save formatted data.
            filter_category (str): Category to filter by.
            column_names (List[str]): Column names for the dataframe.
            attributes (List[str]): List of attributes to retain.
            include_edge_counts (bool): Whether to include edge counts in the output. Default is False.

        Returns:
            pd.DataFrame: Pre-formatted dataframe.
        """
        logger.info('Starting pre-formatting')

        attr_df = PokecFormatter.format_textfile(attribute_filepath, column_names, attributes)
        logger.info(f'Dataframe after formatting text file:\n{attr_df.head()}')

        edgelist_df = pd.read_csv(edgelist_filepath, sep="\t", header=None)

        nodecount_df, edgecount_df = PokecFormatter.create_count_csv(
            attr_df, edgelist_df, filter_category, include_edge_counts
        )
        logger.info(f'Count dataframe:\n{nodecount_df.head()}')
                    
        nodecount_df.to_csv(f"{formatted_data_dir}/pre_formatted_countby_{filter_category}_sort_node.csv", index=False)

        # If edge counts are included, save the CSV with both node and edge counts, sorted by edge count
        if include_edge_counts and edgecount_df is not None:
            logger.info(f'Edge count dataframe:\n{edgecount_df.head()}')
            edgecount_df.to_csv(f"{formatted_data_dir}/pre_formatted_countby_{filter_category}_sorted_by_edges.csv", index=False)

        attr_df.to_csv(f"{formatted_data_dir}/pre_formatted_pokec_attributes.csv", index=False)

        logger.info('Pre-formatting done.')
        return attr_df
    
    @staticmethod
    def formatting(formatted_data_dir: str, edgelist_filepath: str, experiment_path: Path, 
                    filter_category: str, category_values: List[str], 
                    attributes: List[str], 
                    splitpoints_bins: List[int] = None) -> tuple[pd.DataFrame]:
        """
        Formats the data by filtering attributes and edgelists based on specific criteria.

        Args:
            formatted_data_dir (Path): Directory of pre-formatted data.
            edgelist_filepath (Path): Path to the edgelist file.
            experiment_path (Path): Directory to save experiment data.
            filter_category (str): Category to filter by.
            category_values (List[str]): List of category values to filter.
            attributes (List[str]): List of attributes to retain.
            splitpoints_bins (List[int], optional): Bins for specific attribute splitting.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Filtered edgelist and attribute dataframes.
        """
        logger.info('Starting formatting')
        filtered_attr_filepath = experiment_path / "filtered_attributes.csv"
        filtered_el_filepath = experiment_path / "filtered_edgelist.txt"

        # Check if the formatted files already exist
        if filtered_attr_filepath.exists() and filtered_el_filepath.exists():
            logger.info(f'Formatted files already exist at {experiment_path}. Skipping formatting.')
            df_attr = pd.read_csv(filtered_attr_filepath)
            df_el = pd.read_csv(filtered_el_filepath, sep=" ", header=None)
            return df_el, df_attr

        df_attr = pd.read_csv(formatted_data_dir / "pre_formatted_pokec_attributes.csv")

        # Filter the dataframe based on the specified filter category (region) and category values
        df_attr = PokecFormatter.filter_by_category(df_attr, filter_category, category_values, attributes)
        # Select only the rows where the value in the 'AGE' column is between 16 and 99 (inclusive)
        df_attr = df_attr.query("16 <= AGE <= 99")
        df_attr = PokecFormatter.label_values(df_attr, attributes, 'AGE', splitpoints_bins)

        df_el, df_attr = PokecFormatter.filter_edgelist_attributes(edgelist_filepath, df_attr)
        
        df_attr.to_csv(filtered_attr_filepath) 
        df_el.to_csv(filtered_el_filepath, sep=" ", index=None, header=None)

        logger.info('Formatting done.')
        return df_el, df_attr

    @staticmethod
    def format_textfile(filepath: Path, column_names: List[str], attributes: List[str]) -> pd.DataFrame:
        """
        Formats the text file into a structured dataframe with specified columns and attributes.

        Args:
            filepath (Path): Path to the text file.
            column_names (List[str]): List of column names.
            attributes (List[str]): List of attributes to retain.

        Returns:
            pd.DataFrame: Formatted dataframe.
        """
        df = pd.read_csv(filepath, sep="\t", header=None, usecols=[0, 1, 3, 4, 7, 8])
        logger.info(f'Initial dataframe read from file:\n{df.head()}')

        df.columns = column_names
        df = df.dropna(subset=attributes)
        logger.info(f'Dataframe after dropping NaNs:\n{df.head()}')

        for col in df.columns:
            if col != 'region':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except ValueError:
                    logger.info(f'Column {col} could not be converted to numeric and will be left as is.')
        
        logger.info(f'Dataframe after converting to numeric:\n{df.head()}')
        return df
    
    @staticmethod
    def create_count_csv(df: pd.DataFrame, 
                        edgelist_df: pd.DataFrame, 
                        filter_category: str, 
                        include_edge_counts: bool = False):
        """
        Creates a count CSV for the specified filter category displaying the number of nodes,
        edges connected to each category, and optionally, connections to other classes.

        Args:
            df (pd.DataFrame): Dataframe to process (attributes dataframe).
            edgelist_df (pd.DataFrame): Edgelist dataframe with two columns representing node pairs.
            filter_category (str): Category to filter by.
            include_edge_counts (bool): Whether to include edge counts and class connections in the output. Default is False.

        Returns:
            pd.DataFrame: Node count dataframe.
            pd.DataFrame: Edge count dataframe (only if include_edge_counts is True).
        """
        logger.info(f'Creating count CSV for nodes in each {filter_category} class')

        # Count nodes for each class in filter_category
        node_count_df = df[filter_category].value_counts().reset_index()
        node_count_df.columns = [f"{filter_category}_name", "node_count"]
        node_count_df["node_proportion"] = node_count_df["node_count"].astype(float) / node_count_df["node_count"].sum()
        node_count_df = node_count_df.sort_values(by="node_count", ascending=False)

        if include_edge_counts:
            logger.info(f'Including edge counts and connections to other classes in the CSV for {filter_category}')

            # Convert edgelist_df to undirected by dropping duplicate edges (considering both directions)
            undirected_edges = pd.DataFrame(
                np.sort(edgelist_df[[0, 1]].values, axis=1)
            ).drop_duplicates().reset_index(drop=True)
            
            # Mapping from node IDs to their class
            node_to_class = df[filter_category].to_dict()

            # Initialize a dictionary to store the new columns
            new_columns = {
                "edge_count": np.zeros(len(node_count_df), dtype=int),
                "internal_edges": np.zeros(len(node_count_df), dtype=int),  # To store edges inside the class
                "neighbor_count": np.zeros(len(node_count_df), dtype=int),  # To store the number of neighbors
            }
            for class_name in node_count_df[f"{filter_category}_name"]:
                new_columns[f'connections_to_{class_name}'] = np.zeros(len(node_count_df), dtype=int)

            # To store the unique neighbors of each class
            neighbors_per_class = {class_name: set() for class_name in node_count_df[f"{filter_category}_name"]}


            # Count edges connected to each class and between classes
            for _, (node_a, node_b) in undirected_edges.iterrows():
                class_a = node_to_class.get(node_a)
                class_b = node_to_class.get(node_b)
                if class_a is not None and class_b is not None:
                    idx_a = node_count_df[node_count_df[f"{filter_category}_name"] == class_a].index[0]
                    idx_b = node_count_df[node_count_df[f"{filter_category}_name"] == class_b].index[0]

                    new_columns["edge_count"][idx_a] += 1
                    new_columns["edge_count"][idx_a] += 1

                    if class_a == class_b:
                        new_columns["internal_edges"][idx_a] += 1  # Count internal edges (within the same class)
                    else:
                        # Add neighbors to the sets
                        neighbors_per_class[class_a].add(node_b)
                        neighbors_per_class[class_b].add(node_a)
                        new_columns[f'connections_to_{class_b}'][idx_a] += 1
                        new_columns[f'connections_to_{class_a}'][idx_b] += 1

            # Calculate neighbor counts based on unique neighbor sets
            for class_name, neighbors in neighbors_per_class.items():
                idx = node_count_df[node_count_df[f"{filter_category}_name"] == class_name].index[0]
                new_columns["neighbor_count"][idx] = len(neighbors)

            # Add new columns to the DataFrame
            new_columns_df = pd.DataFrame(new_columns, index=node_count_df.index)
            node_count_df = pd.concat([node_count_df, new_columns_df], axis=1)

            node_count_df["edge_proportion"] = node_count_df["edge_count"].astype(float) / node_count_df["edge_count"].sum()

            # Sort by edge count
            edge_count_df = node_count_df.sort_values(by="edge_count", ascending=False)

            return node_count_df, edge_count_df

        return node_count_df, None


    # @staticmethod
    # def create_count_csv(df: pd.DataFrame, 
    #                      edgelist_df: pd.DataFrame, 
    #                      filter_category: str, 
    #                      include_edge_counts: bool = False
    #     ):
    #     """
    #     Creates a count CSV for the specified filter category displaying the number of nodes
    #     and optionally the number of edges connected to each category.

    #     Args:
    #         df (pd.DataFrame): Dataframe to process (attributes dataframe).
    #         edgelist_df (pd.DataFrame): Edgelist dataframe with two columns representing node pairs.
    #         filter_category (str): Category to filter by.
    #         include_edge_counts (bool): Whether to include edge counts in the output. Default is False.

    #     Returns:
    #         pd.DataFrame: Node count dataframe.
    #         pd.DataFrame: Edge count dataframe (only if include_edge_counts is True).
    #     """
    #     logger.info(f'Creating count CSV for nodes in each {filter_category} class')

    #     # Count nodes for each class in filter_category
    #     node_count_df = df[filter_category].value_counts().reset_index()
    #     node_count_df.columns = [f"{filter_category}_name", "node_count"]
    #     node_count_df["node_proportion"] = node_count_df["node_count"].astype(float) / node_count_df["node_count"].sum()

    #     node_count_df = node_count_df.sort_values(by="node_count", ascending=False)

    #     if include_edge_counts:
    #         logger.info(f'Including edge counts in the CSV for {filter_category}')

    #         # Convert edgelist_df to undirected by dropping duplicate edges (considering both directions)
    #         undirected_edges = pd.DataFrame(
    #             np.sort(edgelist_df[[0, 1]].values, axis=1)
    #         ).drop_duplicates().reset_index(drop=True)
        
    #         # Count edges connected to each class in filter_category
    #         def count_edges_connected_to_class(class_name):
    #             # Filter for nodes that belong to the current class
    #             nodes_in_class = df[df[filter_category] == class_name].index
    #             # Count edges where at least one node is within the class
    #             edges_connected = undirected_edges[(undirected_edges[0].isin(nodes_in_class)) | (undirected_edges[1].isin(nodes_in_class))]
    #             return len(edges_connected)

    #         node_count_df["edge_count"] = node_count_df[f"{filter_category}_name"].apply(count_edges_within_class)
    #         node_count_df["edge_proportion"] = node_count_df["edge_count"].astype(float) / node_count_df["edge_count"].sum()

    #         # Sort by edge count
    #         edge_count_df = node_count_df.sort_values(by="edge_count", ascending=False)

    #         return node_count_df, edge_count_df

    #     return node_count_df, None

    @staticmethod
    def filter_by_category(df: pd.DataFrame, filter_category: str, category_values: List[str], attributes: List[str]) -> pd.DataFrame:
        """
        Filters the dataframe based on the specified category and values.

        Args:
            df (pd.DataFrame): Dataframe to filter.
            filter_category (str): Category to filter by.
            category_values (List[str]): List of category values to filter.
            attributes (List[str]): List of attributes to retain.

        Returns:
            pd.DataFrame: Filtered dataframe.
        """
        logger.info('Filtering by category')

        df = df.set_index(filter_category).loc[category_values].reset_index().set_index("user_id")[attributes]
        
        return df

    @staticmethod
    def label_values(df, columns, specific_column, splitpoint_bins):
        """
        Labels the values in the dataframe based on specified columns and bins.

        Args:
            df (pd.DataFrame): Dataframe to label.
            columns (List[str]): Columns to label.
            specific_column (str): Specific column for binning (for example 'AGE').
            splitpoint_bins (List[int]): Bins for the specific column.

        Returns:
            pd.DataFrame: Dataframe with labeled values.
        """
        logger.info('Labeling values')

        df_copy = df.copy()
        for column in columns:
            # If the column is the specific column create bins for the values based on splitpoint bins
            if column == specific_column:
                df_copy['label_' + column] = pd.cut(df[column], splitpoint_bins, labels=range(len(splitpoint_bins)-1))
            else:
                # Create a dictionary with the unique values of the column as keys and the values as the labels
                value_labels = {val: label for label, val in enumerate(df[column].unique())}
                df_copy['label_' + column] = df[column].map(value_labels)

        return df_copy
   
    @staticmethod
    def filter_edgelist_attributes(edgelist_filepath: Path, df_attr: pd.DataFrame):
        """
        Filters the edgelist and attributes dataframes to retain relevant rows, st each node is connected and has attributes.

        Args:
            edgelist_filepath (Path): Path to the edgelist file.
            df_attr (pd.DataFrame): Attributes dataframe.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Filtered edgelist and attributes dataframes.
        """
        logger.info('Filtering edgelist attributes')

        edgelist_df = pd.read_csv(edgelist_filepath, sep="\t", header=None)
        # Filter the edgelist dataframe based on the attributes
        filtered_df_el = edgelist_df[edgelist_df[0].isin(df_attr.index) & edgelist_df[1].isin(df_attr.index)]
        # Filter the attributes dataframe based on the edgelist
        filtered_df_attr = df_attr.loc[(df_attr.index.isin(filtered_df_el[1])) | (df_attr.index.isin(filtered_df_el[0]))]

        return filtered_df_el, filtered_df_attr




