'''
./data_utils/formatting/pokec_formatter.py
Description: Formatter class for handling Pokec dataset-specific formatting tasks.
'''

import pandas as pd
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
                       formatted_data_dir: Path, 
                       filter_category: str, 
                       column_names: List[str],
                       attributes: List[str]) -> pd.DataFrame:
        """
        Pre-formats the raw attribute data by reading the text file,
        performing initial cleaning, and saving outputs.
        """
        logger.info("Starting pre-formatting")

        fp = Path(attribute_filepath)
        df = PokecFormatter.format_textfile(fp, column_names, attributes)
        logger.info(f"Dataframe after formatting text file:\n{df.head()}")

        count_df = PokecFormatter.create_count_csv(df, filter_category)
        logger.info(f"Count dataframe:\n{count_df.head()}")

        # Save the pre-formatted files.
        count_df.to_csv(formatted_data_dir / f"pre_formatted_countby_{filter_category}.csv", index=False)
        df.to_csv(formatted_data_dir / "pre_formatted_pokec_attributes.csv", index=False)

        logger.info("Pre-formatting done.")
        return df

    @staticmethod
    def formatting(formatted_data_dir: str, edgelist_filepath: str, experiment_path: Path, 
                   filter_category: str, category_values: List[str],
                   attributes: List[str], splitpoints_bins: List[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Formats the data by:
         - Loading the pre-formatted attributes.
         - Optionally filtering by region if category_values is non-empty.
         - Filtering by age and labeling values.
         - Filtering the edgelist to keep only nodes that have attributes.

        If category_values is empty, region filtering is skipped (full-graph mode).
        """
        logger.info("Starting formatting")
        preformatted_path = Path(formatted_data_dir) / "pre_formatted_pokec_attributes.csv"
        df_attr = pd.read_csv(preformatted_path)

        # If category_values is provided (non-empty), apply region filtering.
        if category_values:
            df_attr = PokecFormatter.filter_by_category(df_attr, filter_category, category_values, attributes)
        else:
            logger.info("No region filtering applied (full graph mode).")

        # Filter the dataframe by age.
        df_attr = df_attr.query("16 <= AGE <= 99")
        df_attr = PokecFormatter.label_values(df_attr, attributes, 'AGE', splitpoints_bins)

        # Filter the edgelist and adjust attributes to include only nodes present in the edgelist.
        df_el, df_attr = PokecFormatter.filter_edgelist_attributes(Path(edgelist_filepath), df_attr)
        
        # Write out the filtered data for this experiment.
        filtered_attr_filepath = experiment_path / "filtered_attributes.csv"
        filtered_el_filepath = experiment_path / "filtered_edgelist.txt"
        df_attr.to_csv(filtered_attr_filepath)
        df_el.to_csv(filtered_el_filepath, sep=" ", index=False, header=False)

        logger.info("Formatting done.")
        return df_el, df_attr

    @staticmethod
    def format_textfile(filepath: Path, column_names: List[str], attributes: List[str]) -> pd.DataFrame:
        """
        Reads and formats the raw attribute text file into a structured dataframe.
        """
        logger.info("Reading raw attribute file...")
        df = pd.read_csv(filepath, sep="\t", header=None, usecols=[0, 1, 3, 4, 7, 8])
        logger.info(f"Initial dataframe:\n{df.head()}")

        df.columns = column_names
        df = df.dropna(subset=attributes)
        logger.info(f"Dataframe after dropping NaNs:\n{df.head()}")

        for col in df.columns:
            if col != 'region':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    logger.info(f"Could not convert column {col} to numeric: {e}")
        
        logger.info(f"Dataframe after conversion:\n{df.head()}")
        return df

    @staticmethod
    def create_count_csv(df: pd.DataFrame, filter_category: str) -> pd.DataFrame:
        """
        Creates a count CSV for the specified filter category.
        """
        logger.info("Creating count CSV...")
        count_df = df[filter_category].value_counts().reset_index()
        count_df.columns = [f"{filter_category}_name", "count"]
        count_df["proportion"] = count_df["count"].astype(float) / count_df["count"].sum()
        count_df = count_df.sort_values(by="count", ascending=False)
        return count_df

    @staticmethod
    def filter_by_category(df: pd.DataFrame, filter_category: str, category_values: List[str], attributes: List[str]) -> pd.DataFrame:
        """
        Filters the dataframe based on the specified filter category and provided values.
        Assumes that category_values is non-empty.
        """
        logger.info("Filtering by category...")
        # This line performs indexing using the filter_category.
        df_filtered = df.set_index(filter_category).loc[category_values].reset_index().set_index("user_id")[attributes]
        return df_filtered

    @staticmethod
    def label_values(df: pd.DataFrame, columns: List[str], specific_column: str, splitpoint_bins: List[int]) -> pd.DataFrame:
        """
        Creates labeled columns based on bins (for the specific column) or unique values (for others).
        """
        logger.info("Labeling values...")
        df_copy = df.copy()
        for column in columns:
            if column == specific_column:
                df_copy['label_' + column] = pd.cut(df[column], splitpoint_bins, labels=range(len(splitpoint_bins)-1))
            else:
                value_labels = {val: label for label, val in enumerate(df[column].unique())}
                df_copy['label_' + column] = df[column].map(value_labels)
        return df_copy

    @staticmethod
    def filter_edgelist_attributes(edgelist_filepath: Path, df_attr: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filters the edgelist and attributes dataframes to retain only nodes that are present in both.
        """
        logger.info("Filtering edgelist attributes...")
        edgelist_df = pd.read_csv(edgelist_filepath, sep="\t", header=None)
        # Keep only edges where both nodes are in the attribute index.
        filtered_df_el = edgelist_df[edgelist_df[0].isin(df_attr.index) & edgelist_df[1].isin(df_attr.index)]
        # Keep only the rows in attributes corresponding to edges.
        filtered_df_attr = df_attr.loc[df_attr.index.isin(filtered_df_el[0]) | df_attr.index.isin(filtered_df_el[1])]
        return filtered_df_el, filtered_df_attr
