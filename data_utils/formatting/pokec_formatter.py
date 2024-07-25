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
    def pre_formatting(attribute_filepath: str, formatted_data_dir: Path, filter_category: str, 
                        column_names: List[str], attributes: List[str]) -> pd.DataFrame:
        """
        Pre-formats the raw attribute data by filtering and saving to CSV files.

        Args:
            attribute_filepath (str): Path to the raw attribute file.
            formatted_data_dir (Path): Directory to save formatted data.
            filter_category (str): Category to filter by.
            column_names (List[str]): Column names for the dataframe.
            attributes (List[str]): List of attributes to retain.

        Returns:
            pd.DataFrame: Pre-formatted dataframe.
        """
        logger.info('Starting pre-formatting')

        df = PokecFormatter.format_textfile(attribute_filepath, column_names, attributes)
        logger.info(f'Dataframe after formatting text file:\n{df.head()}')

        count_df = PokecFormatter.create_count_csv(df, filter_category)
        logger.info(f'Count dataframe:\n{count_df.head()}')

        count_df.to_csv(f"{formatted_data_dir}/pre_formatted_countby_{filter_category}.csv", index=False)
        df.to_csv(f"{formatted_data_dir}/pre_formatted_pokec_attributes.csv", index=False)

        logger.info('Pre-formatting done.')
        return df
    
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
        df_attr = pd.read_csv(formatted_data_dir / "pre_formatted_pokec_attributes.csv")

        # Filter the dataframe based on the specified filter category (region) and category values
        df_attr = PokecFormatter.filter_by_category(df_attr, filter_category, category_values, attributes)
        # Select only the rows where the value in the 'AGE' column is between 16 and 99 (inclusive)
        df_attr = df_attr.query("16 <= AGE <= 99")
        df_attr = PokecFormatter.label_values(df_attr, attributes, 'AGE', splitpoints_bins)

        df_el, df_attr = PokecFormatter.filter_edgelist_attributes(edgelist_filepath, df_attr)
        
        filtered_attr_filepath = experiment_path / "filtered_attributes.csv"
        filtered_el_filepath = experiment_path / "filtered_edgelist.txt"

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
    def create_count_csv(df: pd.DataFrame, filter_category: str):
        """
        Creates a count CSV for the specified filter category displaying the numbr of nodes in each category.

        Args:
            df (pd.DataFrame): Dataframe to process.
            filter_category (str): Category to filter by.

        Returns:
            pd.DataFrame: Count dataframe with proportions.
        """
        logger.info('Creating count CSV')

        count_df = df[filter_category].value_counts().reset_index()
        count_df.columns = [f"{filter_category}_name", "count"]
        count_df["proportion"] = count_df["count"].astype(float) / count_df["count"].sum()
        count_df = count_df.sort_values(by="count", ascending=False)

        return count_df

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




