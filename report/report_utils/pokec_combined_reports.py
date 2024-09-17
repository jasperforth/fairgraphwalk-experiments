import logging
from typing import List
from pathlib import Path

import pandas as pd
import numpy as np


class PokecCombinedReports:
    """
    A class to process and combine confusion matrix reports from experiments 
    involving crosswalk and baseline models, with support for multiple graph structures.
    
    Attributes:
    ----------
    report_dir : Path
        The directory containing the confusion reports.
    confusion_dir : Path
        The directory where individual confusion reports are stored.
    combined_results_dir : Path
        The directory where combined results will be saved.
    """
    def __init__(self, report_dir: Path):
        """
        Initializes the PokecCombinedReports with the specified report directory.

        Parameters:
        ----------
        report_dir : Path
            The directory containing the confusion reports.
        """
        self.report_dir = report_dir
        self.confusion_dir = self.report_dir / f'confusion_reports'
        self.combined_results_dir = self.report_dir / 'combined_results'
        self.combined_results_dir.mkdir(parents=True, exist_ok=True)


        # Set up logging for the class
        log_file = self.combined_results_dir / 'pokec_combined_reports.log'
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Initialized PokecCombinedReports")


    def run_averaging_df(self, experiment_name: str, graph_list: List[int]):
        """
        Process and combine confusion matrices for each experiment across specified graphs.

        Parameters:
        ----------
        experiment_name : str
            The name of the experiment to process (e.g., 'pokec semi' or 'pokec mixed').
        graph_list : List[int]
            The list of graph numbers to include in the processing.
        """
        logging.info(f"Running averaging for experiment '{experiment_name}' on graphs: {graph_list}")
        for exp_graph_dir in self.confusion_dir.iterdir():
            if exp_graph_dir.is_dir() and exp_graph_dir.name.startswith(experiment_name.replace(" ", "_")):
                graph_number = int(exp_graph_dir.name.split('_')[-1])  # Extract the graph number
                if graph_number in graph_list:
                    logging.info(f"Processing graph {graph_number}")
                    exp_graph_path = str(f"{experiment_name} {graph_number}").replace(" ", "_")
                    combined_df_cw = None
                    combined_df_baseline = None
                    combined_df_fmmc = None

                    
                    
                    # Process crosswalk results
                    for biasing_dir in exp_graph_dir.iterdir():
                        if biasing_dir.is_dir() and biasing_dir.name == "crosswalk": 
                            for exp_dir in biasing_dir.iterdir():
                                combined_df_cw = self.process_experiments_reports_cw(exp_dir, combined_df_cw)

                            # Save the combined crosswalk results into a single CSV file
                            if combined_df_cw is not None:
                                cw_output_path = self.combined_results_dir / f'{exp_graph_path}_results_cw.csv'
                                combined_df_cw.to_csv(cw_output_path, index=False)
                                logging.info(f"Saved crosswalk combined results to {cw_output_path}")

                        # Process baseline results
                        elif biasing_dir.is_dir() and biasing_dir.name == "baseline":
                            for exp_dir in biasing_dir.iterdir():
                                combined_df_baseline = self.process_experiments_reports_baseline(exp_dir, combined_df_baseline)
                            
                            # Save the combined baseline results into a single CSV file
                            if combined_df_baseline is not None:
                                baseline_output_path = self.combined_results_dir / f'{exp_graph_path}_results_baseline.csv'
                                combined_df_baseline.to_csv(baseline_output_path, index=False)
                                logging.info(f"Saved baseline combined results to {baseline_output_path}")

                        # Process fmmc results
                        elif biasing_dir.is_dir() and biasing_dir.name == "fmmc":
                            for exp_dir in biasing_dir.iterdir():
                                combined_df_fmmc = self.process_experiments_reports_baseline(exp_dir, combined_df_fmmc, FMMC=True)
                            
                            # Save the combined baseline results into a single CSV file
                            if combined_df_fmmc is not None:
                                fmmc_output_path = self.combined_results_dir / f'{exp_graph_path}_results_fmmc.csv'
                                combined_df_fmmc.to_csv(fmmc_output_path, index=False)
                                logging.info(f"Saved fmmc combined results to {fmmc_output_path}")


    def process_experiments_reports_cw(self, exp_dir: Path, combined_df: pd.DataFrame):
        """
        Processes individual crosswalk experiment reports and combines them into a single DataFrame.

        Parameters:
        ----------
        exp_dir : Path
            The directory containing individual experiment reports.
        combined_df : pd.DataFrame
            The DataFrame to which new data will be appended.

        Returns:
        -------
        pd.DataFrame
            The updated combined DataFrame with the new crosswalk experiment data.
        """
        if exp_dir.name.startswith("experiment_prewalk"):
            def parse_directory_name_cw(directory_name):
                # Parse directory name for experiment parameters
                parts = directory_name.split('_')
                alpha = float(parts[parts.index('alpha') + 1])
                exponent = float(parts[parts.index('exponent') + 1])
                p = float(parts[parts.index('p') + 1])
                q = float(parts[parts.index('q') + 1])
                sens_attr = '_'.join(parts[parts.index('sens') + 1:parts.index('other')])
                other_attr = '_'.join(parts[parts.index('other') + 1:])
                return alpha, exponent, p, q, sens_attr, other_attr

            alpha, exponent, p, q, sens_attr, other_attr = parse_directory_name_cw(exp_dir.name)
            logging.info(f"Processing crosswalk experiment: {exp_dir.name}")
            file_names = ['average_confusion_report_y.csv', 'average_confusion_report_z.csv']
            for csv_file in file_names:
                file_path = exp_dir / csv_file
                if file_path.exists():
                    # Load the CSV file into a DataFrame with multi-index
                    df = pd.read_csv(file_path, header=[0, 1])
                    unique_classes, unique_cond_classes = self.extract_unique_classes_and_cond_classes(df)

                    # Create column names for F1 scores and support metrics
                    f1_columns = [f'f1_macro_class_{uc}' for uc in unique_classes]
                    f1_cond_columns = [f'f1_macro_cond_{uc_cond}' for uc_cond in unique_cond_classes]
                    for uc in unique_classes:
                        f1_cond_columns += [f'f1_macro_class_{uc}_cond_{uc_cond}' for uc_cond in unique_cond_classes]

                    # Initialize the combined DataFrame if it's None
                    if combined_df is None:
                        combined_df = pd.DataFrame(columns=['alpha', 'exponent', 'p', 'q', 
                                                            'sens_attr', 'other_attr', 'attribute',
                                                            'f1_macro', 'accuracy', 'support',
                                                            *f1_columns, *f1_cond_columns])

                    metrics = self.calculate_metrics(df, unique_classes, unique_cond_classes)
                    attribute = 'other' if csv_file == 'average_confusion_report_y.csv' else 'sensitive'

                    # Create a new DataFrame with the current metrics
                    new_data = pd.DataFrame({
                        'alpha': [alpha],
                        'exponent': [exponent],
                        'p': [p],
                        'q': [q],
                        'sens_attr': [sens_attr],
                        'other_attr': [other_attr],
                        'attribute': [attribute],
                        **metrics
                    })

                    # Filter out any all-NA columns before concatenation
                    new_data = new_data.dropna(axis=1, how='all')

                    # Concatenate new data to the combined DataFrame
                    combined_df = pd.concat([combined_df, new_data], ignore_index=True)
                    logging.info(f"Metrics calculated and added for {csv_file} in {exp_dir.name}")
                else:
                    logging.warning(f"File {csv_file} not found in {exp_dir.name}")

        return combined_df
    

    def process_experiments_reports_baseline(self, exp_dir: Path, combined_df: pd.DataFrame, FMMC: bool=False):
        """
        Processes individual baseline experiment reports and combines them into a single DataFrame.

        Parameters:
        ----------
        exp_dir : Path
            The directory containing individual experiment reports.
        combined_df : pd.DataFrame
            The DataFrame to which new data will be appended.

        Returns:
        -------
        pd.DataFrame
            The updated combined DataFrame with the new baseline experiment data.
        """
        if exp_dir.name.startswith("experiment"):
            # Parse directory name for experiment parameters
            def parse_directory_name_baseline(directory_name, FMMC=False):
                parts = directory_name.split('_')
                p = float(parts[parts.index('p') + 1])
                q = float(parts[parts.index('q') + 1])
                if FMMC==True:
                    selfloops = str(parts[parts.index('selfloops') + 1])
                    return p, q, selfloops
                return p, q
            
            if FMMC==True:
                p, q, selfloops = parse_directory_name_baseline(exp_dir.name, FMMC)
            else:
                p, q = parse_directory_name_baseline(exp_dir.name)

            sens_attr = 'AGE'
            logging.info(f"Processing baseline experiment: {exp_dir.name}")
            file_names = ['average_confusion_report_y.csv', 'average_confusion_report_z.csv']
            for csv_file in file_names:
                file_path = exp_dir / csv_file
                if file_path.exists():
                    # Load the CSV file into a DataFrame with multi-index
                    df = pd.read_csv(file_path, header=[0, 1])
                    unique_classes, unique_cond_classes = self.extract_unique_classes_and_cond_classes(df)

                    # Create column names for F1 scores and support metrics
                    f1_columns = [f'f1_macro_class_{uc}' for uc in unique_classes]
                    f1_cond_columns = [f'f1_macro_cond_{uc_cond}' for uc_cond in unique_cond_classes]
                    for uc in unique_classes:
                        f1_cond_columns += [f'f1_macro_class_{uc}_cond_{uc_cond}' for uc_cond in unique_cond_classes]

                    # Initialize the combined DataFrame if it's None
                    if combined_df is None:
                        if FMMC==True:
                            combined_df = pd.DataFrame(columns=['p', 'q', 'selfloops',
                                                                'sens_attr', 'attribute',
                                                                'f1_macro', 'accuracy', 'support',
                                                                *f1_columns, *f1_cond_columns])
                            
                        else:
                            combined_df = pd.DataFrame(columns=['p', 'q', 
                                    'sens_attr', 'attribute',
                                    'f1_macro', 'accuracy', 'support',
                                    *f1_columns, *f1_cond_columns])

                    metrics = self.calculate_metrics(df, unique_classes, unique_cond_classes)
                    attribute = 'other' if csv_file == 'average_confusion_report_y.csv' else 'sensitive'

                    # Create a new DataFrame with the current metrics
                    if FMMC==True:
                        new_data = pd.DataFrame({
                            'p': [p],
                            'q': [q],
                            'selfloops': [selfloops],
                            'sens_attr': [sens_attr],
                            'attribute': [attribute],
                            **metrics
                        })
                    else:
                        new_data = pd.DataFrame({
                            'p': [p],
                            'q': [q],
                            'sens_attr': [sens_attr],
                            'attribute': [attribute],
                            **metrics
                        })

                    # Filter out any all-NA columns before concatenation
                    new_data = new_data.dropna(axis=1, how='all')

                    # Concatenate new data to the combined DataFrame
                    combined_df = pd.concat([combined_df, new_data], ignore_index=True)
                    logging.info(f"Metrics calculated and added for {csv_file} in {exp_dir.name}")
                else:
                    logging.warning(f"File {csv_file} not found in {exp_dir.name}")


        return combined_df
    

    @staticmethod
    def extract_unique_classes_and_cond_classes(df):
        """
        Extract unique classes and unique conditional classes from the DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame containing confusion matrix metrics.

        Returns:
        -------
        unique_classes : List[str]
            List of unique classes.
        unique_cond_classes : List[str]
            List of unique conditional classes.
        """
        unique_classes = df['cond_class']['metric'].unique()
        unique_classes = [x for x in unique_classes if x.isdigit()]
        unique_cond_classes = df.columns.get_level_values(0).unique()
        unique_cond_classes = [x for x in unique_cond_classes if x.isdigit()]
        
        return unique_classes, unique_cond_classes
    
    
    @staticmethod
    def calculate_metrics(df, unique_classes, unique_cond_classes):
        """
        Calculate metrics from the confusion matrix DataFrame for each class and condition.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame containing confusion matrix metrics.
        unique_classes : List[str]
            List of unique classes.
        unique_cond_classes : List[str]
            List of unique conditional classes.

        Returns:
        -------
        metrics : dict
            A dictionary containing calculated metrics.
        """
        metrics = {
            "f1_macro": df.loc[df['cond_class']['metric'] == 'macro', ('overall', 'f1_score')].values[0],
            "accuracy": df.loc[df['cond_class']['metric'] == 'micro', ('overall', 'f1_score')].values[0],
            "support": df.loc[df['cond_class']['metric'] == 'macro', ('overall', 'support')].values[0],
        }

        # Extract f1_macro and support for each class
        for uc in unique_classes:
            try:
                metrics[f"f1_macro_class_{uc}"] = df.loc[df['cond_class']['metric'] == uc, ('overall', 'f1_score')].values[0]
                metrics[f"support_class_{uc}"] = df.loc[df['cond_class']['metric'] == uc, ('overall', 'support')].values[0]
            except IndexError as e:
                logging.error(f"Error extracting metrics for class {uc}: {e}")
                metrics[f"f1_macro_class_{uc}"] = np.nan
                metrics[f"support_class_{uc}"] = np.nan

        # Extract f1_macro  an support for each conditional class and eac h class bases on conditional class
        for cond_uc in unique_cond_classes:
            try:
                metrics[f"f1_macro_cond_{cond_uc}"] = df.loc[df['cond_class']['metric'] == 'macro', 
                                                            (cond_uc, 'f1_score')].values[0]
                metrics[f"support_cond_{cond_uc}"] = df.loc[df['cond_class']['metric'] == 'macro', 
                                                            (cond_uc, 'support')].values[0]
                for uc in unique_classes:
                    metrics[f"f1_macro_class_{uc}_cond_{cond_uc}"] = df.loc[df['cond_class']['metric'] == uc, 
                                                                            (cond_uc, 'f1_score')].values[0]
                    metrics[f"support_{uc}_cond_{cond_uc}"] = df.loc[df['cond_class']['metric'] == uc,
                                                                      (cond_uc, 'support')].values[0]
            except IndexError as e:
                logging.error(f"Error extracting conditional metrics for cond_class {cond_uc}: {e}")
                metrics[f"f1_macro_cond_{cond_uc}"] = np.nan
                metrics[f"support_cond_{cond_uc}"] = np.nan
                for uc in unique_classes:
                    metrics[f"f1_macro_class_{uc}_cond_{cond_uc}"] = np.nan
                    metrics[f"support_{uc}_cond_{cond_uc}"] = np.nan
        
        return metrics