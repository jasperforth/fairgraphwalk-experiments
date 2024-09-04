
import sys
import logging
import numpy as np
import pandas as pd

import sklearn.metrics as metrics
from collections import defaultdict
from pathlib import Path


class PokecConfusion:
    """
    A class to handle the processing of confusion matrices and related metrics
    from experiments on various graphs, including saving averaged matrices and reports.

    Attributes:
    ----------
    matrices_sums_y : defaultdict
        Accumulated sums of confusion matrices for 'y' attribute across multiple runs.
    matrices_sums_z : defaultdict
        Accumulated sums of confusion matrices for 'z' attribute across multiple runs.
    run_counts : defaultdict
        Counts of runs for each graph, biasing condition, and experiment.
    """
    def __init__(self):
        """
        Initializes the PokecConfusion object with default structures for
        storing confusion matrices and run counts.
        """
        self.matrices_sums_y = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
        self.matrices_sums_z = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
        self.run_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Set up logging
        log_file = Path(f'pokec_confusion_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Initialized PokecConfusion")


    def process_results(self, graph_results_dir: Path, report_dir: Path, df_filtered_attributes: pd.DataFrame, graph_name: str, exp_graph_name: str):
        """
        Process the results by combining confusion matrices across multiple runs.

        Parameters:
        ----------
        graph_results_dir : Path
            The directory containing the results for a specific graph.
        report_dir : Path
            The directory where the final reports will be saved.
        df_filtered_attributes : pd.DataFrame
            DataFrame containing filtered attributes used in the experiments.
        graph_name : str
            The name of the graph being processed.
        exp_graph_name : str
            The experiment-graph combination name used for saving results.
        """
        logging.info(f"Processing results for graph {graph_name} in experiment {exp_graph_name}")
        for biasing_folder in graph_results_dir.iterdir():
            if biasing_folder.is_dir():
                biasing_name = biasing_folder.name
                logging.info(f"Processing biasing condition: {biasing_name}")
                for run_folder in biasing_folder.iterdir():
                    if run_folder.is_dir():
                        d_run_data = self.process_run_folder(run_folder, df_filtered_attributes)
                        for exp_name, exp_data in d_run_data.items():
                            if "matrix_y" not in exp_data:
                                logging.warning(f"'matrix_y' missing in experiment {exp_name} for run {run_folder.name}")
                            if "matrix_z" not in exp_data:
                                logging.warning(f"'matrix_z' missing in experiment {exp_name} for run {run_folder.name}")
                            # print(f"Processing experiment {exp_name} in run {run_folder.name}...")
                            if exp_name not in self.matrices_sums_y[graph_name][biasing_name]:
                                self.matrices_sums_y[graph_name][biasing_name][exp_name] = exp_data["matrix_y"]
                                self.matrices_sums_z[graph_name][biasing_name][exp_name] = exp_data["matrix_z"]
                            else:
                                self.matrices_sums_y[graph_name][biasing_name][exp_name] = self.add_confusion_matrices(
                                    self.matrices_sums_y[graph_name][biasing_name][exp_name], exp_data["matrix_y"], exp_name)
                                self.matrices_sums_z[graph_name][biasing_name][exp_name] = self.add_confusion_matrices(
                                    self.matrices_sums_z[graph_name][biasing_name][exp_name], exp_data["matrix_z"], exp_name)

                            self.run_counts[graph_name][biasing_name][exp_name] += 1

        # Calculate and save the average confusion matrices
        d_average_matrices_y = self.calculate_average_matrices(self.matrices_sums_y, self.run_counts)
        d_average_matrices_z = self.calculate_average_matrices(self.matrices_sums_z, self.run_counts)
        self.save_average_matrices(d_average_matrices_y, report_dir, 'y', exp_graph_name)
        self.save_average_matrices(d_average_matrices_z, report_dir, 'z', exp_graph_name)

        # Save the sklearn classification reports
        self.save_sklearn_classification_reports(d_average_matrices_y, report_dir, 'y', exp_graph_name)
        self.save_sklearn_classification_reports(d_average_matrices_z, report_dir, 'z', exp_graph_name)

        logging.info(f"Confusion reports for {exp_graph_name} created and saved in {report_dir}/confusion_reports.")


    def process_run_folder(self, run_folder, df_filtered_attributes):
        """
        Process individual run folders to extract and process confusion matrices.

        Parameters:
        ----------
        run_folder : Path
            The directory for a specific run of the experiment.
        df_filtered_attributes : pd.DataFrame
            DataFrame containing filtered attributes used in the experiments.

        Returns:
        -------
        dict
            A dictionary containing processed data for each experiment within the run.
        """
        d_run_data = {}
        for params_folder in run_folder.iterdir():
            if params_folder.is_dir():
                experiment_params = params_folder.name
                d_exp_data = {}
                for file in params_folder.iterdir():
                    if file.name == "confusion_y.csv":
                        y_matrix_dict = self.process_confusion_y(file, params_folder, df_filtered_attributes)
                        d_exp_data.update(y_matrix_dict)  # Merge the dictionary

                    if file.name == "confusion_z.csv":
                        z_matrix_dict = self.process_confusion_z(file, params_folder, df_filtered_attributes)
                        d_exp_data.update(z_matrix_dict)  # Merge the dictionary

                d_run_data[experiment_params] = d_exp_data

        return d_run_data
    

    def process_confusion_y(self, file: Path, params_folder: Path, df_filtered_attributes: pd.DataFrame):
        """
        Processes the confusion matrix file for the 'y' attribute.

        Parameters:
        ----------
        file : Path
            The file containing the confusion matrix for 'y'.
        params_folder : Path
            The directory containing experiment parameters.
        df_filtered_attributes : pd.DataFrame
            DataFrame containing filtered attributes used in the experiments.

        Returns:
        -------
        dict
            A dictionary containing the processed confusion matrix for 'y'.
        """
        logging.info(f"Processing confusion matrix for 'y' in {params_folder.name}")
        y_true_col, y_pred_col, cond_col = self.extract_labels(params_folder, 'y')
        
        df_y = pd.read_csv(file).merge(df_filtered_attributes[['user_id', cond_col]], on='user_id').drop("Unnamed: 0", axis=1)
        d_matrix_y = self.create_confusion_matrices(attr_test=df_y[y_true_col], attr_pred=df_y[y_pred_col], cond_attr=df_y[cond_col], attr_var="y")

        return {'matrix_y': d_matrix_y}


    def process_confusion_z(self, file: Path, params_folder: Path, df_filtered_attributes: pd.DataFrame):
        """
        Processes the confusion matrix file for the 'z' attribute.

        Parameters:
        ----------
        file : Path
            The file containing the confusion matrix for 'z'.
        params_folder : Path
            The directory containing experiment parameters.
        df_filtered_attributes : pd.DataFrame
            DataFrame containing filtered attributes used in the experiments.

        Returns:
        -------
        dict
            A dictionary containing the processed confusion matrix for 'z'.
        """
        logging.info(f"Processing confusion matrix for 'z' in {params_folder.name}")
        z_true_col, z_pred_col, cond_col = self.extract_labels(params_folder, 'z')

        df_z = pd.read_csv(file).merge(df_filtered_attributes[['user_id', cond_col]], on='user_id').drop("Unnamed: 0", axis=1)
        d_matrix_z = self.create_confusion_matrices(attr_test=df_z[z_true_col], attr_pred=df_z[z_pred_col], cond_attr=df_z[cond_col], attr_var="z")

        return {'matrix_z': d_matrix_z}
    

    def extract_labels(self, params_folder: Path, var_type: str):
        """
        Extracts the appropriate true label, predicted label, and condition columns
        based on the experiment parameters and the type of variable (y or z).

        Parameters:
        ----------
        params_folder : Path
            The directory containing experiment parameters.
        var_type : str
            The type of variable ('y' or 'z').

        Returns:
        -------
        tuple
            A tuple containing the true label column, predicted label column, and condition column.
        """
        experiment_name = params_folder.name
        if var_type == 'y':
            true_label = experiment_name.split('_other_')[1]
            pred_label = f'pred_{true_label}'
            cond_label = experiment_name.split('_sens_')[1].split('_other_')[0]  # Sens attribute is the condition when working with "y"
        elif var_type == 'z':
            true_label = experiment_name.split('_sens_')[1].split('_other_')[0]
            pred_label = f'pred_{true_label}'
            cond_label = experiment_name.split('_other_')[1]  # "Other" attribute is the condition when working with "z"
        else:
            raise ValueError('var_type must be either "y" or "z"')
        
        return true_label, pred_label, cond_label
    

    @staticmethod
    def create_confusion_matrices(attr_test: pd.DataFrame, attr_pred: pd.DataFrame, cond_attr: pd.DataFrame=None, attr_var: str=None):
        """
        Create confusion matrices for given attributes, optionally conditioned on another attribute.

        Parameters:
        ----------
        attr_test : pd.DataFrame
            The true labels of the attributes.
        attr_pred : pd.DataFrame
            The predicted labels of the attributes.
        cond_attr : pd.DataFrame, optional
            The attribute used for conditioning (default is None).
        attr_var : str, optional
            The variable type ('y' or 'z') (default is None).

        Returns:
        -------
        dict
            A dictionary containing confusion matrices, optionally conditioned on another attribute.
        """
        if attr_var == 'y':
            cond_var = 'z'
        elif attr_var == 'z':
            cond_var = 'y'
        else:
            raise ValueError('attr_var must be either "y" or "z"')
        
        attr_test = np.array(attr_test)
        attr_pred = np.array(attr_pred)

        # Get all unique labels present across both attr_test and attr_pred
        all_labels = np.unique(np.concatenate([attr_test, attr_pred]))
        
        # sklearn confusion matrix
        cm = metrics.confusion_matrix(attr_test, attr_pred, labels=all_labels)
        d_cm = {('overall', attr_var, cond_var): cm}

        if cond_attr is not None:
            cond_attr = np.array(cond_attr)
            unique_cond_values = np.unique(cond_attr)
            for val in unique_cond_values:
                indices = np.where(cond_attr == val)
                cm_cond = metrics.confusion_matrix(attr_test[indices], attr_pred[indices], labels=all_labels)
                d_cm[(val, attr_var, cond_var)] = cm_cond

        return d_cm


    @staticmethod
    def get_or_create_report_dir(report_dir: Path, graph_name:str, biasing_name: str, experiment_name: str, exp_graph_name: str):
        """
        Create or get the directory path for saving reports, ensuring the directory exists.

        Parameters:
        ----------
        report_dir : Path
            The base directory for reports.
        graph_name : str
            The name of the graph.
        biasing_name : str
            The name of the biasing condition.
        experiment_name : str
            The name of the experiment.
        exp_graph_name : str
            The name of the experiment-graph combination.

        Returns:
        -------
        Path
            The directory path where the report will be saved.
        """
        exp_graph_dir = exp_graph_name.replace(" ", "_")
        report_dir = report_dir / 'confusion_reports' / exp_graph_dir / biasing_name / experiment_name
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir
    

    @staticmethod
    def add_confusion_matrices(matrix_a: dict, matrix_b: dict, exp_name: str):
        """
        Add two confusion matrices together, ensuring consistent dimensions by padding with zeros if necessary.

        Parameters:
        ----------
        matrix_a : dict
            The first confusion matrix to be added. Can be None initially.
        matrix_b : dict
            The second confusion matrix to be added.
        exp_name : str
            The name of the experiment, used for error messages.

        Returns:
        -------
        dict
            The resulting confusion matrix after addition, with dimensions adjusted to account for any missing labels.
        """
        if matrix_a is None:
            return matrix_b.copy()
        
        for key, value in matrix_b.items():
            if key in matrix_a:
                # Ensure both matrices have the same dimensions
                if matrix_a[key].shape != value.shape:
                    logging.info(f"Padded matrices for {exp_name} in {key} biasing:")
                    # Find the larger shape
                    max_shape = (
                        max(matrix_a[key].shape[0], value.shape[0]),
                        max(matrix_a[key].shape[1], value.shape[1])
                    )

                    padded_matrix_a = np.zeros(max_shape)
                    padded_matrix_b = np.zeros(max_shape)
                    
                    # Place matrix_a values into the padded matrix
                    padded_matrix_a[:matrix_a[key].shape[0], :matrix_a[key].shape[1]] = matrix_a[key]
                    
                    # Place matrix_b values into the padded matrix
                    padded_matrix_b[:value.shape[0], :value.shape[1]] = value
                    
                    # Add the padded matrices
                    matrix_a[key] = padded_matrix_a + padded_matrix_b
                else:
                    # If dimensions match, simply add the matrices
                    matrix_a[key] += value
            else:
                # Copy matrix_b directly if key is not present in matrix_a
                matrix_a[key] = value.copy()
        
        return matrix_a
    
    
    @staticmethod
    def calculate_average_matrices(matrices_sums: dict, run_counts: dict):
        """
        Calculate the average confusion matrices from the accumulated sums across multiple runs.

        Parameters:
        ----------
        matrices_sums : dict
            A nested dictionary containing summed confusion matrices for each graph, biasing condition, and experiment.
        run_counts : dict
            A nested dictionary containing the count of runs for each graph, biasing condition, and experiment.

        Returns:
        -------
        dict
            A nested dictionary containing the averaged confusion matrices for each graph, biasing condition, and experiment.
        """
        average_matrices = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for graph_name, biasing_data in matrices_sums.items():
            for biasing_name, exps_data in biasing_data.items():
                for exp_name, sum_matrix in exps_data.items():
                    avg_matrix = {}
                    for key, matrix in sum_matrix.items():
                        # Calculate the average by dividing the summed matrix by the run count
                        avg_matrix[key] = matrix / run_counts[graph_name][biasing_name][exp_name]

                    average_matrices[graph_name][biasing_name][exp_name] = avg_matrix

        return average_matrices


    @staticmethod
    def confusion_matrices_to_dataframe(confusion_matrices: dict):
        """
        Convert confusion matrices from dictionary format to a Pandas DataFrame.

        Parameters:
        ----------
        confusion_matrices : dict
            A dictionary containing confusion matrices.

        Returns:
        -------
        pd.DataFrame
            A DataFrame representation of the confusion matrices.
        """
        data = []
        for (cond, attr_var, cond_var), cm in confusion_matrices.items():
            num_classes = cm.shape[0]
            for i in range(num_classes):
                for j in range(num_classes):
                    row = {
                        'cond_attr_value': cond,
                        'attr_var': attr_var,
                        'cond_var': cond_var,
                        'true_label': i,
                        'pred_label': j,
                        'count': cm[i, j]
                    }
                    data.append(row)

        df_cm = pd.DataFrame(data)
        df_cm.set_index(['cond_attr_value', 'attr_var', 'cond_var', 'true_label', 'pred_label'], inplace=True)
        return df_cm


    def save_average_matrices(self, average_matrices: dict, report_dir: Path, matrix_name: str, exp_graph_name: str):
        """
        Save the averaged confusion matrices to CSV files.

        Parameters:
        ----------
        average_matrices : dict
            A dictionary containing the averaged confusion matrices.
        report_dir : Path
            The directory where the reports will be saved.
        matrix_name : str
            The name of the matrix ('y' or 'z').
        exp_graph_name : str
            The name of the experiment-graph combination.
        """
        for graph_name, biasing_data in average_matrices.items():
            for biasing_name, exps_data in biasing_data.items():
                for exp_name, avg_matrix in exps_data.items():
                    report_path = self.get_or_create_report_dir(report_dir, graph_name, biasing_name, exp_name, exp_graph_name)

                    df_avg_matrix = self.confusion_matrices_to_dataframe(avg_matrix)
                    cf_report = self.confusion_report(df_avg_matrix)

                    df_avg_matrix.to_csv(report_path / f"average_confusion_matrix_{matrix_name}.csv")
                    cf_report.to_csv(report_path / f"average_confusion_report_{matrix_name}.csv")



    @staticmethod
    def confusion_matrix_to_labels(df_confusion_matrix: pd.DataFrame):
        """
        Convert a confusion matrix DataFrame back into true and predicted label arrays.

        Parameters:
        ----------
        df_confusion_matrix : pd.DataFrame
            The DataFrame containing the confusion matrix.

        Returns:
        -------
        tuple
            Two arrays: true labels and predicted labels.
        """
        true_labels = []
        pred_labels = []

        for _, row in df_confusion_matrix.iterrows():
            true_label = row['true_label']
            pred_label = row['pred_label']
            count = int(row['count'])

            true_labels.extend([true_label] * count)
            pred_labels.extend([pred_label] * count)

        return np.array(true_labels), np.array(pred_labels)


    def save_sklearn_classification_reports(self, d_average_matrices: dict, report_dir: Path,  matrix_name: str, exp_graph_name: str):
        """
        Save the sklearn classification reports based on the averaged confusion matrices.

        Parameters:
        ----------
        d_average_matrices : dict
            A dictionary containing the averaged confusion matrices.
        report_dir : Path
            The directory where the reports will be saved.
        matrix_name : str
            The name of the matrix ('y' or 'z').
        exp_graph_name : str
            The name of the experiment-graph combination.
        """
        for graph_name, biasing_data in d_average_matrices.items():
            for biasing_name, exps_data in biasing_data.items():
                for exp_name, avg_matrix in exps_data.items():
                    report_path = self.get_or_create_report_dir(report_dir, graph_name, biasing_name, exp_name, exp_graph_name)
                    csv_path = report_path / f'average_confusion_matrix_{matrix_name}.csv'

                    df_avg_matrix = pd.read_csv(csv_path)

                    true_labels, pred_labels = self.confusion_matrix_to_labels(df_avg_matrix)
                    report = metrics.classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)

                    df_report = pd.DataFrame(report).transpose()
                    df_report.to_csv(report_path / f'sklearn_classification_report_{matrix_name}.csv', index=True)


    @staticmethod        
    def confusion_report(df_matrix: pd.DataFrame):
        """
        Generate a detailed report from a confusion matrix DataFrame.

        Parameters:
        ----------
        df_matrix : pd.DataFrame
            The DataFrame containing the confusion matrix.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing various classification metrics.
        """
        cond_classes = df_matrix.index.get_level_values('cond_attr_value').unique()
        attr_classes = df_matrix.index.get_level_values('true_label').unique()

        sample_report_dict = {
            i: {
                "accuracy": 0, 
                "recall": 0, 
                "precision": 0, 
                "f1_score": 0, 
                "specificity": 0, 
                "support": 0
            } 
            for i in attr_classes
        }

        sample_report_dict.update(
            {
                "micro": {}, 
                "macro": {}, 
                "weighted": {}
            }
        )

        sample_report_df = pd.DataFrame.from_dict(sample_report_dict, orient='index')
        final_report_df = pd.DataFrame(columns=pd.MultiIndex.from_product([cond_classes, sample_report_df.columns], names=['cond_class', 'metric']))

        for cond_class in cond_classes:
            df_cond_class = df_matrix[df_matrix.index.get_level_values('cond_attr_value') == cond_class]
            num_attr_classes = len(attr_classes)
            matrix = np.zeros((num_attr_classes, num_attr_classes))

            for attr_class in attr_classes:
                for pred_label in attr_classes:
                    try:
                        count = df_cond_class.loc[(cond_class, slice(None), slice(None), attr_class, pred_label), 'count'].sum()
                        matrix[attr_classes.get_loc(attr_class), attr_classes.get_loc(pred_label)] = count

                    except KeyError as e:
                        logging.error(f"KeyError: {e} - Failed to locate the slice in df_cond_class. Details:")
                        logging.error(f"cond_class: {cond_class}, attr_class: {attr_class}, pred_label: {pred_label}")
                        logging.error(f"df_cond_class head: \n{df_cond_class.head()}")
                        raise


            # Initialize sums and counters for calculating averages
            report_dict = {}
            recall_sum = 0
            recall_weighted_sum = 0
            precision_sum = 0
            precision_weighted_sum = 0
            f1_sum = 0
            f1_weighted_sum = 0
            specificity_sum = 0
            specificity_weighted_sum = 0
                
            tp_sum = 0
            fp_sum = 0
            tn_sum = 0
            fn_sum = 0
            support_sum = 0

            # Calculate metrics for each class
            for idx, attr_class in enumerate(attr_classes):    
                tp = matrix[idx, idx]
                fp = np.sum(matrix[:, idx]) - matrix[idx, idx]
                fn = np.sum(matrix[idx, :]) - tp
                tn = np.sum(matrix) - (tp + fp + fn)

                support = np.sum(matrix[idx, :])
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                recall = tp / (tp + fn) if (tp + fn) != 0 else 0
                precision = tp / (tp + fp) if (tp + fp) != 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) \
                                if (precision + recall) != 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) != 0 else 0           

                # Update sums for weighted averages
                recall_sum += recall
                recall_weighted_sum += recall * support
                precision_sum += precision
                precision_weighted_sum += precision * support
                f1_sum += f1_score
                f1_weighted_sum += f1_score * support
                specificity_sum += specificity
                specificity_weighted_sum += specificity * support
                    
                tp_sum += tp
                fp_sum += fp
                tn_sum += tn
                fn_sum += fn
                support_sum += support

                report_dict[attr_class] = {
                    "accuracy": accuracy, 
                    "recall": recall, 
                    "precision": precision, 
                    "f1_score": f1_score, 
                    "specificity": specificity, 
                    "support": support
                } 

            # Calculate averages
            avg_recall_macro = recall_sum / len(attr_classes)
            avg_precision_macro = precision_sum / len(attr_classes)
            avg_f1_macro = f1_sum / len(attr_classes)
            avg_specifity_macro = specificity_sum / len(attr_classes)

            avg_recall_micro = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) != 0 else 0
            avg_precision_micro = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) != 0 else 0
            avg_f1_micro = tp_sum / (tp_sum + 0.5*(fp_sum + fn_sum)) if (tp_sum + fp_sum + fn_sum) != 0 else 0
            avg_specifity_micro = tn_sum / (tn_sum + fp_sum) if (tn_sum + fp_sum) != 0 else 0

            avg_recall_weighted = recall_weighted_sum / support_sum
            avg_precision_weighted = precision_weighted_sum / support_sum
            avg_f1_weighted = f1_weighted_sum / support_sum
            avg_specificity_weighted = specificity_weighted_sum / support_sum

            report_dict["micro"] = {
                "f1_score": avg_f1_micro, 
                "recall": avg_recall_micro, 
                "precision": avg_precision_micro, 
                "specificity": avg_specifity_micro, 
                "support": support_sum,
            }
            report_dict["macro"] = {
                "f1_score": avg_f1_macro, 
                "recall": avg_recall_macro, 
                "precision": avg_precision_macro, 
                "specificity": avg_specifity_macro,
                "support": support_sum,
            }
            report_dict["weighted"] = {
                "f1_score": avg_f1_weighted, 
                "recall": avg_recall_weighted, 
                "precision": avg_precision_weighted, 
                "specificity": avg_specificity_weighted, 
                "support": support_sum, 
            }

            report_df = pd.DataFrame.from_dict(report_dict, orient='index')
            # Append the report_df to final_report_df for the current z_class
            for metric, values in report_df.items():
                final_report_df[(cond_class, metric)] = values

        return final_report_df
    
