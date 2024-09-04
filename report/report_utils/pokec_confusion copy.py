
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
from pathlib import Path

import biasing


class PokecConfusion:
    def __init__(self):
        self.matrices_sums_y = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
        self.matrices_sums_z = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
        self.run_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


    def process_results(self, 
                        graph_results_dir: Path, 
                        report_dir: Path, 
                        df_filtered_attributes: pd.DataFrame, 
                        graph_name: str, 
                        exp_graph_name: str, 
                        
                        ):
        # print("df_filtered_attributes head: \n", df_filtered_attributes.head())
        for biasing_folder in graph_results_dir.iterdir():
            if biasing_folder.is_dir():
                biasing_name = biasing_folder.name
                # print(f"biasing_folder name : {biasing_name}")
                for run_folder in biasing_folder.iterdir():
                    if run_folder.is_dir():
                        # print(f"run_folder name : {run_folder.name}")
                        d_run_data = self.process_run_folder(run_folder, df_filtered_attributes)
                        print("d_run_data:", d_run_data)
                        # Update dictionaries for confusion matrix sums and run counts

                        for exp_name, exp_data in d_run_data.items():
                            self.matrices_sums_y[graph_name][biasing_name][exp_name] = self.add_confusion_matrices(self.matrices_sums_y[graph_name][biasing_name][exp_name], exp_data["matrix_y"])
                            self.matrices_sums_z[graph_name][biasing_name][exp_name] = self.add_confusion_matrices(self.matrices_sums_z[graph_name][biasing_name][exp_name], exp_data["matrix_z"])
                            self.run_counts[graph_name][biasing_name][exp_name] += 1

                     
        print(f"Confusion reports for {exp_graph_name} created \
                \nand saved in {report_dir}/confusion_reports.")
               

        # Calculate the average confusion matrices
        d_average_matrices_y = self.calculate_average_matrices(self.matrices_sums_y, self.run_counts)
        d_average_matrices_z = self.calculate_average_matrices(self.matrices_sums_z, self.run_counts)

        # Save the average confusion matrices
        self.save_average_matrices(d_average_matrices_y, report_dir, 'y', exp_graph_name)
        self.save_average_matrices(d_average_matrices_z, report_dir, 'z', exp_graph_name)

        # Save the sklearn classification reports
        self.save_sklearn_classification_reports(d_average_matrices_y, report_dir, 'y', exp_graph_name)
        self.save_sklearn_classification_reports(d_average_matrices_z, report_dir, 'z', exp_graph_name)


    def process_run_folder(self, run_folder, df_filtered_attributes):
        d_run_data = {}
        for params_folder in run_folder.iterdir():
            if params_folder.is_dir():
                # print(f"params_folder name : {params_folder.name}")
                d_exp_data = {}
                experiment_params = params_folder.name
                for file in params_folder.iterdir():
                    if file.name == "confusion_y.csv":
                        d_exp_data["matrix_y"] = self.process_confusion_y(file, params_folder, df_filtered_attributes)

                    if file.name == "confusion_z.csv":
                        d_exp_data["matrix_z"] = self.process_confusion_z(file, params_folder, df_filtered_attributes)

                d_run_data[experiment_params] = d_exp_data

        # print(f"d_run_data: \n{d_run_data}")
        return d_run_data
    


    @staticmethod
    def get_or_create_report_dir(report_dir: Path, graph_name:str, biasing_name: str, experiment_name: str, exp_graph_name: str):
        exp_graph_dir = exp_graph_name.replace(" ", "_")
        report_dir = report_dir / 'confusion_reports' / exp_graph_dir / biasing_name / experiment_name
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir


    
    @staticmethod
    def create_confusion_matrices(attr_test: pd.DataFrame, attr_pred: pd.DataFrame, cond_attr: pd.DataFrame=None, attr_var: str=None):
        if attr_var == 'y':
            cond_var = 'z'
        elif attr_var == 'z':
            cond_var = 'y'
        else:
            raise ValueError('attr_var must be either "y" or "z"')
        
        attr_test = np.array(attr_test)
        attr_pred = np.array(attr_pred)
        
        # sklearn confusion matrix
        cm = confusion_matrix(attr_test, attr_pred)
        d_cm = {('overall', attr_var, cond_var): cm}

        if cond_attr is not None:
            cond_attr = np.array(cond_attr)
            unique_cond_values = np.unique(cond_attr)
            for val in unique_cond_values:
                indices = np.where(cond_attr == val)
                cm_cond = confusion_matrix(attr_test[indices], attr_pred[indices])
                d_cm[(val, attr_var, cond_var)] = cm_cond

        return d_cm


    @staticmethod
    def confusion_matrices_to_dataframe(confusion_matrices: dict):
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
        # print("Confusion matrix DataFrame before setting index:", df_cm.head())
        df_cm.set_index(['cond_attr_value', 'attr_var', 'cond_var', 'true_label', 'pred_label'], inplace=True)
        # print("Confusion matrix DataFrame after setting index:", df_cm.head())
        return df_cm


    @staticmethod
    def add_confusion_matrices(matrix_a: dict, matrix_b: dict):
        if matrix_a is None:
            return matrix_b.copy()
        for key, value in matrix_b.items():
            if key in matrix_a:
                matrix_a[key] += value
            else:
                matrix_a[key] = value.copy()
        
        return matrix_a


    def process_confusion_y(self, file: Path, params_folder: Path , df_filtered_attributes: pd.DataFrame):
        y_true_col = "label_AGE" if params_folder.name.endswith("other_label_AGE") else "label_region"
        y_pred_col = "pred_label_AGE" if params_folder.name.endswith("other_label_AGE") else "pred_label_region"
        cond_col = 'label_region' if params_folder.name.endswith("other_label_AGE") else 'label_AGE'

        df_y = pd.read_csv(file).merge(df_filtered_attributes[['user_id', cond_col]], on='user_id').drop("Unnamed: 0", axis=1)
        # print(f"df_y head: \n{df_y.head()}")
        
        d_matrix_y = self.create_confusion_matrices(attr_test=df_y[y_true_col], attr_pred=df_y[y_pred_col], cond_attr=df_y[cond_col], attr_var="y")
        # print(f"d_matrix_y: \n{d_matrix_y}")

        return d_matrix_y
    

    def process_confusion_y(self, file: Path, params_folder: Path , df_filtered_attributes: pd.DataFrame):
        y_true_col = "label_AGE" if params_folder.name.endswith("other_label_AGE") else "label_region"
        y_pred_col = "pred_label_AGE" if params_folder.name.endswith("other_label_AGE") else "pred_label_region"
        cond_col = 'label_region' if params_folder.name.endswith("other_label_AGE") else 'label_AGE'

        df_y = pd.read_csv(file).merge(df_filtered_attributes[['user_id', cond_col]], on='user_id').drop("Unnamed: 0", axis=1)
        # print(f"df_y head: \n{df_y.head()}")

        d_matrix_y = self.create_confusion_matrices(attr_test=df_y[y_true_col], attr_pred=df_y[y_pred_col], cond_attr=df_y[cond_col], attr_var="y")
        # print(f"d_matrix_y: \n{d_matrix_y}")

        return d_matrix_y


    def process_confusion_z(self, file: Path, params_folder: Path , df_filtered_attributes: pd.DataFrame):
        # If we are using the other label, we need to swap the true and pred labels
        z_true_col = 'label_region' if params_folder.name.endswith("other_label_AGE") else 'label_AGE'
        z_pred_col = 'pred_label_region' if params_folder.name.endswith("other_label_AGE") else 'pred_label_AGE'
        cond_col = "label_AGE" if params_folder.name.endswith("other_label_AGE") else "label_region"

        df_z = pd.read_csv(file).merge(df_filtered_attributes[['user_id', cond_col]], on='user_id').drop("Unnamed: 0", axis=1)
        # print(f"df_z head: \n{df_z.head()}")

        d_matrix_z = self.create_confusion_matrices(attr_test=df_z[z_true_col], attr_pred=df_z[z_pred_col], cond_attr=df_z[cond_col], attr_var="z")
        # print(f"d_matrix_z: \n{d_matrix_z}")

        return d_matrix_z


    @staticmethod
    def calculate_average_matrices(matrices_sums: dict, run_counts: dict):
        average_matrices = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for graph_name, biasing_data in matrices_sums.items():
            for biasing_name, exps_data in biasing_data.items():
                for exp_name, sum_matrix in exps_data.items():
                    avg_matrix = {}
                    for key, matrix in sum_matrix.items():
                        avg_matrix[key] = matrix / run_counts[graph_name][biasing_name][exp_name]
                    average_matrices[graph_name][biasing_name][exp_name] = avg_matrix
        return average_matrices


    def save_average_matrices(self, average_matrices: dict, report_dir: Path, matrix_name: str, exp_graph_name: str):
        for graph_name, biasing_data in average_matrices.items():
            for biasing_name, exps_data in biasing_data.items():
                for exp_name, avg_matrix in exps_data.items():
                    report_path = self.get_or_create_report_dir(report_dir, graph_name, biasing_name, exp_name, exp_graph_name)

                    df_avg_matrix = self.confusion_matrices_to_dataframe(avg_matrix)
                    # print("DataFrame columns before setting index:", df_avg_matrix.columns)
                    cf_report = self.confusion_report(df_avg_matrix)

                    df_avg_matrix.to_csv(report_path / f"average_confusion_matrix_{matrix_name}.csv")
                    cf_report.to_csv(report_path / f"average_confusion_report_{matrix_name}.csv")


    @staticmethod
    def confusion_matrix_to_labels(df_confusion_matrix: pd.DataFrame):
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
        for graph_name, biasing_data in d_average_matrices.items():
            for biasing_name, exps_data in biasing_data.items():
                for exp_name, avg_matrix in exps_data.items():
                    report_path = self.get_or_create_report_dir(report_dir, graph_name, biasing_name, exp_name, exp_graph_name)
                    csv_path = report_path / f'average_confusion_matrix_{matrix_name}.csv'

                    df_avg_matrix = pd.read_csv(csv_path)

                    true_labels, pred_labels = self.confusion_matrix_to_labels(df_avg_matrix)
                    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)

                    df_report = pd.DataFrame(report).transpose()
                    df_report.to_csv(report_path / f'sklearn_classification_report_{matrix_name}.csv', index=True)


    @staticmethod        
    def confusion_report(df_matrix: pd.DataFrame):
        cond_classes = df_matrix.index.get_level_values('cond_attr_value').unique()
        attr_classes = df_matrix.index.get_level_values('true_label').unique()

        sample_report_dict = {i: {"accuracy": 0, "recall": 0, "precision": 0, 
                                "f1_score": 0, "specificity": 0, "support": 0} for i in attr_classes}
        sample_report_dict.update({"micro": {}, "macro": {}, "weighted": {}})
        sample_report_df = pd.DataFrame.from_dict(sample_report_dict, orient='index')

        final_report_df = pd.DataFrame(columns=pd.MultiIndex.from_product([cond_classes, sample_report_df.columns], names=['cond_class', 'metric']))

        for cond_class in cond_classes:
            df_cond_class = df_matrix[df_matrix.index.get_level_values('cond_attr_value') == cond_class]
            num_attr_classes = len(attr_classes)
            matrix = np.zeros((num_attr_classes, num_attr_classes))

            for attr_class in attr_classes:
                for pred_label in attr_classes:
                    count = df_cond_class.loc[(cond_class, slice(None), slice(None), attr_class, pred_label), 'count'].sum()
                    matrix[attr_classes.get_loc(attr_class), attr_classes.get_loc(pred_label)] = count

            report_dict = {}
            recall_sum = 0
            recall_weigthed_sum = 0
            precision_sum = 0
            precision_weigthed_sum = 0
            f1_sum = 0
            f1_weigthed_sum = 0
            specificity_sum = 0
            specificity_weigthed_sum = 0
            
            tp_sum = 0
            fp_sum = 0
            tn_sum = 0
            fn_sum = 0
            sup_sum = 0

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
                # Specificity: It tells you what fraction of all negative samples are correctly predicted 
                # as negative by the classifier. It is also known as True Negative Rate (TNR). 
                # To calculate specificity, use the following formula: TN/(TN+FP).
                recall_sum += recall
                recall_weigthed_sum += recall * support
                precision_sum += precision
                precision_weigthed_sum += precision * support
                f1_sum += f1_score
                f1_weigthed_sum += f1_score * support
                specificity_sum += specificity
                specificity_weigthed_sum += specificity * support
                
                tp_sum += tp
                fp_sum += fp
                tn_sum += tn
                fn_sum += fn
                sup_sum += support

                report_dict[attr_class] = {"accuracy": accuracy, "recall": recall, "precision": precision, 
                                    "f1_score": f1_score, "specificity": specificity, "support": support} 

            avg_recall_macro = recall_sum / len(attr_classes)
            #avg_recall_macro = np.mean([report_dict[attr_class]["recall"] for attr_class in attr_classes])
            avg_precision_macro = precision_sum / len(attr_classes)
            #avg_precision_macro = np.mean([report_dict[attr_class]["precision"] for attr_class in attr_classes])
            avg_f1_macro = f1_sum / len(attr_classes)
            #avg_f1_macro = np.mean([report_dict[attr_class]["f1_score"] for attr_class in attr_classes])
            avg_specifity_macro = specificity_sum / len(attr_classes)
            #avg_specifity_macro = np.mean([report_dict[attr_class]["specificity"] for attr_class in attr_classes])

            avg_recall_micro = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) != 0 else 0
            avg_precision_micro = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) != 0 else 0
            avg_f1_micro = tp_sum / (tp_sum + 0.5*(fp_sum + fn_sum)) if (tp_sum + fp_sum + fn_sum) != 0 else 0
            avg_specifity_micro = tn_sum / (tn_sum + fp_sum) if (tn_sum + fp_sum) != 0 else 0


            avg_recall_weighted = recall_weigthed_sum / np.sum(matrix)
            avg_precision_weighted = precision_weigthed_sum / np.sum(matrix)
            avg_f1_weighted = f1_weigthed_sum / np.sum(matrix)
            avg_specifity_weighted = specificity_weigthed_sum / np.sum(matrix)

            report_dict["micro"] = {"f1_score": avg_f1_micro, "recall": avg_recall_micro, "precision": avg_precision_micro, "specificity": avg_specifity_micro, "support": sup_sum}
            report_dict["macro"] = {"f1_score": avg_f1_macro, "recall": avg_recall_macro, "precision": avg_precision_macro, "specificity": avg_specifity_macro, "support": sup_sum}
            report_dict["weighted"] = {"f1_score": avg_f1_weighted, "recall": avg_recall_weighted, "precision": avg_precision_weighted, "specificity": avg_specifity_weighted, "support": sup_sum}

            report_df = pd.DataFrame.from_dict(report_dict, orient='index')
            # Append the report_df to final_report_df for the current z_class
            for metric, values in report_df.items():
                final_report_df[(cond_class, metric)] = values

        return final_report_df

