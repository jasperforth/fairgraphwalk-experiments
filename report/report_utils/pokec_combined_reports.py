from typing import List
from pathlib import Path

import pandas as pd


class PokecCombinedReports:
    def __init__(self, report_dir: Path):
        self.report_dir = report_dir
        self.confusion_dir = self.report_dir / f'confusion_reports'
        self.combined_results_dir = self.report_dir / 'combined_results'
        self.combined_results_dir.mkdir(parents=True, exist_ok=True)

    def run_averaging_df(self, experiment_name: str, graph_list: List[int]):
        # experiment_name: i.e. pokec semi or pokec mixed
        for exp_graph_dir in self.confusion_dir.iterdir():
            if exp_graph_dir.is_dir() and exp_graph_dir.name.startswith(experiment_name.replace(" ", "_")):
                graph_number = int(exp_graph_dir.name.split('_')[-1])  # Extract the graph number
                if graph_number in graph_list:
                    exp_graph_path = str(f"{experiment_name} {graph_number}").replace(" ", "_")
                    combined_df_sens_age = None
                    combined_df_sens_region = None
                    combined_df = None
                    for biasing_dir in exp_graph_dir.iterdir():
                        if biasing_dir.is_dir() and biasing_dir.name == "crosswalk": 
                            for exp_dir in biasing_dir.iterdir():
                                combined_df_sens_age = self.process_experiments_reports_cw(exp_dir, "label_region", combined_df_sens_age)
                                combined_df_sens_region = self.process_experiments_reports_cw(exp_dir, "label_AGE", combined_df_sens_region)
                           
                            combined_df_sens_age.to_csv(self.combined_results_dir / f'{exp_graph_path}_results_cw_sens_AGE.csv', index=False)
                            combined_df_sens_region.to_csv(self.combined_results_dir / f'{exp_graph_path}_results_cw_sens_region.csv', index=False)
                        
                        elif biasing_dir.is_dir() and biasing_dir.name == "baseline":
                            for exp_dir in biasing_dir.iterdir():
                                combined_df = self.process_experiments_reports_baseline(exp_dir, combined_df)
                            
                            combined_df.to_csv(self.combined_results_dir / f'{exp_graph_path}_results_baseline.csv', index=False)


    @staticmethod
    def extract_unique_classes_and_cond_classes(df):
        # Extract unique classes and unique conditional classes
        unique_classes = df['cond_class']['metric'].unique()
        unique_classes = [x for x in unique_classes if x.isdigit()]
        unique_cond_classes = df.columns.get_level_values(0).unique()
        unique_cond_classes = [x for x in unique_cond_classes if x.isdigit()]
        
        return unique_classes, unique_cond_classes
    

    def process_experiments_reports_baseline(self, 
                                            exp_dir: Path, 
                                            combined_df: pd.DataFrame,):                              
        if exp_dir.name.startswith('experiment'):
            def parse_directory_name_baseline(directory_name):
                parts = directory_name.split('_')
                p = float(parts[parts.index('p') + 1])
                q = float(parts[parts.index('q') + 1])
                return p, q
            
            p, q = parse_directory_name_baseline(exp_dir.name)
            sens_attr = 'AGE'
            file_names = ['average_confusion_report_y.csv', 'average_confusion_report_z.csv']
            for csv_file in file_names:
                file_path = exp_dir / csv_file
                if file_path.exists():
                    # Read CSV file and extract unique class and conditional class values
                    df = pd.read_csv(file_path, header=[0, 1])
                    unique_classes, unique_cond_classes =  self.extract_unique_classes_and_cond_classes(df)
                    # Create f1 and conditional f1 column names
                    f1_columns = [f'f1_macro_class_{uc}' for uc in unique_classes]
                    f1_cond_columns = [f'f1_macro_cond_{uc_cond}' for uc_cond in unique_cond_classes]
                    for uc in unique_classes:
                        f1_cond_columns += [f'f1_macro_class_{uc}_cond_{uc_cond}' for uc_cond in unique_cond_classes]

                    # Create an empty DataFrame with the required columns
                    if combined_df is None:
                        combined_df = pd.DataFrame(columns=['p', 'q', 
                                                            'sens_attr', 'attribute', 
                                                            'f1_macro', 'accuracy', 'support',
                                                            *f1_columns, *f1_cond_columns])

                    metrics = self.calculate_metrics(df, unique_classes, unique_cond_classes)
                    attribute = 'other' if csv_file == 'average_confusion_report_y.csv' else 'sensitive'
                    
                    # TODO FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.
                    combined_df = pd.concat([combined_df, pd.DataFrame({
                        'p': [p],
                        'q': [q],
                        'sens_attr': [sens_attr],
                        'attribute': [attribute],
                        **metrics
                    })], ignore_index=True)    

        return combined_df
    

    def process_experiments_reports_cw(self, exp_dir: Path, 
                                other_attr_label: str, 
                                combined_df: pd.DataFrame
                                ):
        if exp_dir.name.endswith(other_attr_label):
            def parse_directory_name_cw(directory_name):
                parts = directory_name.split('_')
                alpha = float(parts[parts.index('alpha') + 1])
                exponent = int(parts[parts.index('exponent') + 1])
                p = float(parts[parts.index('p') + 1])
                q = float(parts[parts.index('q') + 1])
                sens_attr = parts[parts.index('sens') + 2]
                return alpha, exponent, p, q, sens_attr
        
            alpha, exponent, p, q, sens_attr = parse_directory_name_cw(exp_dir.name)
            file_names = ['average_confusion_report_y.csv', 'average_confusion_report_z.csv']
            for csv_file in file_names:
                file_path = exp_dir / csv_file
                if file_path.exists():
                    # Read CSV file and extract unique class and conditional class values
                    df = pd.read_csv(file_path, header=[0, 1])
                    unique_classes, unique_cond_classes = self.extract_unique_classes_and_cond_classes(df)
                    # Create f1 and conditional f1 column names
                    f1_columns = [f'f1_macro_class_{uc}' for uc in unique_classes]
                    f1_cond_columns = [f'f1_macro_cond_{uc_cond}' for uc_cond in unique_cond_classes]
                    for uc in unique_classes:
                        f1_cond_columns += [f'f1_macro_class_{uc}_cond_{uc_cond}' for uc_cond in unique_cond_classes]

                    # Create an empty DataFrame with the requireexp_graph_dir.name.startswith(graph_category.replace(" ", "_")):exp_graph_dir.name.startswith(graph_category.replace(" ", "_")):d columns
                    if combined_df is None:
                        combined_df = pd.DataFrame(columns=['alpha', 'exponent', 'p', 'q', 
                                                            'sens_attr', 'attribute', 
                                                            'f1_macro', 'accuracy', 'support',
                                                            *f1_columns, *f1_cond_columns])

                    metrics = self.calculate_metrics(df, unique_classes, unique_cond_classes)

                    attribute = 'other' if csv_file == 'average_confusion_report_y.csv' else 'sensitive'
                    # TODO FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.
                    combined_df = pd.concat([combined_df, pd.DataFrame({
                        'alpha': [alpha],
                        'exponent': [exponent],
                        'p': [p],
                        'q': [q],
                        'sens_attr': [sens_attr],
                        'attribute': [attribute],
                        **metrics
                    })], ignore_index=True)       
        return combined_df
    
    
    @staticmethod
    def calculate_metrics(df, unique_classes, unique_cond_classes):
        metrics = {
            "f1_macro": df.loc[df['cond_class']['metric'] == 'macro', ('overall', 'f1_score')].values[0],
            "accuracy": df.loc[df['cond_class']['metric'] == 'micro', ('overall', 'f1_score')].values[0],
            "support": df.loc[df['cond_class']['metric'] == 'macro', ('overall', 'support')].values[0],
        }
        # Extract f1_macro and support for each class
        for uc in unique_classes:
            metrics[f"f1_macro_class_{uc}"] = df.loc[df['cond_class']['metric'] == uc, 
                                            ('overall', 'f1_score')].values[0]
            metrics[f"support_class_{uc}"] = df.loc[df['cond_class']['metric'] == uc, 
                                            ('overall', 'support')].values[0]

        # Extract f1_macro  an support for each conditional class and eac h class bases on conditional class
        for cond_uc in unique_cond_classes:
            metrics[f"f1_macro_cond_{cond_uc}"] = df.loc[df['cond_class']['metric'] == 'macro', 
                                                        (cond_uc, 'f1_score')].values[0]
            metrics[f"support_cond_{cond_uc}"] = df.loc[df['cond_class']['metric'] == 'macro', 
                                                        (cond_uc, 'support')].values[0]
            for uc in unique_classes:
                metrics[f"f1_macro_class_{uc}_cond_{cond_uc}"] = df.loc[df['cond_class']['metric'] == uc, 
                                                            (cond_uc, 'f1_score')].values[0]
                metrics[f"support_{uc}_cond_{cond_uc}"] = df.loc[df['cond_class']['metric'] == uc,
                                                                    (cond_uc, 'support')].values[0]
        
        return metrics