from pathlib import Path
import pandas as pd
import logging
from typing import List, Tuple

from data_utils import Graph
from biasing import BiasStrategy, bias_strat
from sampling import SamplingStrategy
from encoding import EncodingStrategy
from evaluation import EvaluationStrategy
from experiment_utils.logging_utils import setup_worker_logging
from experiment_utils.config import DATA_DIR

# Configure logging
log_dir = DATA_DIR
logger = setup_worker_logging("experiment_run", log_dir)

class ExperimentRun:
    """
    Class to manage and execute the experiment pipeline, including biasing, sampling, encoding, and evaluation.
    """
    def __init__(self, 
                 graph: Graph,
                 bias_strategy: BiasStrategy, 
                 sampling_strategy: SamplingStrategy, 
                 encoding_strategy: EncodingStrategy, 
                 evaluation_strategy: EvaluationStrategy, 
                 results_dir: Path):       #self.data_formatter = data_formatter
        self.graph = graph
        self.bias_strategy = bias_strategy
        self.sampling_strategy = sampling_strategy
        self.encoding_strategy = encoding_strategy
        self.evaluation_strategy = evaluation_strategy
        self.results_path = results_dir

    def re_run_embeddings(self, 
                          experiment_graph_dir: Path,
                          result_dir: Path,
                          params_signature: str,
                          n_splits: int,
                          ) -> None:
        """
        Runs the embedding process with the same parameters if embeddings do not already exist.
        
        Args:
            experiment_graph_dir (Path): Directory of the experiment graph.
            result_dir (Path): Directory to save the results.
            params_signature (str): Parameter signature for naming.
            n_splits (int): Number of splits for evaluation.
        """
        logger.info('Starting (re-)run of embeddings')
        
        embedding_dir = experiment_graph_dir / "embeddings"
        embedding_dir.mkdir(parents=True, exist_ok=True)
        
        result_dir.mkdir(parents=True, exist_ok=True)

        # Ensure only valid directories are considered for runs
        valid_run_dirs = [d for d in result_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

        if len(valid_run_dirs) < n_splits:
            for f in range(n_splits):
                run_dir = result_dir / f"run_{f}"
                run_dir.mkdir(parents=True, exist_ok=True)

        # if len([f for f in result_dir.iterdir() if f.name.startswith("run")]) < n_splits:
        #     for f in range(n_splits):
        #         run_dir = result_dir / f"run_{f}"
        #         run_dir.mkdir(parents=True, exist_ok=True)

        filtered_attributes_file = experiment_graph_dir / "filtered_attributes.csv"   

        if not filtered_attributes_file.exists():
            logger.error(f"Filtered attributes file {filtered_attributes_file} does not exist.")
            raise FileNotFoundError(f"Filtered attributes file {filtered_attributes_file} does not exist.")
        
        # Load the attribute DataFrame and filter relevant columns
        attributes_df = pd.read_csv(filtered_attributes_file)
        
        # Extract relevant attributes from columns
        age_attributes = [col for col in attributes_df.columns if '_AGE' in col]
        location_attributes = [col for col in attributes_df.columns if '_region' in col]
        
        # Combine all relevant attributes
        relevant_columns = ['user_id'] + age_attributes + location_attributes
        
        # Filter the DataFrame to only include relevant columns
        df_filtered_attributes = attributes_df[relevant_columns]

        existing_embeddings = [
            f.name.replace(".emb.gz", "", 1) 
            for f in embedding_dir.iterdir() 
            if f.is_file() and not f.name.startswith('.') and f.name.endswith(".emb.gz")
        ]

        if params_signature in existing_embeddings:
            logger.info(f"Embedding {params_signature} for {experiment_graph_dir.name} already exists. Skipping...")
            embedding_file_path = embedding_dir / f"{params_signature}.emb.gz"
            for run_dir in result_dir.iterdir():         
                if (run_dir / f"experiment_{params_signature}" / "confusion_y.csv").exists() \
                    and (run_dir / f"experiment_{params_signature}" / "confusion_z.csv").exists():
                        logger.info(f"Results for {run_dir.name} for {params_signature} already exist. Skipping evaluation...")
                else:
                    logger.info(f"Running evaluation for {run_dir.name} for {params_signature}...")
                    self.evaluation_strategy.re_evaluate(df_attributes=df_filtered_attributes, 
                                                         embedding_filepath=embedding_file_path, 
                                                         n_splits=n_splits)

        else:
            logger.info(f"Embedding {params_signature} for {experiment_graph_dir.name} does not exist. Running experiment...")
            # Apply the bias strategy to adapt the graph edge weights.
            biased_graph = self.bias_strategy.adapt_weights()  
            logger.info(f'Bias strategy applied')

            # Generate random walks using the sampling strategy on the biased graph.
            sampled_walks = self.sampling_strategy.generate_walks(biased_graph)
            logger.info('Random walks generated')

            # Train the embedding model using the encoding strategy on the sampled walks.
            trained_model = self.encoding_strategy.fit(sampled_walks)
            logger.info('Encoding model training complete')

            # Save the trained embedding to a file.
            embedding_file_path = self.encoding_strategy.embedding_to_file(trained_model, embedding_dir)
            logger.info(f'Embedding saved to {embedding_file_path}')

            # Evaluate the embedding model using the evaluation strategy.
            eval_ = self.evaluation_strategy.evaluate(biased_graph, embedding_file_path)
            logger.info('Evaluation complete')
