# ./experiment_utils/experiment_run.py
# Description: Manages and executes the experiment pipeline.

import logging
from pathlib import Path
from typing import List

import pandas as pd

from experiment_utils.logging_setup import setup_worker_logging
from data_utils.graph.graph import Graph
from biasing.bias_strat import BiasStrategy
from sampling.sampling_strat import SamplingStrategy
from encoding.encoding_strat import EncodingStrategy
from evaluation.evaluation_strat import EvaluationStrategy

class ExperimentRun:
    """
    Manages and executes the experiment pipeline:
      - Biasing: adjusts edge weights
      - Sampling: generates random walks from the (biased) graph
      - Encoding: trains the embedding model and saves embeddings
      - Evaluation: assesses embeddings using a chosen strategy

    The run_pipeline() method checks for an existing embedding file (and evaluation files).
      - If everything is present ("ALL_COMPLETE"), it skips the pipeline.
      - If the embedding exists but some runs are incomplete ("PARTIAL"), it only re-runs missing evaluations.
      - If no embedding exists ("NONE"), it runs the full pipeline.
    """

    def __init__(self, 
                 graph: Graph,
                 bias_strategy: BiasStrategy, 
                 sampling_strategy: SamplingStrategy, 
                 encoding_strategy: EncodingStrategy, 
                 evaluation_strategy: EvaluationStrategy, 
                 results_dir: Path):
        self.graph = graph
        self.bias_strategy = bias_strategy
        self.sampling_strategy = sampling_strategy
        self.encoding_strategy = encoding_strategy
        self.evaluation_strategy = evaluation_strategy
        self.results_dir = results_dir
        self.logger = logging.getLogger(__name__)

    def _prepare_dirs(self, experiment_graph_dir: Path, result_dir: Path, n_splits: int) -> Path:
        embedding_dir = experiment_graph_dir / "embeddings"
        embedding_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        # Create run directories
        for i in range(n_splits):
            run_dir = result_dir / f"run_{i}"
            run_dir.mkdir(parents=True, exist_ok=True)
        return embedding_dir

    def _check_existing_results(self,
                                embedding_dir: Path,
                                experiment_graph_dir: Path,
                                result_dir: Path,
                                params_signature: str,
                                n_splits: int) -> str:
        """
        Determine if the embedding + evaluations already exist.

        Returns one of:
         - "ALL_COMPLETE": Embedding file + all confusion files exist for every run.
         - "PARTIAL":      Embedding exists, but at least one run lacks confusion files.
         - "NONE":         No embedding file found.
        """
        filtered_attributes_file = experiment_graph_dir / "filtered_attributes.csv"
        if not filtered_attributes_file.exists():
            self.logger.error(f"Filtered attributes file {filtered_attributes_file} does not exist.")
            raise FileNotFoundError(f"Filtered attributes file {filtered_attributes_file} does not exist.")

        # Check if embedding already exists
        existing_embeddings = [
            f.name.replace(".emb.gz", "", 1)
            for f in embedding_dir.iterdir()
            if f.is_file() and not f.name.startswith('.') and f.name.endswith(".emb.gz")
        ]
        if params_signature in existing_embeddings:
            self.logger.info(f"Embedding {params_signature} already exists in {experiment_graph_dir.name}.")

            # Check if every run's confusion files exist
            all_evaluated = True
            for run_dir in result_dir.iterdir():
                eval_dir = run_dir / f"experiment_{params_signature}"
                # If either confusion file is missing => partial
                if not (eval_dir / "confusion_y.csv").exists() or not (eval_dir / "confusion_z.csv").exists():
                    all_evaluated = False
                    break

            if all_evaluated:
                self.logger.info("All evaluations exist for this embedding. Skipping full pipeline.")
                return "ALL_COMPLETE"
            elif params_signature in existing_embeddings:
                self.logger.info("Embedding exists but some evaluations are missing. Partial evaluation will be run.")
                return "PARTIAL"
            else:
                self.logger.error("No embedding files found. Full pipeline will be run.")
                return "NONE"

    def run_pipeline(self,
                     experiment_graph_dir: Path,
                     result_dir: Path,
                     params_signature: str,
                     n_splits: int) -> None:
        """
        Run the entire biasing/sampling/encoding/evaluation pipeline,
        or skip / partially re-run depending on the state of existing files.
        """
        self.logger.info("Starting experiment pipeline execution...")
        embedding_dir = self._prepare_dirs(experiment_graph_dir, result_dir, n_splits)

        status = self._check_existing_results(
            embedding_dir, experiment_graph_dir, result_dir, params_signature, n_splits
        )

        # ALL_COMPLETE => Nothing to do
        if status == "ALL_COMPLETE":
            return

        # PARTIAL => Only re-run missing evaluations
        elif status == "PARTIAL":
            attributes_df = pd.read_csv(experiment_graph_dir / "filtered_attributes.csv")
            # Example: Grab all relevant columns
            age_attributes = [c for c in attributes_df.columns if '_AGE' in c]
            location_attributes = [c for c in attributes_df.columns if '_region' in c]
            relevant_columns = ['user_id'] + age_attributes + location_attributes
            df_filtered_attributes = attributes_df[relevant_columns]

            embedding_file_path = embedding_dir / f"{params_signature}.emb.gz"
            for run_dir in result_dir.iterdir():
                eval_dir = run_dir / f"experiment_{params_signature}"
                missing_y = not (eval_dir / "confusion_y.csv").exists()
                missing_z = not (eval_dir / "confusion_z.csv").exists()
                if missing_y or missing_z:
                    self.logger.info(f"Running evaluation for {run_dir.name} for {params_signature}...")
                    self.evaluation_strategy.re_evaluate(
                        df_attributes=df_filtered_attributes,
                        embedding_filepath=embedding_file_path,
                        n_splits=n_splits
                    )
            return

        else:
            # NONE => No embedding found => full pipeline
            self.logger.info(f"Embedding {params_signature} not found. Running full pipeline...")
            biased_graph = self.bias_strategy.adapt_weights()
            self.logger.info("Bias strategy applied.")

            sampled_walks = self.sampling_strategy.generate_walks(biased_graph)
            self.logger.info("Random walks generated.")

            trained_model = self.encoding_strategy.fit(sampled_walks)
            self.logger.info("Model training complete.")

            embedding_file_path = self.encoding_strategy.embedding_to_file(trained_model, embedding_dir)
            self.logger.info(f"Embedding saved to {embedding_file_path}.")

            self.evaluation_strategy.evaluate(biased_graph, embedding_file_path)
            self.logger.info("Evaluation complete.")
