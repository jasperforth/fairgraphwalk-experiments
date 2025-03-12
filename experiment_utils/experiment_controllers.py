'''
./experiment_utils/experiment_controllers.py
Description: A collection of static methods that run baseline, CFN proxy, and CrossWalk experiments in parallel across multiple graphs.
Called by: experiments_main.py
Calls: ExperimentRun, NoBias, CrossWalkBias, Node2VecSampling, SkipGramEncoder, LabelPropagationEvaluation
'''
from math import log
from pathlib import Path
from joblib import Parallel, delayed
import logging

from experiment_utils.experiment_pipeline_runner import ExperimentRun
from biasing.no_bias import NoBias
from biasing.crosswalk_bias import CrossWalkBias
from sampling.node2vec_sampling import Node2VecSampling
from encoding.skipgram_encode import SkipGramEncoder
from evaluation.labelpropagation_evaluation import LabelPropagationEvaluation
from experiment_utils.logging_setup import setup_worker_logging

logger = logging.getLogger(__name__)


class ExperimentControllers:
    """
    A collection of static methods that run baseline, CFN proxy,
    and CrossWalk experiments in parallel across multiple graphs.
    """

    @staticmethod
    def run_baseline_experiments(config: dict,
                                 generated_graphs: list,
                                 resources_dir: Path,
                                 result_dir: Path,
                                 log_dir: Path):
        """
        Runs baseline Node2Vec experiments in parallel for all graphs.
        """
        exp_params = config["experiment_params"]
        logger = logging.getLogger(__name__)
        logger.info("Running baseline experiments for all graphs...")

        try:
            Parallel(n_jobs=exp_params["workers"], verbose=100)(
                delayed(ExperimentControllers._run_baseline_for_one)(
                    config, i, graph_obj, exp_graph_dir, p, q, result_dir, log_dir
                )
                for i, (graph_obj, exp_graph_dir) in enumerate(generated_graphs)
                for p in config["node2vec"]["p_values"]
                for q in config["node2vec"]["q_values"]
            )
            logger.info("Finished baseline experiments for all graphs")
        except Exception as e:
            logger.error(f"Error in baseline experiments for all graphs: {e}")

    @staticmethod
    def _run_baseline_for_one(config: dict,
                              index: int,
                              graph_obj,
                              exp_graph_dir: Path, 
                              p: float, 
                              q: float,
                              result_dir: Path, 
                              log_dir: Path):
        logger = setup_worker_logging(name=f"worker_baseline_{index}", log_dir=log_dir)
        logger.info(f"Worker initialized for baseline graph {index}")
        graph_name = f"graph_{index}"
        params_signature = f"p_{p}_q_{q}"
        n_splits = config["label_propagation"]["n_splits"]

        run_dir = result_dir / f"graph_{index}" / "baseline"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create and run the pipeline
        exp = ExperimentRun(
            graph=graph_obj,
            bias_strategy=NoBias(graph=graph_obj),
            sampling_strategy=Node2VecSampling(
                p=p, q=q,
                graph_name=graph_name,
                walk_length=config["node2vec"]["walk_length"],
                num_walks=config["node2vec"]["num_walks"],
                quiet=False,
                log_dir=log_dir
            ),
            encoding_strategy=SkipGramEncoder(
                params_signature=params_signature,
                experiment_graph_dir=exp_graph_dir,
                window=config["node2vec"]["window"],
                min_count=config["node2vec"]["min_count"],
                batch_words=config["node2vec"]["batch_words"], 
                log_dir=log_dir
            ),
            evaluation_strategy=LabelPropagationEvaluation(
                result_dir=run_dir,
                params_signature=params_signature,
                sensitive_attribute_name=config["sensitive_attribute"],
                control_attribute_name=config["control_attribute"],
                graph_name=graph_name,
                train_size=config.get("label_propagation", {}).get("train_size", 0.5), 
                log_dir=log_dir
            ),
            results_dir=run_dir,
        )
        exp.run_pipeline(
            experiment_graph_dir=exp_graph_dir,
            result_dir=run_dir,
            params_signature=params_signature,
            n_splits=n_splits
        )
        logger.info(f"Completed baseline run for graph {index} (p={p}, q={q})")

    @staticmethod
    def run_cfn_proxy(config: dict,
                      generated_graphs: list,
                      resources_dir: Path,
                      result_dir: Path, 
                      log_dir: Path):
        """
        CFN proxy calculation for CrossWalk. We run in parallel for each graph
        and for both the 'sensitive_attribute' and 'control_attribute'.
        """
        exp_params = config["experiment_params"]
        logger = logging.getLogger(__name__)
        logger.info("Running CFN proxy calculation for all graphs...")
        prewalk_length = config["crosswalk"]["prewalk_length"]
        
        try:
            Parallel(n_jobs=exp_params["workers"], verbose=100)(
                delayed(ExperimentControllers._compute_cfn_for_one)(
                    config, i, graph_obj, exp_graph_dir, sens, prewalk_length, log_dir
                )
                for i, (graph_obj, exp_graph_dir) in enumerate(generated_graphs)
                for sens in [config["sensitive_attribute"], config["control_attribute"]]
            )
            logger.info("Finished CFN proxy calculation for all graphs")
        except Exception as e:
            logger.error(f"Error in CFN proxy calculation for all graphs: {e}")

    @staticmethod
    def _compute_cfn_for_one(config: dict,
                             index: int,
                             graph_obj,
                             exp_graph_dir: Path,
                             sens_attr: str,
                             prewalk_length: int, 
                             log_dir: Path):
        logger = setup_worker_logging(name=f"cfn_worker_{index}", log_dir=log_dir)
        logger.info(f"Worker initialized for CFN graph {index})")
        graph_name = f"graph_{index}"
        CrossWalkBias(
            graph=graph_obj,
            experiment_graph_dir=exp_graph_dir,
            sensitive_attribute_name=sens_attr,
            graph_name=graph_name,
            prewalk_length=prewalk_length,
            log_dir=log_dir
        ).pre_compute_biasing_params()
        logger.info(f"Completed CFN proxy calculation for graph {index} ({sens_attr})")

    @staticmethod
    def run_crosswalk_experiments(config: dict,
                                  generated_graphs: list,
                                  resources_dir: Path,
                                  result_dir: Path, 
                                  log_dir: Path):
        """
        Runs CrossWalk experiments in parallel for all graphs, exchanging
        sensitive/control attributes as well.
        """
        exp_params = config["experiment_params"]
        logger = logging.getLogger(__name__)
        logger.info("Running CrossWalk experiments for all graphs with sensitive/control exchanges...")

        try:
            Parallel(n_jobs=exp_params["workers"], verbose=100)(
                delayed(ExperimentControllers._run_crosswalk_for_one)(
                    config, i, graph_obj, exp_graph_dir, sens, control, alpha, exponent, p, q, result_dir, log_dir
                )
                for i, (graph_obj, exp_graph_dir) in enumerate(generated_graphs)
                for (sens, control) in [
                    (config["sensitive_attribute"], config["control_attribute"]),
                    (config["control_attribute"], config["sensitive_attribute"])
                ]
                for alpha in config["crosswalk"]["alphas"]
                for exponent in config["crosswalk"]["exponents"]
                for p in config["node2vec"]["p_values"]
                for q in config["node2vec"]["q_values"]
            )
            logger.info("Finished CrossWalk experiments for all graphs")
        except Exception as e:
            logger.error(f"Error in CrossWalk experiments for all graphs: {e}")

    @staticmethod
    def _run_crosswalk_for_one(config: dict,
                               index: int,
                               graph_obj,
                               exp_graph_dir: Path,
                               sens: str,
                               control: str,
                               alpha: float,
                               exponent: float,
                               p: float,
                               q: float,
                               result_dir: Path, 
                               log_dir: Path
                               ):
        """
        Single CrossWalk experiment run for one combination of graph / params / attributes.
        """
        logger = setup_worker_logging(name=f"crosswalk_worker_{index}", log_dir=log_dir)
        logger.info(f"Worker initialized for crosswalk graph {index}")
        graph_name = f"graph_{index}"
        params_signature = (f"prewalk_{config['crosswalk']['prewalk_length']}_"
                            f"alpha_{alpha}_exponent_{exponent}_p_{p}_q_{q}_sens_{sens}_control_{control}")
        n_splits = config["label_propagation"]["n_splits"]

        run_dir = result_dir / f"graph_{index}" / "crosswalk"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create and run the pipeline
        ExperimentRun(
            graph=graph_obj,
            bias_strategy=CrossWalkBias(
                graph=graph_obj,
                experiment_graph_dir=exp_graph_dir,
                sensitive_attribute_name=sens,
                alpha=alpha,
                exponent=exponent,
                graph_name=graph_name,
                prewalk_length=config["crosswalk"]["prewalk_length"],
                log_dir=log_dir
            ),
            sampling_strategy=Node2VecSampling(
                p=p, q=q,
                graph_name=graph_name,
                walk_length=config["node2vec"]["walk_length"],
                num_walks=config["node2vec"]["num_walks"],
                quiet=False,
                log_dir=log_dir
            ),
            encoding_strategy=SkipGramEncoder(
                params_signature=params_signature,
                experiment_graph_dir=exp_graph_dir,
                window=config["node2vec"]["window"],
                min_count=config["node2vec"]["min_count"],
                batch_words=config["node2vec"]["batch_words"],
                log_dir=log_dir
            ),
            evaluation_strategy=LabelPropagationEvaluation(
                result_dir=run_dir,
                params_signature=params_signature,
                sensitive_attribute_name=sens,
                control_attribute_name=control,
                graph_name=graph_name,
                train_size=config.get("label_propagation", {}).get("train_size", 0.5),
                log_dir=log_dir
            ),
            results_dir=run_dir,
        ).run_pipeline(
            experiment_graph_dir=exp_graph_dir,
            result_dir=run_dir,
            params_signature=params_signature,
            n_splits=n_splits
        )
        logger.info(f"Completed CrossWalk run for graph {index} (sens={sens}, control={control}, alpha={alpha}, exponent={exponent}, p={p}, q={q})")
