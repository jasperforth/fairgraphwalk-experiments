import logging
import sys
from pathlib import Path
import random
import numpy as np
from joblib import Parallel, delayed
from datetime import datetime

# testing sparse
# from biasing.FMMC_bias_sparse import FMMCBias

from biasing.FMMC_bias import FMMCBias

# Add the parent directory of the current file to the system path for module imports.
file = Path(__file__).resolve()
prent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from experiment_utils.config import (
    RAW_ATTRIBUTES_FILEPATH, EDGELIST_FILEPATH,
    FORMATTED_DATA_DIR, DATA_DIR, SEED, FILTER_CATEGORY, COLUMN_NAMES,
    ATTRIBUTES, SPLITPOINTS_BINS, P_VALUES, Q_VALUES, ALPHAS, EXPONENTS,
    WORKERS, N_SPLITS, SENS_ATTR_NAME, OTHER_ATTR_NAME, PREWALK_LENGTH,
    SELFLOOPS
)
from experiment_utils import ExperimentRun
from biasing.no_bias import NoBias
from biasing.crosswalk_bias import CrossWalkBias
from data_utils.formatting.pokec_formatter import PokecFormatter
from data_utils.graph.pokec_graph import PokecGraph
from sampling.node2vec_sampling import Node2VecSampling
from encoding.skipgram_encode import SkipGramEncoder
from evaluation.labelpropagation_evaluation import LabelPropagationEvaluation

log_dir = DATA_DIR
logger = logging.getLogger(__name__)


def create_directory(path):
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


def new_run_experiment_pokec_mixed(graphsize: str):
    # Set up directories and seeds for the experiment.
    experiment_name = f"new_pokec_mixed_graphsize_{graphsize}"
    formatted_data_dir = FORMATTED_DATA_DIR
    experiment_dir = DATA_DIR / experiment_name
    resources_dir = experiment_dir / 'resources'
    result_dir = experiment_dir / 'results'

    for dir_path in [formatted_data_dir, experiment_dir, resources_dir, result_dir]:
        create_directory(dir_path)

    seed = SEED
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f'Seed set to {seed}')

    # Pre-format data if necessary
    try:
        files = list(formatted_data_dir.glob("*pre_formatted*"))
        specific_files = [file for file in files if file.name.startswith("pre_formatted")]
        if len(specific_files) < 2:
            logger.info('Pre-formatting data')
            PokecFormatter().pre_formatting(attribute_filepath=RAW_ATTRIBUTES_FILEPATH,
                                            edgelist_filepath=EDGELIST_FILEPATH,
                                            formatted_data_dir=formatted_data_dir,
                                            filter_category=FILTER_CATEGORY,
                                            column_names=COLUMN_NAMES,
                                            attributes=ATTRIBUTES,
                                            )
        else:
            logger.info('Data already pre-formatted')

    except Exception as e:
        logger.error(f"Error in pre-formatting data: {e}")
        return

    # sys.exit()

    # Define subgraph categories. Also the location attribute.
    # For distinct regions.
    if graphsize == "vsmall1": #2
        region_categories = {
            # 939 at 91%
            'graph_dir_0': ["ceska republika, cz - kralovehradecky kraj",
                            "ceska republika, cz - karlovarsky kraj",
                            "ceska republika, cz - vysocina", ],
        }
    elif graphsize == "vsmall2":
        region_categories = {
            # 512 at 99,9 (1 connecting edge)%
            'graph_dir_0': ["trnavsky kraj, sastin-straze",
                            "ceska republika, cz - vysocina",
                            "bratislavsky kraj, svaty jur", ],
        }
    elif graphsize == "small1": #4
        region_categories = {
            # 2276 at 96%
            'graph_dir_0': ["presovsky kraj, velky saris",
                            "presovsky kraj, vysoke tatry",
                            "presovsky kraj, podolinec", ],
        }
    elif graphsize == "small2":
        region_categories = {
            # 6514 edges 68%
            'graph_dir_0': ["bratislavsky kraj, bratislava - vrakuna",
                            "bratislavsky kraj, bratislava - raca",
                            "bratislavsky kraj, bratislava - podunaj. biskupice", ],
        }
    elif graphsize == "large1": #6
        region_categories = {
            # 36688 edges 93%
            'graph_dir_0': ['banskobystricky kraj, detva',
                            'banskobystricky kraj, filakovo',
                            'presovsky kraj, spisska bela',
                            'bratislavsky kraj, bratislava - dubravka'],
        }
    elif graphsize == "large2":
        region_categories = {
            # 69151 edges 60%
            'graph_dir_0': ["trnavsky kraj, holic",
                            "trnavsky kraj, velky meder",
                            "trnavsky kraj, gbely",
                            "presovsky kraj, podolinec",
                            "presovsky kraj, spisske podhradie",
                            "presovsky kraj, spisska stara ves",
                            "presovsky kraj, spisska bela",
                            "trenciansky kraj, dubnica nad vahom",
                            "trenciansky kraj, nova dubnica", ],
        }
    elif graphsize == "large3": #8
        region_categories = {
            # 26339 edges 51%
            'graph_dir_0': ["bratislavsky kraj, bratislava - ruzinov",
                            "bratislavsky kraj, bratislava - stare mesto",
                            "bratislavsky kraj, bratislava - nove mesto",
                            "bratislavsky kraj, bratislava - vajnory",
                            "bratislavsky kraj, bratislava - vrakuna"],
        }

    # Format the data for each subgraph.
    for category_name, dataset in region_categories.items():
        experiment_graph_dir = resources_dir / category_name
        experiment_graph_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Formatting data for {category_name}')
        try:
            PokecFormatter().formatting(formatted_data_dir=formatted_data_dir,
                                        edgelist_filepath=EDGELIST_FILEPATH,
                                        experiment_path=experiment_graph_dir,
                                        filter_category=FILTER_CATEGORY,
                                        category_values=dataset,
                                        attributes=ATTRIBUTES,
                                        splitpoints_bins=SPLITPOINTS_BINS,  # Splitpoints for age attribute
                                        )
            logger.info(f'Finished formatting data for {category_name}')
        except Exception as e:
            logger.error(f"Error formatting data for {category_name}: {e}")
            return

    # Create a list of tuples with the paths to the edgelist and attributes files for each subgraph.
    data_tuples = []
    for subfolder in resources_dir.iterdir():
        if subfolder.is_dir():
            edgelist_file, attributes_file = None, None
            for file in subfolder.iterdir():
                if file.name == 'filtered_edgelist.txt':
                    edgelist_file = file
                elif file.name == 'filtered_attributes.csv':
                    attributes_file = file
            if edgelist_file and attributes_file:
                data_tuples.append((edgelist_file, attributes_file))
            else:
                logger.error("No edgelist or attributes file found in subdirectory.")
                raise ValueError("No edgelist or attributes file found in subdirectory.")

    sorted_data_tuples = sorted(data_tuples, key=lambda x: int(x[0].parts[-2][-1]))

    # Generates a list of tuple containing graph objects and corresponding save directories.
    generated_pokec_subgraphs_directories = [
        (PokecGraph.graph_from_edgelist(tuple[0], tuple[1]), resources_dir / f"graph_dir_{i}")
        for i, tuple in enumerate(sorted_data_tuples)]

    for i, (graph, dir_path) in enumerate(generated_pokec_subgraphs_directories):
        logger.info(f"Graph {i}: {graph}")
        logger.info(f"Number of nodes in graph {i}: {len(graph.graph)}")
        logger.info(f"Number of edges in graph {i}: {len(graph.graph.edges())}")
        logger.info(f"Number of attributes in graph {i}: {len(graph.attributes)}")

    bl_total = len(generated_pokec_subgraphs_directories) * len(P_VALUES) * len(Q_VALUES)
    logger.info(f"Total number of baseline experiments: {bl_total}")
    cw_total = (len(generated_pokec_subgraphs_directories) * len(P_VALUES) * len(Q_VALUES) *
                len(ALPHAS) * len(EXPONENTS) * 2)
    logger.info(f"Total number of crosswalk experiments: {cw_total}")

    def run_baseline_experiment(graph: PokecGraph,
                                experiment_graph_dir: Path,
                                p: float, q: float,
                                graph_name: str,
                                sensitive_attribute_name: str,
                                other_attribute_name: str,
                                params_signature: str,
                                result_dir: Path,
                                walk_length: int = 80,
                                num_walks: int = 10,
                                train_size: float = 0.5,
                                window: int = 10,
                                min_count: int = 1,
                                batch_words: int = 4,
                                quiet: bool = False,
                                ):

        experiment_graph_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        ExperimentRun(graph=graph,
                      bias_strategy=NoBias(
                          graph=graph),
                      sampling_strategy=Node2VecSampling(
                          p=p, q=q,
                          graph_name=graph_name,
                          walk_length=walk_length,
                          num_walks=num_walks,
                          quiet=quiet),
                      encoding_strategy=SkipGramEncoder(
                          params_signature=params_signature,
                          experiment_graph_dir=experiment_graph_dir,
                          window=window,
                          min_count=min_count,
                          batch_words=batch_words),
                      evaluation_strategy=LabelPropagationEvaluation(
                          result_dir=result_dir,
                          params_signature=params_signature,
                          sensitive_attribute_name=sensitive_attribute_name,
                          other_attribute_name=other_attribute_name,
                          graph_name=graph_name,
                          train_size=train_size),
                      results_dir=result_dir,
                      ).re_run_embeddings(experiment_graph_dir=experiment_graph_dir,
                                          result_dir=result_dir,
                                          params_signature=params_signature,
                                          n_splits=N_SPLITS,
                                          )

    def calculate_cfn(graph: PokecGraph,
                      experiment_graph_dir: Path,
                      sensitive_attribute_name: str,
                      graph_name: str,
                      prewalk_length: int = 6):
        CrossWalkBias(graph=graph, experiment_graph_dir=experiment_graph_dir,
                      sensitive_attribute_name=sensitive_attribute_name,
                      graph_name=graph_name,
                      prewalk_length=prewalk_length).pre_compute_biasing_params()

    def run_crosswalk_experiment(graph: PokecGraph,
                                 experiment_graph_dir: Path,
                                 alpha: float,
                                 exponent: float,
                                 p: float, q: float,
                                 graph_name: str,
                                 sensitive_attribute_name: str,
                                 other_attribute_name: str,
                                 params_signature: str,
                                 result_dir: Path,
                                 prewalk_length: int = 6,
                                 walk_length: int = 80,
                                 num_walks: int = 10,
                                 train_size: float = 0.5,
                                 window: int = 10,
                                 min_count: int = 1,
                                 batch_words: int = 4,
                                 quiet: bool = False,
                                 ):
        experiment_graph_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        ExperimentRun(graph=graph,
                      bias_strategy=CrossWalkBias(
                          graph=graph, experiment_graph_dir=experiment_graph_dir,
                          sensitive_attribute_name=sensitive_attribute_name,
                          alpha=alpha, exponent=exponent,
                          graph_name=graph_name,
                          prewalk_length=prewalk_length),
                      sampling_strategy=Node2VecSampling(
                          p=p, q=q,
                          graph_name=graph_name,
                          walk_length=walk_length,
                          num_walks=num_walks,
                          quiet=quiet),
                      encoding_strategy=SkipGramEncoder(
                          params_signature=params_signature,
                          experiment_graph_dir=experiment_graph_dir,
                          window=window,
                          min_count=min_count,
                          batch_words=batch_words),
                      evaluation_strategy=LabelPropagationEvaluation(
                          result_dir=result_dir,
                          params_signature=params_signature,
                          sensitive_attribute_name=sensitive_attribute_name,
                          other_attribute_name=other_attribute_name,
                          graph_name=graph_name,
                          train_size=train_size),
                      results_dir=result_dir,
                      ).re_run_embeddings(experiment_graph_dir=experiment_graph_dir,
                                          result_dir=result_dir,
                                          params_signature=params_signature,
                                          n_splits=N_SPLITS
                                          )

    def run_fmmc_experiment(graph: PokecGraph,
                            experiment_graph_dir: Path,
                            selfloops: bool,
                            p: float, q: float,
                            graph_name: str,
                            sensitive_attribute_name: str,
                            other_attribute_name: str,
                            params_signature: str,
                            result_dir: Path,
                            prewalk_length: int = 6,
                            walk_length: int = 80,
                            num_walks: int = 10,
                            train_size: float = 0.5,
                            window: int = 10,
                            min_count: int = 1,
                            batch_words: int = 4,
                            quiet: bool = False,
                            ):
        experiment_graph_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        ExperimentRun(graph=graph,
                      bias_strategy=FMMCBias(graph=graph, experiment_graph_dir=experiment_graph_dir,
                                             sensitive_attribute_name=sensitive_attribute_name, graph_name=graph_name,
                                             prewalk_length=prewalk_length,selfloops=selfloops),
                      sampling_strategy=Node2VecSampling(
                          p=p, q=q,
                          graph_name=graph_name,
                          walk_length=walk_length,
                          num_walks=num_walks,
                          quiet=quiet),
                      encoding_strategy=SkipGramEncoder(
                          params_signature=params_signature,
                          experiment_graph_dir=experiment_graph_dir,
                          window=window,
                          min_count=min_count,
                          batch_words=batch_words),
                      evaluation_strategy=LabelPropagationEvaluation(
                          result_dir=result_dir,
                          params_signature=params_signature,
                          sensitive_attribute_name=sensitive_attribute_name,
                          other_attribute_name=other_attribute_name,
                          graph_name=graph_name,
                          train_size=train_size),
                      results_dir=result_dir,
                      ).re_run_embeddings(experiment_graph_dir=experiment_graph_dir,
                                          result_dir=result_dir,
                                          params_signature=params_signature,
                                          n_splits=N_SPLITS
                                          )

    # # Compute embeddings for selected graphs
    # for graph_nr, (graph, experiment_graph_dir) in enumerate(generated_pokec_subgraphs_directories):
    #     logger.info(f'Running baseline experiments for graph {graph_nr}')
    #     # run baseline

    # run FMMC without parallelization
    for graph_nr, (graph, experiment_graph_dir) in enumerate(generated_pokec_subgraphs_directories):
        for selfloops in SELFLOOPS:
            logger.info(f'Running FMMC experiments for graph {graph_nr} with selfloops {selfloops}')
            try:
                for p in P_VALUES:
                    for q in Q_VALUES:
                        run_fmmc_experiment(
                            graph=graph,
                            experiment_graph_dir=experiment_graph_dir,
                            p=p,
                            q=q,
                            graph_name=f"graph_{graph_nr}",
                            sensitive_attribute_name=SENS_ATTR_NAME,
                            other_attribute_name=OTHER_ATTR_NAME,
                            params_signature=f"p_{p}_q_{q}_{graphsize}_selfloops_{selfloops}",
                            result_dir=result_dir / f"graph_{graph_nr}" / "fmmc",
                            quiet=False,
                            selfloops=selfloops,
                        )
                logger.info(f'Finished FMMC experiments for graph {graph_nr} with selfloops {selfloops}')
            except Exception as e:
                logger.error(f"Error running FMMC experiments for graph {graph_nr} with selfloops {selfloops}: {e}")
    #####

    logger.info(f'Running baseline experiments for all graphs')
    try:
        Parallel(n_jobs=WORKERS, verbose=100)(
                delayed(run_baseline_experiment)(
                            graph=graph, experiment_graph_dir=experiment_graph_dir,
                            p=p, q=q,
                            graph_name = f"graph_{graph_nr}",
                            sensitive_attribute_name=SENS_ATTR_NAME,
                            other_attribute_name=OTHER_ATTR_NAME,
                            params_signature=f"p_{p}_q_{q}",
                            result_dir=result_dir / f"graph_{graph_nr}" / "baseline",
                            quiet=False,
                            )
                    for graph_nr, (graph, experiment_graph_dir) in enumerate(generated_pokec_subgraphs_directories)
                        for p in P_VALUES
                            for q in Q_VALUES)
        logger.info(f'Finished baseline n2v experiments for all graphs')
    except Exception as e:
        logger.error(f"Error running baseline experiments for all graphs: {e}")

    logger.info(f'Running CFN proxy calculation for all graphs')
    # precompute biasing prxy params cfn
    try:
        Parallel(n_jobs=WORKERS, verbose=100)(
                delayed(calculate_cfn)(
                            graph=graph, experiment_graph_dir=experiment_graph_dir,
                            sensitive_attribute_name=sens,
                            graph_name = f"graph_{graph_nr}",
                            prewalk_length= PREWALK_LENGTH)
                    for graph_nr, (graph, experiment_graph_dir) in enumerate(generated_pokec_subgraphs_directories)
                        for sens in [SENS_ATTR_NAME, OTHER_ATTR_NAME])
        logger.info(f'Finished CFN proxy calculation for all graphs')
    except Exception as e:
        logger.error(f"Error running CFN proxy calculation for all graphs: {e}")

    # run crosswalk with progression updates on all combinations of parameters
    # run for sens, other as sens, other and vice versa
    logger.info(f'Running CrossWalk for all graphs with sensitive attribute and other attribute forth and back')
    try:
        Parallel(n_jobs=WORKERS, verbose=100)(
                delayed(run_crosswalk_experiment)(
                            graph=graph, experiment_graph_dir=experiment_graph_dir,
                            alpha=alpha, exponent=exponent,
                            p=p, q=q,
                            graph_name = f"graph_{graph_nr}",
                            sensitive_attribute_name=sens,
                            other_attribute_name=other,
                            params_signature=f"prewalk_{PREWALK_LENGTH}_alpha_{alpha}_exponent_{exponent}_p_{p}_q_{q}_sens_{sens}_other_{other}",
                            result_dir=result_dir / f"graph_{graph_nr}" / "crosswalk",
                            prewalk_length=PREWALK_LENGTH,
                            quiet=False,
                            )
                    for graph_nr, (graph, experiment_graph_dir) in enumerate(generated_pokec_subgraphs_directories)
                        for sens, other in [(SENS_ATTR_NAME, OTHER_ATTR_NAME), (OTHER_ATTR_NAME, SENS_ATTR_NAME)]
                            for alpha in ALPHAS
                                for exponent in EXPONENTS
                                    for p in P_VALUES
                                        for q in Q_VALUES)
        logger.info(f'Finished CrossWalk for all graphs')
    except Exception as e:
        logger.error(f"Error running CrossWalk for all graphs: {e}")

