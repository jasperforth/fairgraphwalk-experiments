#!/usr/bin/env python
# File: experiments/experiment_main.py
# Description: Unified experiment pipeline entry point (single source of truth).
#   - Loads configuration (config.yml) and region categories (region_categories.json).
#   - Prepares directories and formats data.
#   - Constructs graph objects and invokes ExperimentControllers for baseline, CFN, and CrossWalk runs.
# Usage:
#   python experiment_main.py --config config.yml [--experiment_to_run <mode_override>]

import argparse
import json
import logging
from math import log
import random
import sys
from pathlib import Path
import yaml
import numpy as np

print(f"Python executable: {sys.executable}")
# Add parent directory for module imports.
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from experiment_utils.logging_setup import setup_main_logging, setup_worker_logging
from experiment_utils.experiment_controllers import ExperimentControllers
from data_utils.formatting.pokec_formatter import PokecFormatter
from data_utils.graph.pokec_graph import PokecGraph

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_region_categories(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)

def create_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise

def main():
    """
    Main entry point for the unified experiment pipeline.
    Parses command-line arguments, loads config, optionally overrides experiment lists
    and base_dir, and runs the pipeline (baseline + crosswalk).
    """
    parser = argparse.ArgumentParser(description="Unified Experiment Pipeline")
    parser.add_argument("--config", type=str, default="experiments/config.yml",
                        help="Path to YAML configuration")
    parser.add_argument("--experiment_to_run", type=str, default="both",
                        help="Choose: 'baseline', 'crosswalk', or 'both'")
    parser.add_argument("--experiment_id", type=int, default=None,
                        help="If provided, override config['experiments'] with the single experiment at this index. "
                             "Useful for Slurm array jobs.")
    parser.add_argument("--demo_subset", action="store_true",
                        help="If set, override config['experiments'] with a smaller subset (e.g., for quick demos).")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Override the base_dir in the config.")
    args = parser.parse_args()

    # Load the config
    config = load_yaml_config(args.config)
    # Save the config file's directory to resolve relative paths later.
    config["_config_dir"] = Path(args.config).parent

    # Optionally override the base_dir if provided as an argument
    if args.base_dir is not None:
        old_base = config.get("base_dir", None)
        config["base_dir"] = args.base_dir
        logger.info(f"Overriding base_dir: '{old_base}' -> '{args.base_dir}'")

    # Optionally override experiment list by HPC Slurm array index
    if args.experiment_id is not None:
        all_exps = config.get("experiments", [])
        if args.experiment_id < 0 or args.experiment_id >= len(all_exps):
            raise ValueError(f"Invalid experiment_id {args.experiment_id}, must be in range [0, {len(all_exps)-1}].")
        # Keep only the selected experiment
        selected_experiment = all_exps[args.experiment_id]
        config["experiments"] = [selected_experiment]
        logger.info(f"Overriding experiments with single experiment at index {args.experiment_id}: {selected_experiment}")

    # Optionally override to a "demo" subset if flagged
    if args.demo_subset:
        # For example, override with a smaller set of experiments
        # e.g., only run ["distinct_demo", "semi_demo"] 
        # or something minimal for a quick test
        config["experiments"] = ["demo_distinct", "demo_semi"]
        logger.info("Using a reduced demo subset of experiments: ['demo_distinct', 'demo_semi']")

    # Call the main runner
    run_experiment(config, exp_run_mode=args.experiment_to_run)


def run_experiment(config: dict, exp_run_mode: str = "both") -> None:
    # The full pipeline logic: sets up logging, data directories, formatting, runs baseline & crosswalk, etc.
    log_dir = Path(config["data_dir"].format(
        base_dir=config["base_dir"], project_name=config["project_name"]
    )) / "logs"
    logger = setup_main_logging(log_dir)
    logger.info("Main logging active.")

    # Expand environment paths
    base_dir = config["base_dir"]
    project_name = config["project_name"]
    data_dir = Path(config["data_dir"].format(base_dir=base_dir, project_name=project_name))
    raw_dir = Path(config["raw_dir"].format(base_dir=base_dir))

    # --- Resolve the region_categories.json path relative to the config file ---
    config_dir = config.get("_config_dir", Path("."))
    filter_categories_file = config["filter_categories_file"]
    filter_categories_path = Path(filter_categories_file)
    if not filter_categories_path.is_absolute():
        filter_categories_path = config_dir / filter_categories_file
    region_categories_all = load_region_categories(str(filter_categories_path))
    logger.info(f"Loaded region categories from: {filter_categories_path}")

    # Get experiment modes 
    experiments_to_run = config.get("experiments", [])
    exp_params = config["experiment_params"]
    fmt_params = config["formatter_params"]
    logger.info(f"Experiments to run: {experiments_to_run}")

    # Raw data paths
    raw_attributes_file = raw_dir / config["raw_attributes_file"]
    edgelist_file = raw_dir / config["edgelist_file"]

    # Directory for pre-formatted data
    formatted_data_dir = data_dir / "formatted"
    create_directory(formatted_data_dir)

    # Create top-level directories for each experiment mode
    for exp_mode in experiments_to_run:
        exp_output_dir = data_dir / f"{project_name}_{exp_mode}"
        for subdir in [exp_output_dir, exp_output_dir / "resources", exp_output_dir / "results"]:
            create_directory(subdir)

    # Seed.
    seed = exp_params.get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Seed set to {seed}")

    # Pre-format data if not already available
    preformatted_files = list(formatted_data_dir.glob("*pre_formatted*"))
    if len(preformatted_files) < 2:
        logger.info("Pre-formatting data...")
        PokecFormatter().pre_formatting(
            attribute_filepath=str(raw_attributes_file),
            formatted_data_dir=formatted_data_dir,
            filter_category=fmt_params["filter_category_key"],
            column_names=fmt_params["column_names"],
            attributes=fmt_params["attributes"]
        )
    else:
        logger.info("Data already pre-formatted.")

    # Process each experiment mode
    for exp_mode in experiments_to_run:
        logger.info(f"Starting experiment mode: {exp_mode}")
        filter_config = config["filter"].get(exp_mode, {})
        filter_type = filter_config.get("type", "full")

        if filter_type == "region":
            region_key = filter_config.get("region_categories_key")
            if region_key not in region_categories_all:
                logger.error(f"Region key '{region_key}' not found in JSON file.")
                continue
            region_categories = region_categories_all[region_key]
        elif filter_type == "full":
            region_categories = {"full": []}
        else:
            logger.error(f"Unsupported filter type: {filter_type}")
            continue

        exp_output_dir = data_dir / f"{project_name}_{exp_mode}"
        resources_dir = exp_output_dir / "resources"
        result_dir = exp_output_dir / "results"

        # Format data for each subgraph category.
        for category_name, category_values in region_categories.items():
            exp_graph_dir = resources_dir / category_name
            create_directory(exp_graph_dir)
            logger.info(f"Formatting data for category '{category_name}'...")
            try:
                PokecFormatter().formatting(
                    formatted_data_dir=formatted_data_dir,
                    edgelist_filepath=str(edgelist_file),
                    experiment_path=exp_graph_dir,
                    filter_category=fmt_params["filter_category_key"],
                    category_values=category_values,
                    attributes=fmt_params["attributes"],
                    splitpoints_bins=fmt_params["splitpoints_bins"]
                )
                logger.info(f"Finished formatting data for '{category_name}'.")
            except Exception as e:
                logger.error(f"Error formatting data for '{category_name}': {e}")
                continue

        # Collect formatted subgraph files.
        data_tuples = []
        for subfolder in resources_dir.iterdir():
            if subfolder.is_dir():
                edgelist_f, attributes_f = None, None
                for f in subfolder.iterdir():
                    if f.name == "filtered_edgelist.txt":
                        edgelist_f = f
                    elif f.name == "filtered_attributes.csv":
                        attributes_f = f
                if edgelist_f and attributes_f:
                    data_tuples.append((edgelist_f, attributes_f))
                else:
                    logger.error(f"Missing formatted files in {subfolder}")
                    continue

        # sorted_data_tuples = sorted(data_tuples, key=lambda x: x[0].parent.name)
        # generated_graphs = [
        #     (PokecGraph.graph_from_edgelist(str(t[0]), str(t[1])), resources_dir / f"graph_dir_{i}")
        #     for i, t in enumerate(sorted_data_tuples)
        # ]

        # for full graph mode, we don't need to sort the data tuples
        # TODO test this for filtered graph modes
        sorted_data_tuples = sorted(data_tuples, key=lambda x: x[0].parent.name)

        # Decide on the experiment graph directory based on the filter type.
        if filter_type == "full":
            # In full mode, the filtered files were written to a folder (e.g., "full")
            # so we use that folder directly.
            generated_graphs = [
                (PokecGraph.graph_from_edgelist(str(t[0]), str(t[1])), t[0].parent)
                for t in sorted_data_tuples
            ]
        else:
            # In region-filtered modes, you might want to assign new folder names
            # (e.g. "graph_dir_0", "graph_dir_1", ...) so that each region is handled separately.
            generated_graphs = [
                (PokecGraph.graph_from_edgelist(str(t[0]), str(t[1])), resources_dir / f"graph_dir_{i}")
                for i, t in enumerate(sorted_data_tuples)
            ]

        for i, (graph_obj, _) in enumerate(generated_graphs):
            logger.info(f"Graph {i}: nodes={len(graph_obj.graph)}, edges={len(graph_obj.graph.edges())}, attributes={len(graph_obj.attributes)}")

        # --- RUN THE SELECTED EXPERIMENTS ---
        # The controllers handle baseline, CFN, CrossWalk, etc.
        if exp_run_mode in ["baseline", "both"]:
            ExperimentControllers.run_baseline_experiments(
                config, generated_graphs, resources_dir, result_dir, log_dir
            )
        # CFN is relevant only for crosswalk
        if exp_run_mode in ["crosswalk", "both"]:
            ExperimentControllers.run_cfn_proxy(
                config, generated_graphs, resources_dir, result_dir, log_dir
            )
            ExperimentControllers.run_crosswalk_experiments(
                config, generated_graphs, resources_dir, result_dir, log_dir
            )

        logger.info(f"Finished experiment mode: {exp_mode}")
    logger.info("All experiments completed.")

if __name__ == "__main__":
    main()