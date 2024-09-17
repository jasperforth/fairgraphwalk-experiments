import sys
from pathlib import Path

# Add the parent directory of the current file to the system path for module imports.
file=Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from experiment_utils.config import DATA_DIR
from experiment_utils.logging_utils import setup_main_logging
from experiments.FMMC_pokec_mixed import new_run_experiment_pokec_mixed

log_dir = DATA_DIR 
logger = setup_main_logging(log_dir)

def main():
    logger.info("Starting experiments")
    # run_experiment_pokec_EXAMPLE_distinct()
    # run_experiment_pokec_EXAMPLE_semi()
    # run_experiment_pokec_distinct()
    # run_experiment_pokec_semi()
    # run_experiment_pokec_mixed()
    # run_experiment_corr_pokec_allexp_graphs0()

    # new_run_experiment_pokec_distinct()
    new_run_experiment_pokec_mixed("small")
    logger.info("Experiments completed")

if __name__ == "__main__":
    main()