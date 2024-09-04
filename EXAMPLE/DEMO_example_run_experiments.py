import sys
from pathlib import Path

# Add the parent directory of the current file to the system path for module imports.
file=Path(__file__).resolve()
prent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from experiments._EXAMPLE_distinct import run_experiment_pokec_EXAMPLE_distinct
from experiments._EXAMPLE_semi import run_experiment_pokec_EXAMPLE_semi
from experiments.pokec_distinct import run_experiment_pokec_distinct
from experiments.pokec_semi import run_experiment_pokec_semi
from experiments.pokec_mixed import run_experiment_pokec_mixed
from experiment_utils.config import DATA_DIR
from experiment_utils.logging_utils import setup_main_logging

log_dir = DATA_DIR 
logger = setup_main_logging(log_dir)

if __name__ == "__main__":
    logger.info("Starting experiments")
    run_experiment_pokec_EXAMPLE_distinct()
    run_experiment_pokec_EXAMPLE_semi()
    # run_experiment_pokec_distinct()
    # run_experiment_pokec_semi()
    # run_experiment_pokec_mixed()
    logger.info("Experiments completed")