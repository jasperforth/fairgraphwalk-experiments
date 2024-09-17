import sys
from pathlib import Path

# Add the parent directory of the current file to the system path for module imports.
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from experiment_utils.config import DATA_DIR
from experiment_utils.logging_utils import setup_main_logging
from experiments.FMMC_pokec_distinct_sequ import new_run_experiment_pokec_distinct
from experiments.FMMC_pokec_semi_sequ import new_run_experiment_pokec_semi
from experiments.FMMC_pokec_mixed_sequ_2 import new_run_experiment_pokec_mixed

log_dir = DATA_DIR
logger = setup_main_logging(log_dir)

def main(task_id):
    logger.info(f"Starting experiment with task_id: {task_id}")
    
    if task_id == 0:
        logger.info("Running experiment: new_run_experiment_pokec_distinct('small')")
        new_run_experiment_pokec_distinct("small")
    elif task_id == 1:
        logger.info("Running experiment: new_run_experiment_pokec_semi('small')")
        new_run_experiment_pokec_semi("small")
    elif task_id == 2:
        logger.info("Running experiment: new_run_experiment_pokec_mixed('vsmall1')")
        new_run_experiment_pokec_mixed("vsmall1")
    elif task_id == 3:
        logger.info("Running experiment: new_run_experiment_pokec_mixed('vsmall2')")
        new_run_experiment_pokec_mixed("vsmall2")
    elif task_id == 4:
        logger.info("Running experiment: new_run_experiment_pokec_mixed('small1')")
        new_run_experiment_pokec_mixed("small1")
    elif task_id == 5:
        logger.info("Running experiment: new_run_experiment_pokec_mixed('small2')")
        new_run_experiment_pokec_mixed("small2")
    elif task_id == 6:
        logger.info("Running experiment: new_run_experiment_pokec_mixed('large1')")
        new_run_experiment_pokec_mixed("large1")
    elif task_id == 7:
        logger.info("Running experiment: new_run_experiment_pokec_mixed('large2')")
        new_run_experiment_pokec_mixed("large2")
    elif task_id == 8:
        logger.info("Running experiment: new_run_experiment_pokec_mixed('large3')")
        new_run_experiment_pokec_mixed("large2")

    else:
        logger.error(f"Invalid task ID: {task_id}")

if __name__ == "__main__":
    task_id = int(sys.argv[1])
    # task_id = 1
    main(task_id)

