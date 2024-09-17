import sys
from pathlib import Path

# Add the parent directory of the current file to the system path for module imports.
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from experiment_utils.config import DATA_DIR
from experiment_utils.logging_utils import setup_main_logging
from experiments.FMMC_pokec_distinct import new_run_experiment_pokec_distinct
from experiments.FMMC_pokec_semi import new_run_experiment_pokec_semi
from experiments.FMMC_pokec_mixed import new_run_experiment_pokec_mixed

log_dir = DATA_DIR
logger = setup_main_logging(log_dir)

def main(task_id):
    logger.info(f"Starting experiment with task_id: {task_id}")
    
    if task_id == 0:
        logger.info("Running experiment: new_run_experiment_pokec_distinct('small')")
        new_run_experiment_pokec_distinct("small")
    elif task_id == 1:
        logger.info("Running experiment: new_run_experiment_pokec_distinct('large')")
        new_run_experiment_pokec_distinct("large")
        # logger.info("the end")

    elif task_id == 2:
        logger.info("Running experiment: new_run_experiment_pokec_semi('small')")
        new_run_experiment_pokec_semi("small")
    elif task_id == 3:
        logger.info("Running experiment: new_run_experiment_pokec_semi('large')")
        new_run_experiment_pokec_semi("large")

    elif task_id == 4:
        logger.info("Running experiment: new_run_experiment_pokec_mixed('large')")
        new_run_experiment_pokec_mixed("large")

    else:
        logger.error(f"Invalid task ID: {task_id}")

if __name__ == "__main__":
    task_id = int(sys.argv[1])
    # task_id = 1
    main(task_id)

