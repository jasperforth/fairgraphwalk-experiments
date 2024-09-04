import sys
from pathlib import Path

# Add the parent directory of the current file to the system path for module imports.
file=Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from experiment_utils.config import DATA_DIR
from experiment_utils.logging_utils import setup_main_logging
from experiments._EXAMPLE_distinct import run_experiment_pokec_EXAMPLE_distinct
from experiments._EXAMPLE_semi import run_experiment_pokec_EXAMPLE_semi
from experiments._EXAMPLE_mixed import run_experiment_pokec_EXAMPLE_mixed

log_dir = DATA_DIR 
logger = setup_main_logging(log_dir)

def main(task_id):
    logger.info("Starting experiments")
    
    if task_id == 0:
        run_experiment_pokec_EXAMPLE_distinct()
    elif task_id == 1:
        run_experiment_pokec_EXAMPLE_semi()
    elif task_id == 2:
        run_experiment_pokec_EXAMPLE_mixed()
    else:
        logger.error("Invalid task ID")

if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)