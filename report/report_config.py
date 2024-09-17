import sys
from pathlib import Path

file=Path(__file__).resolve()
prent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# from experiment_utils.config import ATTRIBUTES, DATA_DIR, PROJECT_NAME


# Set variablesa or use data from experiment_ustils.config
# PROJECT_NAME = PROJECT_NAME
PROJECT_NAME = "pokec_FMMC_the_final_experiment_002"
# PROJECT_NAME = "testing_repo"

DATA_DIR = Path(f"/mnt/c/Users/Erik/Desktop/BA_exp/{PROJECT_NAME}/data")
REPORT_DIR = DATA_DIR / "reports"

# Set the partial experiment name depending of the subset you want to analyze
SUB_EXPERIMENTS_NAME_START = "new_pokec_"
# Set attributes or use attributes from experiment_utils.config
# ATTRIBUTES = ATTRIBUTES
ATTRIBUTES = ["region", "AGE"]

