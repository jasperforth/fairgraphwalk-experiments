import sys
from pathlib import Path

file=Path(__file__).resolve()
prent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from experiment_utils.config import ATTRIBUTES, DATA_DIR, PROJECT_NAME


# Set variablesa or use data from experiment_ustils.config
PROJECT_NAME = PROJECT_NAME
# PROJECT_NAME = "testing_repo"

# DATA_DIR = Path(f"/Volumes/tcs_jf_fair_node_sampling/{PROJECT_NAME}/data")
REPORT_DIR = DATA_DIR / "reports"

# Set the partial experiment name depending of the subset you want to analyze
SUB_EXPERIMENTS_NAME_START = "pokec_"
# Set attributes or use attributes from experiment_utils.config
ATTRIBUTES = ATTRIBUTES

