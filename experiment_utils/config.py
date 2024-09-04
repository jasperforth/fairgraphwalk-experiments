
from pathlib import Path
from typing import Final

# TODO refactor config into a Dataclass

# Experiment Date
PROJECT_NAME = "pokec_test_HPCGU"
# PROJECT_NAME = "test_edgecount2"

# Directories
RAW_DIR = "/scratch/pyllm/forth/DATA/raw/_pokec_raw_snap"
DATA_DIR = Path(f"/scratch/pyllm/forth/DATA/{PROJECT_NAME}")

FORMATTED_DATA_DIR = DATA_DIR / "formatted"

# File paths
RAW_ATTRIBUTES_FILEPATH = f"{RAW_DIR}/soc-pokec-profiles.txt"
EDGELIST_FILEPATH = f"{RAW_DIR}/soc-pokec-relationships.txt"

# Attributes and categories
FILTER_CATEGORY = "region"
COLUMN_NAMES = ["user_id", "public", "gender", "region", "AGE", "body"]
ATTRIBUTES = ["region", "AGE"]
SENS_ATTR_NAME = "label_AGE"
OTHER_ATTR_NAME = "label_region"

# Age split points with intervals (15,18], (18, 21], (21, 100]
# 16-18, 19-21, 22-99
SPLITPOINTS_BINS = [15, 18, 21, 100]

# Parallelism and randomness
WORKERS = 8
SEED = 42

# Cross-validation
N_SPLITS = 25

# Node2Vec parameters
P_VALUES = [1]
Q_VALUES = [1]
WALK_LENGTHS = [80]
DIM: Final[int] = 128

# CrossWalk parameters
PREWALK_LENGTH = 6
ALPHAS = [0.2, 0.5, 0.8]
EXPONENTS = [2, 8]

# QOL
INCLUDE_EDGECOUNTS = False # default