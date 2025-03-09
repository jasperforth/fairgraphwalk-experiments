#!/usr/bin/env bash
# File: experiments/demo_run.sh
# Description: Simple script to run a quick demo with the unified pipeline.
# Usage:
#   bash experiments/demo_run.sh

# We call experiment_main.py with the --demo_subset flag, which tells it to run a smaller set of experiments.
python experiments/experiments_main.py \
  --config experiments/config.yml \
  --experiment_to_run both \
  --demo_subset
