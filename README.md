# Fair Node Sampling Project

## Overview

This repository was created to run experiments that contributed to the paper [Fairness Through Controlled (Un)Awareness in Node Embeddings](https://arxiv.org/abs/2407.20024) by Dennis Vetter, Jasper Forth, Gemma Roig, and Holger Dell. The project demonstrates how the parametrization of the CrossWalk algorithm influences the inference of sensitive attributes from node embeddings, offering a way to improve fairness in machine learning systems using graph embeddings.

The modular architecture of this repository is open for further experimental setups, including new fairness biasing strategies (Work in Progress), the inclusion of link prediction (Work in Progress), and the relaxation of using two attributes (Age and Location) to allow multiple attributes, as well as the integration of more general subgraph selections and synthetic node label generation (Work in Progress).

## Deployment

The experiments were originally deployed on a shared cluster using 128 nodes in parallel and are now running on the Goethe-NHR Cluster. We recommend using **micromamba** for environment management on the cluster.

## Experimental Results

The experimental results are not stored in this repository but are available on request. 

For the results, please refer to the paper or contact:

forth [at] em.uni-frankfurt.de

## Requirements

- **Python Version:** 3.10.0
- **Dependencies:** Listed in both `requirements.txt` and the new `environment.yaml` file for easy environment setup.

We recommend using **micromamba** or **conda** to manage the environment, though `pip` is also supported. Instructions are provided below.

### Using Micromamba

1. Install **micromamba** and create a new environment:
   ```bash
   micromamba create --name fair_graph310 python=3.10
   micromamba activate fair_graph310
   ```
2. Install dependencies from the `environment.yaml` file:
   ```bash
   micromamba env update --file environment.yaml
   ```

### Using Conda

1. Create a new environment:
   ```bash
   conda env create --file environment.yaml
   conda activate fair_graph310
   ```

### Using Pip

1. Create a virtual environment:
   ```bash
   python3 -m venv fair_graph310
   source fair_graph310/bin/activate
   ```
2. Install dependencies from the `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The configuration file is located in the `experiment_utils` folder. Set the parameters, directories, and raw file paths in the config file.

## Examples

To try the code, use the `DEMO_example_run_experiments.py` script:

```bash
python DEMO_example_run_experiments.py
```

In the `experiments` folder, you'll find a variety of experimental setups. By default, two example experiments are chosen to run within an acceptable time frame (approximately 3 hours and 30 minutes with 4 workers). These experiments include:

- `pokec_distinct` with graph 0
- `pokec_distinct` with graph 1
- `pokec_semi` with graph 0

For running the full Pokec experiments, adjust the `run_experiments.py` and config files, and ensure you have sufficient computational resources.

*Note:* For a shorter example, run only the distinct experiment with graph 0 (should complete in approximately 34 minutes with 4 workers).

## Logging

Logs are saved in the `./data/logfiles` folder. One file is used for the main process and another for parallel threads.

## Reporting

### Process Results

The results of the evaluation can be processed in the reporting section. Adjust the `report/config` file to the project and run the `report/process_results` notebook. This generates averaged confusion reports and a combined CSV for further analysis and plotting.

### Plotting

Note: The `ICML_notebooks` section will be updated as soon as we finalize the structure and integrate the latest plotting methods.

## Planned Improvements

- **Use Dataclass for Configuration**: Replace the global config script with a `dataclass` for managing configuration directly within experiments.
- **Modularize Plotting**: Refactor the plotting utilities for better modularity and flexibility.
- **Relaxation of Attributes**: Extend the current model to allow multiple sensitive attributes instead of restricting to only Age and Location (region), along with generalized subgraph selections and the creation of synthetic node labels.

## About the Paper

You can find the .pdf of the paper [here](paper_fairness_through_controlled_un_awareness.pdf)

### Abstract

Graph representation learning is central for applying machine learning (ML) models to complex graphs, such as social networks. Ensuring ‘fair’ representations is essential due to societal implications and the use of sensitive personal data. This paper demonstrates how the parametrization of the CrossWalk algorithm influences the ability to infer sensitive attributes from node embeddings. By fine-tuning hyperparameters, it is possible to either significantly enhance or obscure the detectability of these attributes. This functionality offers a valuable tool for improving the fairness of ML systems utilizing graph embeddings, making them adaptable to different fairness paradigms.

### Key Contributions

- **Parametrization of CrossWalk**: Demonstrates how to adjust the detectability of sensitive attributes from node embeddings.
- **Fairness Analysis**: Evaluates the impact of fairness improvements on embedding quality using a non-sensitive control attribute.
- **Integrated Implementation**: Provides tools to apply and evaluate both CrossWalk and node2vec, enabling easy comparison of their embeddings.

### Results from the Paper

Note: The original processed averaged data for the results from the paper is provided as a .csv file in `ICML_notebooks/data`.


## Citation

If you use this repository or the results in your research, please consider citing the following paper:

```bibtex
@misc{vetter2024fairnesscontrolledunawarenessnode,
      title={Fairness Through Controlled (Un)Awareness in Node Embeddings}, 
      author={Dennis Vetter and Jasper Forth and Gemma Roig and Holger Dell},
      year={2024},
      eprint={2407.20024},
      archivePrefix={arXiv},
      primaryClass={cs.SI},
      url={https://arxiv.org/abs/2407.20024}, 
}
```
