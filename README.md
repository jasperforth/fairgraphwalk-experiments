# Fair Node Sampling Project

**Notice:** Due to unforeseen circumstances, the final upload of the codebase is slightly delayed. We are currently integrating the latest plotting methods. Soon, you will find the complete code for the experiments, including smaller computable examples, available here.

## Overview

This project is based on the paper **"Fairness Through Controlled (Un)Awareness in Node Embeddings"** by Dennis Vetter, Jasper Forth, Gemma Roig, and Holger Dell. The research demonstrates how the parametrization of the CrossWalk algorithm can influence the inference of sensitive attributes from node embeddings, thus offering a tool to improve the fairness of machine learning systems using graph embeddings.

## Requirements

- **Python Version:** 3.10.0
- **Dependencies:** Listed in `requirements.txt`. We recommend using a virtual environment to install the necessary packages.

Load the raw data from [Pokec Social Network](https://snap.stanford.edu/data/soc-Pokec.html).

## Configuration

The configuration file is located in the `experiment_utils` folder. Set the parameters, directories, and raw file paths in the config file.

## Examples

To try the code, use the `example_run_experiments.py` script:

```bash
python example_run_experiments.py
```
In the experiments folder, you can find the experiments. By default, two example experiments are chosen to run in acceptable time (approximately 3 hours and 30 minutes with 4 workers). These experiments include:

pokec_distinct with graph 0
pokec_distinct with graph 1
pokec_semi with graph 0
For running the full Pokec experiments, adjust the run_experiments.py and config files, and ensure you have adequate computational resources.

Note: For a shorter example, run only distinct with graph 0 (should complete in approximately 34 minutes with 4 workers).

## Logging

Logs are saved in the ./data/logfiles folder.
One file for the main and one file for all paralell threads together.

## Reporting

### Process Results
The results of the evaluation can be processed in the reporting section. 
Adjust the report/config file to the project and run the report/process_results notebook. 
This generates averaged confusion reports and a combined CSV for further analysis and plotting.

### Plotting 
Note: The reporting section will be updated soon as we finalize the structure and integrate the latest plotting methods.

## About the Paper

### Abstract
Graph representation learning is central for applying machine learning (ML) models to complex graphs, such as social networks. Ensuring ‘fair’ representations is essential due to societal implications and the use of sensitive personal data. This paper demonstrates how the parametrization of the CrossWalk algorithm influences the ability to infer sensitive attributes from node embeddings. By fine-tuning hyperparameters, it is possible to either significantly enhance or obscure the detectability of these attributes. This functionality offers a valuable tool for improving the fairness of ML systems utilizing graph embeddings, making them adaptable to different fairness paradigms.

### Key Contributions
- Parametrization of CrossWalk: Demonstrates how to adjust the detectability of sensitive attributes from node embeddings.
- Fairness Analysis: Evaluates the impact of fairness improvements on embedding quality using a non-sensitive control attribute.
- Integrated Implementation: Provides tools to apply and evaluate both CrossWalk and node2vec, enabling easy comparison of their embeddings.
Thank you for your patience and understanding.

### Results from the Paper
Note: The processed data for the results from the paper will be provided as .csv file.

