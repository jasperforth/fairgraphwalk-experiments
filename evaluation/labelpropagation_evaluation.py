from pathlib import Path
import os
import logging
import pandas as pd
import gzip

from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Set the number of CPUs to run on and environment variables for parallel processing, to be set before importing numpy
# https://rcpedia.stanford.edu/topicGuides/parallelProcessingPython.html
# Note: parallelization is managed on experiment_run level, to avoid subparallelization set ncore to 1!
ncore = "1"
os.environ["OMP_NUM_THREADS"] = ncore
os.environ["OPENBLAS_NUM_THREADS"] = ncore
os.environ["MKL_NUM_THREADS"] = ncore
os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
os.environ["NUMEXPR_NUM_THREADS"] = ncore

import numpy as np

from .evaluation_strat import EvaluationStrategy
from data_utils.graph.graph import Graph
from experiment_utils.logging_utils import setup_worker_logging
from experiment_utils.config import DATA_DIR

# Configure logging
log_dir = DATA_DIR
setup_worker_logging("eval_label_propagation", log_dir)
logger = logging.getLogger(__name__)

# Ignore the RuntimeWarning: invalid value encountered in divide  probabilities /= normalizer
np.seterr(invalid='ignore')

class LabelPropagationEvaluation(EvaluationStrategy):
    """
    Label propagation evaluation strategy for graph embeddings.
    """
    def __init__(self, result_dir: Path, params_signature: str, 
                 sensitive_attribute_name: str, other_attribute_name: str, 
                 graph_name:str,  train_size: float = 0.5):
        """
        Initialize the LabelPropagationEvaluation.
        
        Args:
            result_dir (Path): Directory to save the results.
            params_signature (str): Signature of the parameter set.
            sensitive_attribute_name (str): Name of the sensitive attribute.
            other_attribute_name (str): Name of the other attribute.
            graph_name (str): Name of the graph.
            train_size (float, optional): Proportion of the dataset to include in the train split. Defaults to 0.5.
        """
        super().__init__(result_dir, params_signature, sensitive_attribute_name, 
                         other_attribute_name, train_size)
        self.graph_name = graph_name

    # TODO check whats up with evaluate and re-evaluate
    def evaluate(self, graph: Graph, embedding_filepath: Path):
        """
        Evaluate the embeddings using label propagation.
        
        Args:
            graph (Graph): Input graph.
            embedding_filepath (Path): Path to the embedding file.
        """
        logger.info(f"Evaluating embeddings for {self.graph_name} with parameters {self.params_signature}")

        df_attributes = graph.attributes
        d_emb, dim =self.read_embeddings(embedding_filepath)
        df_labels = self.read_labels(df_attributes, d_emb, self.sensitive_attribute_name, self.other_attribute_name)

        logger.info(f"Computing label propagation for {self.graph_name} {self.params_signature}")
        self.label_propagation_clf(df_labels, d_emb, dim, 
                                    self.result_dir,
                                    self.params_signature,
                                    self.sensitive_attribute_name, 
                                    self.other_attribute_name,
                                    self.train_size)
        
    def re_evaluate(self, df_attributes: pd.DataFrame, embedding_filepath: Path, n_splits: int = 25):
        """
        Re-evaluate the embeddings using label propagation.
        
        Args:
            df_attributes (pd.DataFrame): DataFrame containing node attributes.
            embedding_filepath (Path): Path to the embedding file.
            n_splits (int, optional): Number of splits for cross-validation. Defaults to 25.
        """
        logger.info(f"Re-evaluating embeddings for {self.graph_name} with parameters {self.params_signature}")

        d_emb, dim =self.read_embeddings(embedding_filepath)
        df_labels = self.read_labels(df_attributes, d_emb, self.sensitive_attribute_name, self.other_attribute_name)

        logger.info(f"Computing label propagation for {self.graph_name} {self.params_signature}")
        self.label_propagation_clf(df_labels, d_emb, dim, 
                                    self.result_dir,
                                    self.params_signature,
                                    self.sensitive_attribute_name, 
                                    self.other_attribute_name,
                                    self.train_size,
                                    n_splits)


    @staticmethod
    def label_propagation_clf(df_labels: pd.DataFrame, d_emb: dict, dim: int, result_dir: Path, params_signature: str,
                                sensitive_attribute_name: str, other_attribute_name: str, train_size: float, n_splits: int = 25) -> pd.DataFrame:
        """
        Perform label propagation classification.
        
        Args:
            df_labels (pd.DataFrame): DataFrame containing labels.
            d_emb (dict): Dictionary containing embeddings.
            dim (int): Dimension of the embeddings.
            result_dir (Path): Directory to save the results.
            params_signature (str): Signature of the parameter set.
            sensitive_attribute_name (str): Name of the sensitive attribute.
            other_attribute_name (str): Name of the other attribute.
            train_size (float): Proportion of the dataset to include in the train split.
            n_splits (int, optional): Number of splits for cross-validation. Defaults to 25.
        
        Returns:
            pd.DataFrame: DataFrame with the results.
        """
        assert len(df_labels.index) == len(d_emb)

        n = len(d_emb)
        X = np.zeros([n, dim])
        other_label = np.zeros([n])
        sens_label = np.zeros([n])
        for i, row in df_labels.iterrows():
            id = row['user_id']
            X[i, :] = d_emb[id]
            other_label[i] = row[other_attribute_name]
            sens_label[i] = row[sensitive_attribute_name]

        #The default kernel ('rbf') is often causing no convergence on our dataset. 
        # To improve the results,  use the 'knn' kernel option which has shown better performance.
        lp = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=500, n_jobs=1, tol=1e-3)
        # CrossValidation with stratified sampling for n_splits (default 25) runs.
        shs_split = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=42)

        with logging_redirect_tqdm():
            for label, y in enumerate([other_label, sens_label]):
                for run, (train_idx, test_idx) in tqdm(enumerate(shs_split.split(X, y))):
                    X_train = X  # the whole embedding
                    X_test = X[test_idx]
                    # mark all train node attributes as -1
                    y_train = np.full(X_train.shape[0], -1)
                    y_train[train_idx] = y[train_idx]
                    
                    model = lp.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
        
                    run_result_dir = result_dir / f"run_{run}" 
                    experiment_run_result_dir = run_result_dir / f"experiment_{params_signature}"
                    experiment_run_result_dir.mkdir(parents=True, exist_ok=True)

                    if label == 0:
                        df_labels_y = df_labels.copy()
                        for index, id in enumerate(test_idx):
                            assert id == test_idx[index], f"ID mismatch: id={id}, test_idx[index]={test_idx[index]}"  # Check if the IDs match
                            df_labels_y.loc[id, f'pred_{other_attribute_name}'] = y_pred[index]
                        df_labels_y.drop(columns=[sensitive_attribute_name], inplace=True)
                        df_labels_y.dropna(inplace=True)
                        df_labels_y.to_csv(experiment_run_result_dir / f"confusion_y.csv")
                    elif label == 1:
                        df_labels_z = df_labels.copy()
                        for index, id in enumerate(test_idx):
                            assert id == test_idx[index], f"ID mismatch: id={id}, test_idx[index]={test_idx[index]}"  # Check if the IDs match
                            df_labels_z.loc[id, f'pred_{sensitive_attribute_name}'] = y_pred[index]
                        df_labels_z.drop(columns=[other_attribute_name], inplace=True)
                        df_labels_z.dropna(inplace=True)
                        df_labels_z.to_csv(experiment_run_result_dir / f"confusion_z.csv")


    @staticmethod
    def read_embeddings(embedding_path: Path or str):
        """
        Read embeddings from a file.
        
        Args:
            embedding_path (Path): Path to the embedding file.
        
        Returns:
            tuple: Dictionary of embeddings and dimension of the embeddings.
        """
        d_emb = dict()
        # Check if the file is a .gz file
        if type(embedding_path) != str and embedding_path.suffix == '.gz':
            with gzip.open(embedding_path, 'rt') as emb:
                for i_l, line in enumerate(emb):
                    s = line.split()
                    if i_l == 0:
                        dim = int(s[1])
                        continue
                    d_emb[int(s[0])] = [float(x) for x in s[1:]]
        else:
            raise NotImplementedError(f"Embedding file {embedding_path} not supported")

        return d_emb, dim


    @staticmethod
    def read_labels(df_att: pd.DataFrame, d_emb: dict, sensitive_attribute_name: str, other_attribute_name: str) -> pd.DataFrame:
        """
        Read labels and match them with embeddings.
        
        Args:
            df_att (pd.DataFrame): DataFrame containing attributes.
            d_emb (dict): Dictionary containing embeddings.
            sensitive_attribute_name (str): Name of the sensitive attribute.
            other_attribute_name (str): Name of the other attribute.
        
        Returns:
            pd.DataFrame: DataFrame containing user_id, other_attribute_name, and sensitive_attribute_name.
        """
        label_data = []
        for index, row in df_att.iterrows():
            user_id = row['user_id']
            if user_id in d_emb:
                label_data.append([user_id,row[other_attribute_name],row[sensitive_attribute_name]])
        return pd.DataFrame(label_data,columns=['user_id', other_attribute_name, sensitive_attribute_name]).reset_index(drop=True)


