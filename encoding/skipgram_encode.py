import os
import shutil
import gzip
import logging

from pathlib import Path
from typing import List
from gensim.models import Word2Vec
import pkg_resources

from .encoding_strat import EncodingStrategy

# Configure logging
logger = logging.getLogger(__name__)

# TODO cite Word2Vec and n2v
# TODO eventually limit ram in Word2Vec params
class SkipGramEncoder(EncodingStrategy):
    """
    Skip-gram encoder using gensim's Word2Vec for generating node embeddings.
    """
    def __init__(self, params_signature: str, experiment_graph_dir: Path, embedding_dimension: int = 128, **embedding_params):
        """
        Initialize the SkipGramEncoder.
        
        Args:
            params_signature (str): Signature of the parameter set.
            experiment_graph_dir (Path): Directory for the experiment graph.
            embedding_dimension (int, optional): Dimension of the embedding. Defaults to 128.
            **embedding_params: Additional parameters for the Word2Vec model.
        """
        super().__init__(params_signature, experiment_graph_dir, embedding_dimension, embedding_params)
    # 
    def fit(self, walks: List[str]) -> Word2Vec:
        """
        Train the Word2Vec model using the provided random walks to create the embeddings using gensim's Word2Vec based on n2v
        
        Args:
            walks (List[List[str]]): List of random walks.
        
        Returns:
            Word2Vec: Trained Word2Vec model.
        """
        logger.info('Starting skip-gram model training...')

        skip_gram_params = self.embedding_params
        gensim_version = pkg_resources.get_distribution("gensim").version
        size_key = 'vector_size' if gensim_version >= '4.0.0' else 'size'

        if size_key not in skip_gram_params:
            skip_gram_params[size_key] = self.embedding_dimension
        if 'sg' not in skip_gram_params:
            skip_gram_params['sg'] = 1

        
        logger.info('Skip-gram model training complete.')

        # wokrs=1 to avoid subparallelization
        return Word2Vec(walks, workers=1, **skip_gram_params)

    def embedding_to_file(self, model: Word2Vec, embedding_dir: Path) -> Path:
        """
        Save the trained embeddings to a file and compress it.
        
        Args:
            model (Word2Vec): Trained Word2Vec model.
            embedding_dir (Path): Directory to save the embeddings.
        
        Returns:
            Path: Path to the compressed embedding file.
        """
        embedding_filepath = (embedding_dir / f"{self.params_signature}.emb").absolute().as_posix()

        logger.info(f'Saving embeddings to {embedding_filepath}')
        model.wv.save_word2vec_format(embedding_filepath)

        compressed_filepath = embedding_filepath + '.gz'

        with open(embedding_filepath, 'rb') as file_in:
            with gzip.open(compressed_filepath, 'wb') as file_out:
                shutil.copyfileobj(file_in, file_out)
                        
        # Remove the original (uncompressed) embedding file.
        os.remove(embedding_filepath)
        logger.info(f'Embeddings saved and compressed to {compressed_filepath}')
        
        return Path(embedding_filepath + '.gz')
 

    