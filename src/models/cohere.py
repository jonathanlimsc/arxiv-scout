import os
import cohere
import numpy as np

from typing import List

COHERE_API_KEY = os.environ.get('COHERE_API_KEY', None)

class CohereModel():
    def __init__(self):
        self.client = self.init_client()
        self.cached_embeddings = None

    def init_client(self) -> cohere.Client:
        cohere_api_key = os.environ.get('COHERE_API_KEY', None)
        if cohere_api_key is None:
            raise Exception("Unable to initialise CohereModel as COHERE_API_KEY environment variable is not available")
        cohere_client = cohere.Client(cohere_api_key)
        return cohere_client

    def get_embeddings(self, texts: List[str], from_cache=False) -> np.ndarray:
        """
        Get document embeddings given the list of texts

        Args:
            texts: List of document strings
            from_cache: If True, return cached copy of embeddings. Otherwise, generate new embeddings from model and cache them.
        """
        if from_cache and self.cached_embeddings is not None:
            # Retrieve text embedding for query only. Document embeddings are from cache
            query_embedding = self.client.embed(texts=[texts[0]],
                                                model='small',
                                                truncate='LEFT').embeddings
            # Get document embeddings from cache
            doc_embeddings = self.cached_embeddings
            res_embeddings = np.concatenate([np.array(query_embedding), doc_embeddings], axis=0)
        else:
            # Retrieve embeddings from model for all the texts
            res_embeddings = self.client.embed(texts=texts,
                                                model='small',
                                                truncate='LEFT').embeddings
            res_embeddings = np.array(res_embeddings)

            # Refresh cache, excluding the query embedding at index 0
            self.cached_embeddings = res_embeddings[1:]
        
        return res_embeddings