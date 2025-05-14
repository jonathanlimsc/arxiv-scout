import os
import cohere
import numpy as np

from typing import List

COHERE_API_KEY = os.environ.get('COHERE_API_KEY', None)
MODEL_NAME = os.environ.get('MODEL_NAME', 'embed-english-light-v3.0')
MODEL_BATCH_SIZE = os.environ.get('MODEL_BATCH_SIZE', 96)

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
                                                model=MODEL_NAME,
                                                input_type='search_query',
                                                truncate='END').embeddings[0]
            # Get document embeddings from cache
            doc_embeddings = self.cached_embeddings
            res_embeddings = [query_embedding]
            res_embeddings.extend(doc_embeddings)
            res_embeddings = np.array(res_embeddings)
        else:
            # Retrieve embeddings from model for all the texts
            res_embeddings = []
            query_text = texts[0]
            doc_texts = texts[1:]
            query_embedding = self.client.embed(texts=[query_text],
                                                model=MODEL_NAME,
                                                input_type='search_query',
                                                truncate='END').embeddings[0]
            res_embeddings.append(query_embedding)

            for i in range(0, len(doc_texts), MODEL_BATCH_SIZE):
                batch_texts = texts[i:i + MODEL_BATCH_SIZE]
                # Generate embeddings for the current batch
                res_embeddings_batch = self.client.embed(texts=batch_texts,
                                                        model=MODEL_NAME,
                                                        input_type='search_document',
                                                        truncate='END').embeddings
                res_embeddings.extend(res_embeddings_batch)


            res_embeddings = np.array(res_embeddings)

            # Refresh cache, excluding the query embedding at index 0
            self.cached_embeddings = res_embeddings[1:]
        
        return res_embeddings