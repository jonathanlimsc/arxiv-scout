import os
import cohere
import numpy as np

from typing import List

COHERE_API_KEY = os.environ.get('COHERE_API_KEY', None)

class CohereModel():
    def __init__(self):
        self.client = self.init_client()

    def init_client(self) -> cohere.Client:
        cohere_api_key = os.environ.get('COHERE_API_KEY', None)
        if cohere_api_key is None:
            raise Exception("Unable to initialise CohereModel as COHERE_API_KEY environment variable is not available")
        cohere_client = cohere.Client(cohere_api_key)
        return cohere_client

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        res_embeddings = self.client.embed(texts=texts,
                                            model='small',
                                            truncate='LEFT').embeddings

        return np.array(res_embeddings)