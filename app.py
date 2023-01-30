import os
import dotenv

from flask import Flask, request, jsonify
from pathlib import Path

from src.data.arxiv_downloader import ArxivDownloader
from src.models.cohere import CohereModel
from src.utils import (
    compute_top_k,
)
from flask_config import (
    TOP_K,
    RECORD_KEYS,
)

# Load environment variables if in dev env
if os.environ.get('ENVIRONMENT') == 'dev':
    PROJ_DIR = Path.cwd()
    DOTENV_PATH = PROJ_DIR / '.env'
    dotenv.load_dotenv(DOTENV_PATH)

app = Flask(__name__)

ARXIV_DOWNLOADER = None
MODEL = None

@app.before_first_request
def startup():
    global ARXIV_DOWNLOADER
    global MODEL
    ARXIV_DOWNLOADER = ArxivDownloader(download_refresh_interval_days=1)
    ARXIV_DOWNLOADER.retrieve_arxiv_articles_df()
    MODEL = CohereModel()

@app.route('/health', methods=['GET'])
def health():
    return {'status': 'OK'}, 200

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data.get('query', None)
    if query is None:
        return jsonify({'data': []})

    # Retrieve article data
    articles_df = ARXIV_DOWNLOADER.retrieve_arxiv_articles_df()

    # Generate embeddings from model
    res_embeddings = MODEL.get_embeddings(texts=[query]+list(articles_df['combined_text']))
    query_embedding = res_embeddings[0]
    article_embeddings = res_embeddings[1:]

    # Find top k similar articles to query
    top_k_indices = compute_top_k(query_embedding, article_embeddings, k=TOP_K)
    records = articles_df.iloc[top_k_indices][RECORD_KEYS].to_dict(orient='records')

    return jsonify({'data': records})