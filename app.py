import os
import dotenv

from flask import (
    Flask, 
    request, 
    jsonify,
    redirect,
    Response
)

from pathlib import Path

from src.data.arxiv_downloader import ArxivDownloader
from src.models.cohere import CohereModel
from src.utils import (
    compute_top_k,
)
from src.constants import DEFAULT_TOP_K
from flask_config import (
    RECORD_KEYS,
)

# Load environment variables from dotenv file if in dev env
if os.environ.get('ENVIRONMENT', 'dev') == 'dev':
    PROJ_DIR = Path.cwd()
    DOTENV_PATH = PROJ_DIR / '.env'
    dotenv.load_dotenv(DOTENV_PATH)

app = Flask(__name__)

ARXIV_DOWNLOADER = None
MODEL = None
TOP_K = int(os.environ.get('TOP_K', DEFAULT_TOP_K))

def _build_cors_preflight_response():
    res = Response()
    res.headers.add("Access-Control-Allow-Origin", "*")
    res.headers.add('Access-Control-Allow-Headers', "*")
    res.headers.add('Access-Control-Allow-Methods', "*")
    return res

@app.before_first_request
def startup():
    global ARXIV_DOWNLOADER
    global MODEL
    ARXIV_DOWNLOADER = ArxivDownloader(download_refresh_interval_days=1)
    MODEL = CohereModel()
    # Download latest Arxiv articles and generate embeddings
    articles_df, _ = ARXIV_DOWNLOADER.retrieve_arxiv_articles_df()
    MODEL.populate_cache_embeddings(texts=list(articles_df['combined_text']))

@app.route('/', methods=['GET'])
def index():
    return redirect('/health')

@app.route('/health', methods=['GET'])
def health():
    return {'status': 'OK'}, 200

@app.route('/api/ping', methods=['GET'])
def ping():
    return {'response': 'pong'}, 200

@app.route('/api/query', methods=['POST', 'OPTIONS'])
def query():
    # Handle CORS pre-flight request
    # Reference: https://dothanhlong.org/how-to-enable-cors-in-python-flask/
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    data = request.get_json()
    query = data.get('query', None)
    if query is None:
        return jsonify({'data': []})

    # Retrieve article data
    articles_df, is_from_cache = ARXIV_DOWNLOADER.retrieve_arxiv_articles_df()
    
    # Generate embeddings from model
    res_embeddings = MODEL.get_embeddings(texts=[query]+list(articles_df['combined_text']), from_cache=is_from_cache)
    query_embedding = res_embeddings[0]
    article_embeddings = res_embeddings[1:]

    # Find top k similar articles to query
    top_k_indices, similarity_scores = compute_top_k(query_embedding, article_embeddings, k=TOP_K)
    results_df = articles_df.iloc[top_k_indices].reset_index(drop=True).copy()
    results_df['similarity'] = similarity_scores
    records = results_df[RECORD_KEYS].to_dict(orient='records')
    res = jsonify({'data': records})
    # To allow CORS
    # References: https://dev.to/authress/i-got-a-cors-error-now-what-hpb
    # https://stackoverflow.com/questions/26980713/solve-cross-origin-resource-sharing-with-flask
    res.headers.add('Access-Control-Allow-Origin', '*')
    return res