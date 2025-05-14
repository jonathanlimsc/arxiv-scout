arxiv-scout
==============================
[![DOI](https://zenodo.org/badge/595349248.svg)](https://zenodo.org/badge/latestdoi/595349248)

Semantic embedding-based search for the latest Arxiv papers 

[App Demo](https://jonathanlimsc.com/projects/arxiv-scout/app) | [Blogpost](https://jonathanlimsc.com/projects/arxiv-scout/)

![Arxiv Scout App](assets/arxiv-scout-screenshot.png)

## Installation

1. Create a conda environment and pip install dependencies.
```
conda create -n arxiv-scout python=3.8
conda activate arxiv-scout
pip install -r requirements.txt
```

2. Sign up for a [Cohere developer account](https://dashboard.cohere.ai/) and create a .env file at the project root directory. Initialize the environment variable `COHERE_API_KEY` with your Cohere API key. This is necessary for the dev environment. To run in production, set the environment variable `COHERE_API_KEY` to your Cohere API key via your production platform's secrets or environment variables management system.
```
# In .env file
COHERE_API_KEY = <YOUR_API_KEY_HERE>
```

3. Run notebooks in the `/notebooks` directory.
```
jupyter notebook
```

4. Run the Flask application to query
```
flask --app app:app run
```

### Example request
**POST /api/query**
```
import requests
import json

url = "http://127.0.0.1:5000/api/query"

payload = json.dumps({
  "query": "Attention transformers are all you need"
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)


```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── constants.py   <- Global constants that can be used by any module
    │   ├── utils.py       <- Utility functions
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── arxiv_downloader.py <- Contains ArxivDownloader class to handle communication to Arxiv API
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── cohere.py  <- CohereModel class to instantiate connections and make requests to Cohere API for embeddings
    │   │ 
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    ├── app.py             <- Flask microservice app that is deployed as the backend on Render
    ├── flask_config.py    <- Configurations for Flask app
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

## Citation

If used this work or you found this work useful, please use this BibTeX to cite this repository in your publications or works:
```
@software{jonathan_lim_siu_chi_2023_7595801,
  author       = {Jonathan Lim Siu Chi},
  title        = {jonathanlimsc/arxiv-scout: Arxiv Scout v1.0},
  month        = feb,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v1.0},
  doi          = {10.5281/zenodo.7595801},
  url          = {https://doi.org/10.5281/zenodo.7595801}
}
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
