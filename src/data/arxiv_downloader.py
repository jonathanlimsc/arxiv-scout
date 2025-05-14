import os
import requests
import pandas as pd
import xmltodict

from datetime import datetime, timedelta

from ..constants import ARXIV_CATEGORIES

ARXIV_QUERY_STR = "+OR+".join(ARXIV_CATEGORIES)
MAX_CHARS = 750
MAX_NUM_PAPERS_TO_RETRIEVE = os.environ.get('MAX_NUM_PAPERS_TO_RETRIEVE', 50)

class ArxivDownloader():
    def __init__(self, download_refresh_interval_days: int = 1):
        self.download_refresh_interval_days = download_refresh_interval_days
        self.last_download_time = None
        self.latest_results = None

    def retrieve_arxiv_articles_df(self) -> pd.DataFrame:
        """
        Retrieves Arxiv article dataframe

        Returns
            articles_df: Arxiv article dataframe
            is_from_cache: Whether the documents were a cached copy. This can usefully be passed to downstream models to return a cached version of document embeddings for fast response times
        """
        # Max results in one go is 1000
        url = f"http://export.arxiv.org/api/query?search_query={ARXIV_QUERY_STR}&sortBy=lastUpdatedDate&start=0&max_results={MAX_NUM_PAPERS_TO_RETRIEVE}"
        
        curr_time = datetime.now()
        is_from_cache = None

        if self.last_download_time is None or (curr_time - self.last_download_time) > timedelta(days=self.download_refresh_interval_days):
            payload={}
            headers = {}

            response = requests.request("GET", url, headers=headers, data=payload)
            parsed_xml = xmltodict.parse(response.text)
            
            articles_dict = {'link': [],
                            'updated_ts': [],
                            'published_ts': [],
                            'title': [],
                            'summary': [],
                            'author': [],
                            'category': [],
                            }
            
            for article in parsed_xml['feed']['entry']:
                try:
                    articles_dict['link'].append(article['id'])
                    articles_dict['updated_ts'].append(article['updated'])
                    articles_dict['published_ts'].append(article['published'])
                    articles_dict['title'].append(article['title'])
                    articles_dict['summary'].append(article['summary'])
                    articles_dict['author'].append(", ".join([author['name'] for author in article['author']]) if isinstance(article['author'], list) else article['author']['name'])
                    articles_dict['category'].append(article['arxiv:primary_category']['@term'])
                except Exception as e:
                    print(f"Exception: {e}")
                    continue

            articles_df = pd.DataFrame.from_dict(articles_dict)

            # Generate combined text from title and summary
            articles_df['summary'] = articles_df['summary'].str.replace('\n', ' ')
            articles_df['combined_text'] = (articles_df['title'] + '. ' + articles_df['summary']).str.lower()
            articles_df['combined_text'] = articles_df['combined_text'].str[:MAX_CHARS]
            
            # Update cache
            self.latest_results = articles_df
            self.last_download_time = curr_time
            is_from_cache = False

        # Retrieve from cache
        else:
            articles_df = self.latest_results
            is_from_cache = True

        return articles_df, is_from_cache