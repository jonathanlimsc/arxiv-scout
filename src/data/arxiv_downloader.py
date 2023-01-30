import requests
import pandas as pd
import xmltodict

from datetime import datetime, timedelta

from ..constants import ARXIV_CATEGORIES

ARXIV_QUERY_STR = "+OR+".join(ARXIV_CATEGORIES)

class ArxivDownloader():
    def __init__(self, download_refresh_interval_days: int = 1):
        self.download_refresh_interval_days = download_refresh_interval_days
        self.last_download_time = None
        self.latest_results = None

    def retrieve_arxiv_articles_df(self) -> pd.DataFrame:
        # Max results in one go is 1000
        url = f"http://export.arxiv.org/api/query?search_query={ARXIV_QUERY_STR}&start=0&max_results=1000"
        
        curr_time = datetime.now()
        
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
        
            # Update cache
            self.latest_results = articles_df
            self.last_download_time = curr_time

        # Retrieve from cache
        else:
            articles_df = self.latest_results

        return articles_df