import logging
import pandas as pd
from tqdm import tqdm
from .base import Preprocessor
from bs4 import BeautifulSoup
import urllib.request
import time

class ICD9DataPreprocessor(Preprocessor):
    icd9data_base_url = 'http://www.icd9data.com'

    def load_data(self) -> pd.DataFrame:
        logging.info('Starting to query ICD9 data')
        return self._query_hierarchy_from('http://www.icd9data.com/2015/Volume1/default.htm', 'root', '-1')

    def _open_url(self, url):
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(request)
        return BeautifulSoup(response, 'html.parser', from_encoding=response.info().get_param('charset'))

    def _open_url_gentle(self, url, max_retries=10, timeout_s=3, error_timeout_s=60):
        try:
            time.sleep(timeout_s)
            return self._open_url(url)
        except urllib.error.HTTPError as error:
            logging.error('Error trying to query URL %s: %s', url, error)
            if max_retries < 0:
                raise error
            else:
                time.sleep(error_timeout_s)
                return self._open_url_gentle(url, max_retries-1, timeout_s, error_timeout_s)

    def _query_leaf_hierarchy_from(self, parent_url, parent_name, parent_code):
        logging.debug('Querying ICD9 data from %s', parent_url)
        soup = self._open_url_gentle(parent_url)

        hierarchy_df = pd.DataFrame(columns=['parent_url', 'parent_name', 'parent_code', 'child_url', 'child_name', 'child_code'])
        definition_list = soup.find_all(class_='codeHierarchyUL')[0]
        for list_item in definition_list.find_all('li'):
            child_url = self.icd9data_base_url + list_item.a['href']
            child_name = list_item.find_all(class_='threeDigitCodeListDescription')[0].get_text()
            child_code = list_item.a.get_text()
            hierarchy_df = hierarchy_df.append({
                'parent_url': parent_url, 
                'parent_name': parent_name, 
                'parent_code': parent_code, 
                'child_url': child_url, 
                'child_name': child_name,
                'child_code': child_code,
            }, ignore_index=True)
        return hierarchy_df

    def _query_hierarchy_from(self, parent_url, parent_name, parent_code) -> pd.DataFrame:
        logging.debug('Querying ICD9 data from %s', parent_url)
        soup = self._open_url_gentle(parent_url)

        hierarchy_df = pd.DataFrame(columns=['parent_url', 'parent_name', 'parent_code', 'child_url', 'child_name', 'child_code'])
        definition_list = soup.find_all(class_='definitionList')[0]
        for list_item in tqdm(definition_list.find_all('li'), desc='Parsing child codes from code ' + str(parent_code)):
            child_url = self.icd9data_base_url + list_item.a['href']
            child_text = list_item.get_text()
            child_code = child_text.split(' ')[0]
            child_name = ' '.join(child_text.split(' ')[1:])
            hierarchy_df = hierarchy_df.append({
                'parent_url': parent_url, 
                'parent_name': parent_name, 
                'parent_code': parent_code, 
                'child_url': child_url, 
                'child_name': child_name,
                'child_code': child_code,
            }, ignore_index=True)
            if '-' in child_code:
                hierarchy_df = hierarchy_df.append(self._query_hierarchy_from(child_url, child_name, child_code), ignore_index=True)
            else:
                hierarchy_df = hierarchy_df.append(self._query_leaf_hierarchy_from(child_url, child_name, child_code), ignore_index=True)

        return hierarchy_df