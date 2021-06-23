import logging
import pandas as pd
from tqdm import tqdm
from .base import Preprocessor
from bs4 import BeautifulSoup
import urllib.request
import time
from typing import List, Dict, Any
from pathlib import Path


class ICD9DataPreprocessor(Preprocessor):
    icd9data_base_url = "http://www.icd9data.com"

    def __init__(
        self,
        icd9_file: Path,
        icd9_hierarchy_file: Path = Path("data/hierarchy_icd9.csv"),
    ):
        self.icd9_file = icd9_file
        self.icd9_hierarchy_file = icd9_hierarchy_file

    def load_data(self) -> pd.DataFrame:
        logging.info("Trying to read icd9_df from %s", self.icd9_file)
        if not self.icd9_file.is_file():
            icd9_df = self._query_data()
            icd9_df.to_csv(self.icd9_file, index=False)

        return pd.read_csv(self.icd9_file, dtype=str)

    def load_data_as_hierarchy(self) -> pd.DataFrame:
        logging.info(
            "Trying to read icd9_hierarchy_df from %s", self.icd9_hierarchy_file
        )
        if not self.icd9_hierarchy_file.is_file():
            icd9_hierarchy_df = self._generate_icd9_hierarchy()
            icd9_hierarchy_df.to_csv(self.icd9_hierarchy_file, index=False)

        icd9_hierarchy_df = pd.read_csv(self.icd9_hierarchy_file, dtype=str)
        icd9_hierarchy_df["level_all"] = icd9_hierarchy_df.apply(lambda x: list(x), axis=1)
        return icd9_hierarchy_df

    def _find_icd9_parents_for_child(
        self, icd9_df: pd.DataFrame, child_code: str
    ) -> List[str]:
        direct_parents = [
            x
            for x in set(icd9_df[icd9_df["child_code"] == child_code]["parent_code"])
            if not x == child_code
        ]
        if len(direct_parents) == 0:
            return []
        if len(direct_parents) > 1:
            logging.warn(
                "Found multiple icd9 parents for child %s: %s",
                child_code,
                ",".join(direct_parents),
            )

        parent_code = direct_parents[0]
        if parent_code == "-1":
            return []

        next_parents = self._find_icd9_parents_for_child(icd9_df, parent_code)
        return [parent_code] + next_parents

    def _generate_icd9_hierarchy(self) -> pd.DataFrame:
        icd9_df = self.load_data()

        logging.info("Converting icd9_df to hierarchy")
        child_codes = set(icd9_df["child_code"])
        children_to_parents = {}
        for child_code in tqdm(child_codes, "Converting icd9 data to hierarchy dict"):
            children_to_parents[child_code] = self._find_icd9_parents_for_child(
                icd9_df, child_code
            )

        max_parents = max([len(x) for x in children_to_parents.values()]) + 1
        child_hierarchy = pd.DataFrame(
            columns=["level_" + str(i) for i in range(max_parents)], dtype=str
        )
        for child_code, parents in tqdm(
            children_to_parents.items(),
            desc="Converting icd9 hierarchy dict to dataframe",
        ):
            parents = [str(x) for x in parents if len(str(x)) > 0]
            while len(parents) < max_parents:
                parents = [child_code] + parents

            child_to_parents: Dict[str, str] = {}
            for parent_idx in range(len(parents)):
                child_to_parents["level_" + str(parent_idx)] = parents[parent_idx]

            child_hierarchy = child_hierarchy.append(
                child_to_parents, ignore_index=True,
            )

        return child_hierarchy

    def _query_data(self) -> pd.DataFrame:
        logging.info("Starting to query ICD9 data")
        return self._query_hierarchy_from(
            "http://www.icd9data.com/2015/Volume1/default.htm", "root", "-1"
        )

    def _open_url(self, url):
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(request)
        return BeautifulSoup(
            response, "html.parser", from_encoding=response.info().get_param("charset")
        )

    def _open_url_gentle(self, url, max_retries=10, timeout_s=3, error_timeout_s=60):
        try:
            time.sleep(timeout_s)
            return self._open_url(url)
        except urllib.error.HTTPError as error:
            logging.error("Error trying to query URL %s: %s", url, error)
            if max_retries < 0:
                raise error
            else:
                time.sleep(error_timeout_s)
                return self._open_url_gentle(
                    url, max_retries - 1, timeout_s, error_timeout_s
                )

    def _query_leaf_hierarchy_from(self, parent_url, parent_name, parent_code):
        logging.debug("Querying ICD9 data from %s", parent_url)
        soup = self._open_url_gentle(parent_url)

        hierarchy_df = pd.DataFrame(
            columns=[
                "parent_url",
                "parent_name",
                "parent_code",
                "child_url",
                "child_name",
                "child_code",
            ]
        )
        definition_list = soup.find_all(class_="codeHierarchyUL")[0]
        for list_item in definition_list.find_all("li"):
            child_url = self.icd9data_base_url + list_item.a["href"]
            child_name = list_item.find_all(class_="threeDigitCodeListDescription")[
                0
            ].get_text()
            child_code = list_item.a.get_text()
            hierarchy_df = hierarchy_df.append(
                {
                    "parent_url": parent_url,
                    "parent_name": parent_name,
                    "parent_code": parent_code,
                    "child_url": child_url,
                    "child_name": child_name,
                    "child_code": child_code,
                },
                ignore_index=True,
            )
        return hierarchy_df

    def _query_hierarchy_from(
        self, parent_url, parent_name, parent_code
    ) -> pd.DataFrame:
        logging.debug("Querying ICD9 data from %s", parent_url)
        soup = self._open_url_gentle(parent_url)

        hierarchy_df = pd.DataFrame(
            columns=[
                "parent_url",
                "parent_name",
                "parent_code",
                "child_url",
                "child_name",
                "child_code",
            ]
        )
        definition_list = soup.find_all(class_="definitionList")[0]
        for list_item in tqdm(
            definition_list.find_all("li"),
            desc="Parsing child codes from code " + str(parent_code),
        ):
            child_url = self.icd9data_base_url + list_item.a["href"]
            child_text = list_item.get_text()
            child_code = child_text.split(" ")[0]
            child_name = " ".join(child_text.split(" ")[1:])
            hierarchy_df = hierarchy_df.append(
                {
                    "parent_url": parent_url,
                    "parent_name": parent_name,
                    "parent_code": parent_code,
                    "child_url": child_url,
                    "child_name": child_name,
                    "child_code": child_code,
                },
                ignore_index=True,
            )
            if "-" in child_code:
                hierarchy_df = hierarchy_df.append(
                    self._query_hierarchy_from(child_url, child_name, child_code),
                    ignore_index=True,
                )
            else:
                hierarchy_df = hierarchy_df.append(
                    self._query_leaf_hierarchy_from(child_url, child_name, child_code),
                    ignore_index=True,
                )

        return hierarchy_df
