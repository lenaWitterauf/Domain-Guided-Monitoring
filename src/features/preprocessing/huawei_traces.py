import logging
import pandas as pd
from tqdm import tqdm
from .base import Preprocessor
from typing import List, Dict
from pathlib import Path
import json


class HuaweiTracePreprocessor(Preprocessor):
    def __init__(
        self, trace_base_directory: Path,
    ):
        self.trace_base_directory = trace_base_directory

    def load_data(self) -> pd.DataFrame:
        logging.info(
            "Parsing Huawei traces starting from %s", self.trace_base_directory
        )
        trace_records = self._read_traces()
        trace_df = pd.DataFrame.from_records(trace_records)
        trace_df['timestamp'] = pd.to_datetime(trace_df['timestamp'], utc=True)
        return trace_df

    def _read_traces(self) -> List[Dict[str, str]]:
        trace_records: List[Dict[str, str]] = []
        for directory in self.trace_base_directory.iterdir():
            directory_name = directory.name
            directory_files = [x for x in directory.iterdir()]
            for directory_file in tqdm(
                directory_files,
                desc="Reading traces from directory {}...".format(directory_name),
            ):
                trace_records = trace_records + self._read_trace(
                    directory_file, 
                )

        return trace_records

    def _read_trace(
        self, trace_file_path: Path,
    ) -> List[Dict[str, str]]:
        with open(trace_file_path) as trace_file:
            trace_json = json.load(trace_file)

        trace_records = []
        child_ids = set()
        children = trace_json["children"]
        while len(children) > 0:
            child = children[0]
            children = children[1:]
            child_id = child["trace_id"]
            if child_id in child_ids:
                continue

            for payload_key in [
                k for k in child["info"] if k.startswith("meta.raw_payload.")
            ]:
                payload_data = child["info"][payload_key]
                flattened_data = {
                    "parent_trace_id": trace_file_path.name[
                        : len(trace_file_path.name) - len(".json")
                    ],
                    "Hostname": child["info"]["host"],
                    "timestamp": payload_data["timestamp"],
                    "trace_name": child["info"]["name"],
                    "trace_service": child["info"]["service"],
                    "trace_project": child["info"]["project"],
                    "payload": payload_data["name"],
                }
                if "etype" in payload_data["info"]:
                    flattened_data["etype"] = payload_data["info"]["etype"]
                if "request" in payload_data["info"]:
                    payload_request_info = payload_data["info"]["request"]
                    flattened_data["http_url"] = payload_request_info[
                        "path"
                    ]
                    flattened_data["http_method"] = payload_request_info[
                        "method"
                    ]
                if "function" in payload_data["info"]:
                    flattened_data["function"] = payload_data["info"][
                        "function"
                    ]["name"]
                trace_records.append(flattened_data)

            children = children + child["children"]
            child_ids.update(child_id)

        return trace_records
