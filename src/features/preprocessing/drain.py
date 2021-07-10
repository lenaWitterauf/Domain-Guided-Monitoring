import re
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import logging
from .base import Preprocessor

### Original source code from
### https://appsrv.cse.cuhk.edu.hk/~pjhe/Drain.py
class DrainLogCluster:
	def __init__(self, log_template: List[str] = [], log_ids: List[int] = []):
		self.log_template = log_template
		self.log_ids = log_ids
		self.parent_node: Optional[DrainNode] = None


class DrainNode:
	def __init__(self, depth: int = 0, token: str = "", path: List[str] = []):
		self.depth = depth
		self.token = token
		self.path = path
		self.child_mapping: Dict[str, DrainNode] = dict()
		self.child_clusters: List[DrainLogCluster] = []


class DrainParameters:
	def __init__(
		self,
		rex: List[Tuple[str, str]] = [],
		depth=4,
		st=0.4,
		maxChild=100,
		removable=True,
		removeCol=None,
	):
		self.depth = depth
		self.st = st
		self.maxChild = maxChild
		self.removable = removable
		self.removeCol = removeCol
		self.rex = rex


class Drain(Preprocessor):
	def __init__(self, parameters: DrainParameters,
		data_df: pd.DataFrame,
		data_df_column_name: str):
		self.parameters = parameters
		self.data_df = data_df
		self.data_df_column_name = data_df_column_name

	def load_data(self) -> pd.DataFrame:
		_, clusters = self._load_data()
		return self._to_cluster_df(clusters)

	def _has_numbers(self, s: str):
		return any(char.isdigit() for char in s)

	def _is_version_string(self, s: str):
		return (s[0] == "v") and all(c.isdigit() for c in s[1:].replace(".", ""))

	def _tree_search(
		self, root_node: DrainNode, sequence: List[str]
	) -> Optional[DrainLogCluster]:
		sequence_length = len(sequence)
		sequence_length_str = str(sequence_length)
		if sequence_length_str not in root_node.child_mapping:
			return None

		parent_node: DrainNode = root_node.child_mapping[sequence_length_str]
		current_depth = 1
		for token in sequence:
			if (
				current_depth >= self.parameters.depth
				or current_depth > sequence_length
			):
				break

			if token in parent_node.child_mapping:
				parent_node = parent_node.child_mapping[token]
			elif "*" in parent_node.child_mapping:
				parent_node = parent_node.child_mapping["*"]
			else:
				return None
			current_depth += 1

		potential_log_clusters = parent_node.child_clusters
		return self._fast_match(potential_log_clusters, sequence)

	def _find_or_add_parent_node(
		self, first_layer_node: DrainNode, log_cluster: DrainLogCluster
	) -> DrainNode:
		parent_node = first_layer_node
		current_depth = 1
		for template_idx in range(len(log_cluster.log_template)):
			current_depth = template_idx + 1
			if current_depth >= self.parameters.depth:
				break

			token = log_cluster.log_template[template_idx]
			if token in parent_node.child_mapping:
				parent_node = parent_node.child_mapping[token]
				continue

			should_be_added_to_whitecard_token = (
				self._has_numbers(token) and not self._is_version_string(token)
				or "*" in parent_node.child_mapping and len(parent_node.child_mapping) >= self.parameters.maxChild
				or "*" not in parent_node.child_mapping and len(parent_node.child_mapping) + 1 >= self.parameters.maxChild
			)
			token_to_add = "*" if should_be_added_to_whitecard_token else token
			if token_to_add not in parent_node.child_mapping:
				parent_node.child_mapping[token_to_add] = DrainNode(depth=current_depth + 1, token=token_to_add, path=parent_node.path + [token_to_add])
			parent_node = parent_node.child_mapping[token_to_add]
			
		return parent_node

	def _add_cluster_to_prefix_tree(
		self, root_node: DrainNode, log_cluster: DrainLogCluster
	):
		sequence_length_str = str(len(log_cluster.log_template))
		if sequence_length_str not in root_node.child_mapping:
			root_node.child_mapping[sequence_length_str] = DrainNode(depth=1, token=sequence_length_str, path=[sequence_length_str])
		first_layer_node = root_node.child_mapping[sequence_length_str]

		parent_node = self._find_or_add_parent_node(first_layer_node=first_layer_node, log_cluster=log_cluster)
		log_cluster.parent_node = parent_node
		if len(parent_node.child_clusters) == 0:
			parent_node.child_clusters = [log_cluster]
		else:
			parent_node.child_clusters.append(log_cluster)

	def _calculate_sequence_distance(
		self, template: List[str], sequence: List[str]
	) -> Tuple[float, int]:
		assert len(template) == len(sequence)

		num_sim_tokens = 0
		num_parameters = 0
		for token1, token2 in zip(template, sequence):
			if token1 == "*":
				num_parameters += 1
				continue
			if token1 == token2:
				num_sim_tokens += 1

		similarity = float(num_sim_tokens) / len(template)
		return similarity, num_parameters

	def _fast_match(
		self, log_clusters: List[DrainLogCluster], sequence: List[str]
	) -> Optional[DrainLogCluster]:
		max_similarity = -1.0
		max_num_parameters = -1
		max_cluster = None

		for cluster in log_clusters:
			(
				cluster_similarity,
				cluster_num_parameters,
			) = self._calculate_sequence_distance(cluster.log_template, sequence)
			if cluster_similarity > max_similarity or (
				cluster_similarity == max_similarity
				and cluster_num_parameters > max_num_parameters
			):
				max_similarity = cluster_similarity
				max_num_parameters = cluster_num_parameters
				max_cluster = cluster

		if max_similarity >= self.parameters.st:
			return max_cluster
		return None

	def _get_template(self, sequence_1: List[str], sequence_2: List[str]) -> List[str]:
		assert len(sequence_1) == len(sequence_2)
		return [
			sequence_1[idx] if sequence_1[idx] == sequence_2[idx] else "*"
			for idx in range(len(sequence_1))
		]

	def _to_cluster_df(self, log_clusters: List[DrainLogCluster]) -> pd.DataFrame:
		cluster_mappings = []
		for cluster_idx in range(len(log_clusters)):
			log_cluster = log_clusters[cluster_idx]
			cluster_template = " ".join(log_cluster.log_template)
			cluster_path = " ".join(log_cluster.parent_node.path) if log_cluster.parent_node is not None else ""
			for log_id in log_cluster.log_ids:
				cluster_mappings.append(
					{
						"log_idx": log_id,
						"cluster_id": cluster_idx,
						"cluster_template": cluster_template,
						"cluster_path": cluster_path,
					}
				)

		return pd.DataFrame.from_records(cluster_mappings)

	def _str_tree_from_node(self, node: DrainNode, current_depth: int) -> str:
		out_str = ""
		for _ in range(current_depth):
			out_str += "\t"

		if node.depth == 0:
			out_str += "Root Node"
		elif node.depth == 1:
			out_str += "<" + str(node.token) + ">"
		else:
			out_str += node.token

		for child in node.child_mapping:
			out_str = (
				out_str
				+ "\n"
				+ self._str_tree_from_node(
					node.child_mapping[child], current_depth + 1
				)
			)

		return out_str

	def _load_data(
		self
	) -> Tuple[DrainNode, List[DrainLogCluster]]:
		root_node = DrainNode()
		log_clusters: List[DrainLogCluster] = []

		for idx, row in tqdm(
			self.data_df.iterrows(),
			desc="Generating DRAIN clusters from log_df",
			total=len(self.data_df),
		):
			log_id = idx
			log_line = row[self.data_df_column_name]
			for regex, replacement in self.parameters.rex:
				log_line = re.sub(regex, replacement, log_line)

			log_line_words = [
				word.lower().strip()
				for word in log_line.split()
				if len(word.strip()) > 0
			]
			if len(log_line_words) == 0:
				continue
			matched_log_cluster = self._tree_search(root_node, log_line_words)
			if matched_log_cluster is None:
				new_log_cluster = DrainLogCluster(
					log_template=log_line_words, log_ids=[log_id]
				)
				log_clusters.append(new_log_cluster)
				self._add_cluster_to_prefix_tree(root_node, new_log_cluster)
			else:
				matched_log_cluster.log_ids.append(log_id)
				matched_log_cluster.log_template = self._get_template(
					log_line_words, matched_log_cluster.log_template
				)

		return root_node, log_clusters

