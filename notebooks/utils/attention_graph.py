from pyvis.network import Network
from typing import Dict, List, Tuple, Set, Optional

from loading import load_attention_weights


def calculate_attention_importances(
    attention_weights: Dict[str, Dict[str, float]]
) -> Dict[str, List[Tuple[str, float]]]:
    attention_importances: Dict[str, List[Tuple[str, float]]] = {}
    for from_node, attention in attention_weights.items():
        for to_node, weight in attention.items():
            if to_node in attention_importances:
                attention_importances[to_node].append((from_node, float(weight)))
            else:
                attention_importances[to_node] = [(from_node, float(weight))]

    return attention_importances


def calculate_shared_attention_weights(
    attention_weights: Dict[str, Dict[str, float]],
    shared_attention_importances: Dict[str, List[Tuple[str, float]]],
) -> Dict[str, float]:
    shared_attention_weights = {k: 0.0 for k in attention_weights}
    for parent_node, connection_infos in shared_attention_importances.items():
        for connection_info in connection_infos:
            child_node = connection_info[0]
            weight = connection_info[1]
            shared_attention_weights[child_node] += weight

    return shared_attention_weights


def _node_name(node_id: str) -> str:
    if not str(node_id).isdigit():
        return node_id

    return "#" + str(node_id)


def convert_to_node_mapping(
    all_nodes: List[str], use_node_mapping: bool = True
) -> Dict[str, str]:
    node_names = list(set([_node_name(x) for x in all_nodes]))
    node_mapping = {}
    for idx in range(len(node_names)):
        if use_node_mapping:
            node_mapping[node_names[idx]] = "feature" + str(idx)
        else:
            node_mapping[node_names[idx]] = node_names[idx]
    return node_mapping


def _load_attention_data(
    run_id: str, local_mlflow_dir: str
) -> Optional[
    Tuple[
        Dict[str, Dict[str, float]],
        Dict[str, List[Tuple[str, float]]],
        Dict[str, float],
        Dict[str, List[Tuple[str, float]]],
    ]
]:
    attention_weights = load_attention_weights(run_id, local_mlflow_dir)
    if attention_weights is None:
        return None
    attention_importances = calculate_attention_importances(attention_weights)
    shared_attention_importances = {
        k: v for k, v in attention_importances.items() if len(v) > 1
    }
    shared_attention_weights = calculate_shared_attention_weights(
        attention_weights, shared_attention_importances
    )

    print("Number of features", len(attention_weights))
    print("Total number of hidden features", len(attention_importances))
    print("Number of shared hidden features", len(shared_attention_importances))
    print(
        "Number of features with >0.5 shared embedding",
        len([k for k, v in shared_attention_weights.items() if float(v) > 0.5]),
    )
    return (
        attention_weights,
        attention_importances,
        shared_attention_weights,
        shared_attention_importances,
    )


def gather_colored_connections(
    reference_run_id: str,
    local_mlflow_dir: str,
    attention_weights: Dict[str, Dict[str, float]],
    feature_node_mapping: Dict[str, str],
) -> Set[Tuple[str, str]]:
    if reference_run_id is None:
        return set()

    reference_attention_weights = load_attention_weights(
        reference_run_id, local_mlflow_dir
    )
    if reference_attention_weights is None:
        return set()
    reference_connections = set(
        [
            (child, parent)
            for child, parents in reference_attention_weights.items()
            for parent in parents
        ]
    )
    return calculate_colored_connections(
        reference_connections, attention_weights, feature_node_mapping
    )


def calculate_colored_connections(
    reference_connections: Set[Tuple[str, str]],
    attention_weights: Dict[str, Dict[str, float]],
    feature_node_mapping: Dict[str, str],
) -> Set[Tuple[str, str]]:
    original_connections = set(
        [
            (child, parent)
            for child, parents in attention_weights.items()
            for parent in parents
        ]
    )
    colored_connections = (
        original_connections - reference_connections
        if len(original_connections) > len(reference_connections)
        else reference_connections - original_connections
    )
    return set(
        [
            (
                feature_node_mapping.get(_node_name(v[0]), _node_name(v[0])),
                feature_node_mapping.get(_node_name(v[1]), _node_name(v[1])),
            )
            for v in colored_connections
        ]
    )


def _create_graph_visualization(
    attention_weights: Dict[str, Dict[str, float]],
    threshold: float,
    run_name: str,
    node_mapping: Dict[str, str],
    colored_connections: Set[Tuple[str, str]],
):
    feature_nodes = set(
        [
            _node_name(k)
            for k, vs in attention_weights.items()
            if len([v for v in vs.values() if float(v) > threshold]) > 0
        ]
    )
    feature_nodes = set([node_mapping.get(x, x) for x in feature_nodes])

    attention_importances = calculate_attention_importances(attention_weights)
    hidden_nodes = set(
        [
            _node_name(k)
            for k, vs in attention_importances.items()
            if len([v for v in vs if v[1] > threshold]) > 0
        ]
    )
    hidden_nodes = set([node_mapping.get(x, x) for x in hidden_nodes])

    net = Network(height="100%", width="100%")
    added_nodes = set()
    for parent_node, connection_infos in attention_importances.items():
        parent_node_name = _node_name(parent_node)
        parent_node_name = node_mapping.get(parent_node_name, parent_node_name)
        if parent_node_name not in hidden_nodes:
            continue
        for connection_info in connection_infos:
            child_node_name = _node_name(connection_info[0])
            child_node_name = node_mapping.get(child_node_name, child_node_name)
            if child_node_name not in feature_nodes:
                continue

            weight = connection_info[1]
            if weight > threshold:
                if child_node_name not in added_nodes:
                    added_nodes.add(child_node_name)
                    net.add_node(
                        child_node_name, label=child_node_name, color="#7fcac0"
                    )
                if parent_node_name not in added_nodes:
                    added_nodes.add(parent_node_name)
                    net.add_node(
                        parent_node_name, label=parent_node_name, color="#7d92c3"
                    )
                if (child_node_name, parent_node_name) in colored_connections:
                    net.add_edge(
                        source=child_node_name,
                        to=parent_node_name,
                        title=weight,
                        value=weight,
                        arrowStrikethrough=False,
                        color="red",
                    )
                else:
                    net.add_edge(
                        source=child_node_name,
                        to=parent_node_name,
                        title=weight,
                        value=weight,
                        arrowStrikethrough=False,
                    )
    net.show("attention_{}_{}.html".format(run_name, str(threshold).replace(".", "")))
    return (colored_connections, node_mapping)


def create_graph_visualization(
    run_id: str,
    local_mlflow_dir: str,
    threshold: float,
    run_name: str,
    use_node_mapping: bool = True,
) -> Optional[Dict[str, str]]:
    attention_weights = load_attention_weights(run_id, local_mlflow_dir)
    if attention_weights is None:
        return None

    feature_node_mapping = convert_to_node_mapping(
        [x for x in attention_weights], use_node_mapping
    )
    return _create_graph_visualization(
        attention_weights,
        threshold=threshold,
        run_name=run_name,
        node_mapping=feature_node_mapping,
        colored_connections=set(),
    )


def create_graph_visualization_reference(
    run_id: str,
    reference_run_id: str,
    local_mlflow_dir: str,
    threshold: float,
    run_name: str,
    use_node_mapping: bool = True,
):
    attention_weights = load_attention_weights(run_id, local_mlflow_dir)
    if attention_weights is None:
        return None

    feature_node_mapping = convert_to_node_mapping(
        [x for x in attention_weights], use_node_mapping
    )
    colored_connections = gather_colored_connections(
        reference_run_id=reference_run_id,
        local_mlflow_dir=local_mlflow_dir,
        attention_weights=attention_weights,
        feature_node_mapping=feature_node_mapping,
    )

    return _create_graph_visualization(
        attention_weights,
        threshold=threshold,
        run_name=run_name,
        node_mapping=feature_node_mapping,
        colored_connections=colored_connections,
    )

