"""
Graphify bridge for AI Memory System.
Provides functions to associate fact records with graphify node IDs and query graph nodes.
Handles graphify update CWD manifest quirk by changing directory temporarily.
Degrades gracefully with warnings when graph is unavailable or CLI fails.
"""
import os
import re
import subprocess
import sys
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

try:
    import networkx as nx
except ImportError:
    nx = None

# Constants for bounds
MAX_QUESTION_LENGTH = 5000
MAX_TEXT_LENGTH = 5000
ALLOWED_ALGORITHMS = {"bfs", "dfs"}
MIN_BUDGET = 1
MAX_BUDGET = 5000
MAX_GRAPH_FILE_SIZE = 10 * 1024 * 1024  # 10 MiB
MAX_NODE_IDS_PER_CALL = 100

GRAPHIFY_OUT_DIR = "graphify-out"
GRAPH_JSON_FILE = "graph.json"
MANIFEST_FILE = "manifest.json"


class GraphifyBridgeError(RuntimeError):
    """Raised when graphify bridge encounters an error."""


class GraphifyBridge:
    """Bridge between AI Memory System facts and graphify code graphs."""

    def __init__(self, graphify_cmd: str = "graphify"):
        """
        Initialize the bridge.

        Args:
            graphify_cmd: Path or name of graphify executable. Defaults to 'graphify' (assumes in PATH).
        """
        self.graphify_cmd = graphify_cmd
        self.graphify_available = True
        try:
            self._verify_graphify_installed()
        except GraphifyBridgeError as e:
            # Graphify not available or failed; we'll degrade gracefully
            self.graphify_available = False
            print(f"Warning: Graphify not available or failed: {e}. Falling back to networkx if available.", file=sys.stderr)

    def _verify_graphify_installed(self) -> None:
        """Check that graphify is available in the system."""
        try:
            result = subprocess.run(
                [self.graphify_cmd, "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise GraphifyBridgeError(
                    f"Graphify command failed: {result.stderr.strip()}"
                )
        except FileNotFoundError:
            raise GraphifyBridgeError(
                f"Graphify not found. Ensure '{self.graphify_cmd}' is in PATH or provide full path."
            )
        except subprocess.TimeoutExpired:
            raise GraphifyBridgeError("Graphify help command timed out.")

    def _validate_query_args(self, question: str, algorithm: str, budget: int) -> None:
        """Validate query arguments according to bounds."""
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if len(question) > MAX_QUESTION_LENGTH:
            raise ValueError(f"question length exceeds maximum of {MAX_QUESTION_LENGTH}")
        if algorithm not in ALLOWED_ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALLOWED_ALGORITHMS}")
        if not isinstance(budget, int):
            raise TypeError("budget must be an integer")
        if budget < MIN_BUDGET or budget > MAX_BUDGET:
            raise ValueError(f"budget must be between {MIN_BUDGET} and {MAX_BUDGET}")

    def update_graph(self, project_path: Union[str, Path]) -> Path:
        """
        Run graphify update on the given project path, handling CWD manifest quirk.
        Uses subprocess cwd parameter instead of changing global state.

        Args:
            project_path: Path to the project directory to update.

        Returns:
            Path to the generated graph.json file.

        Raises:
            GraphifyBridgeError: If graphify update fails.
        """
        if not self.graphify_available:
            raise GraphifyBridgeError("Graphify is not available; cannot update graph.")

        project_path = Path(project_path).resolve()
        if not project_path.is_dir():
            raise GraphifyBridgeError(f"Project path does not exist: {project_path}")

        # Run graphify update with cwd set to project_path
        result = subprocess.run(
            [self.graphify_cmd, "update", "."],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,  # Reasonable timeout for update
            cwd=str(project_path),
        )

        if result.returncode != 0:
            raise GraphifyBridgeError(
                f"Graphify update failed: {result.stderr.strip()}"
            )

        # Determine graph.json location (should be in graphify-out/)
        graph_path = project_path / GRAPHIFY_OUT_DIR / GRAPH_JSON_FILE
        if not graph_path.is_file():
            # Fallback: check if graph.json is in current directory (should not happen with update)
            graph_path = project_path / GRAPH_JSON_FILE
            if not graph_path.is_file():
                raise GraphifyBridgeError(
                    f"Graphify update did not produce {GRAPH_JSON_FILE} in {project_path / GRAPHIFY_OUT_DIR} or {project_path}"
                )

        # Verify manifest.json exists as well (per requirement)
        manifest_path = project_path / GRAPHIFY_OUT_DIR / MANIFEST_FILE
        if not manifest_path.is_file():
            # Not fatal, but we can warn
            print(f"Warning: Expected manifest.json not found at {manifest_path}", file=sys.stderr)

        return graph_path

    def query_graph(
        self,
        question: str,
        graph_path: Optional[Union[str, Path]] = None,
        algorithm: str = "bfs",
        budget: int = 2000,
    ) -> str:
        """
        Query the graph using graphify CLI.

        Args:
            question: Natural language question to ask the graph.
            graph_path: Path to graph.json file. If None, attempts to find graph.json in default locations.
            algorithm: Either 'bfs' (default) or 'dfs'.
            budget: Maximum output tokens (default 2000).

        Returns:
            Graphify query output as string.

        Raises:
            GraphifyBridgeError: If graphify query fails and networkx fallback is not available or fails.
        """
        # Validate arguments
        self._validate_query_args(question, algorithm, budget)

        # Determine graph path
        if graph_path is None:
            graph_path = self._find_default_graph()
        else:
            graph_path = Path(graph_path).resolve()

        if not graph_path.is_file():
            raise GraphifyBridgeError(f"Graph file not found: {graph_path}")

        # Check graph file size
        try:
            file_size = graph_path.stat().st_size
            if file_size > MAX_GRAPH_FILE_SIZE:
                raise GraphifyBridgeError(
                    f"Graph file size {file_size} exceeds maximum of {MAX_GRAPH_FILE_SIZE} bytes"
                )
        except OSError as e:
            raise GraphifyBridgeError(f"Unable to read graph file size: {e}")

        # Build command
        cmd = [
            self.graphify_cmd,
            "query",
            question,  # Pass question directly, no extra quotes
            "--graph",
            str(graph_path),
        ]
        if algorithm == "dfs":
            cmd.append("--dfs")
        if budget != 2000:
            cmd.extend(["--budget", str(budget)])

        # Try graphify if available
        graphify_error = None
        if self.graphify_available:
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    return result.stdout.strip()
                else:
                    graphify_error = GraphifyBridgeError(
                        f"Graphify query failed: {result.stderr.strip()}"
                    )
            except subprocess.TimeoutExpired:
                graphify_error = GraphifyBridgeError("Graphify query command timed out.")
        else:
            graphify_error = GraphifyBridgeError("Graphify not available")

        # Fallback to networkx if available
        if nx is not None:
            # Explicit warning on degraded/missing path
            print(f"Warning: Falling back to networkx due to: {graphify_error}", file=sys.stderr)
            return self._query_with_networkx(question, graph_path, algorithm)
        else:
            raise GraphifyBridgeError(
                f"Graphify query failed and networkx not available: {graphify_error}"
            )
    def _find_default_graph(self) -> Path:
        """
        Look for graph.json in common locations relative to current working directory.
        Returns first found.

        Returns:
            Path to graph.json.

        Raises:
            GraphifyBridgeError: If no graph.json found.
        """
        cwd = Path.cwd()
        # Common possibilities
        candidates = [
            cwd / GRAPHIFY_OUT_DIR / GRAPH_JSON_FILE,
            cwd / GRAPH_JSON_FILE,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise GraphifyBridgeError(
            f"Could not find {GRAPH_JSON_FILE} in {cwd} or {cwd / GRAPHIFY_OUT_DIR}"
        )

    def _query_with_networkx(
        self, question: str, graph_path: Path, algorithm: str
    ) -> str:
        """
        Fallback query using networkx when graphify CLI fails.
        Very basic implementation: returns nodes matching question in node labels.

        Args:
            question: Search question.
            graph_path: Path to graph.json.
            algorithm: Unused in this fallback, kept for interface compatibility.

        Returns:
            String description of matching nodes and their edges.
        """
        try:
            data = json.loads(graph_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise GraphifyBridgeError(f"Failed to load graph file: {e}")

        # Build networkx graph (supports both directed and undirected)
        directed = data.get("directed", False)
        G = nx.DiGraph() if directed else nx.Graph()

        # Add nodes
        for node in data.get("nodes", []):
            node_id = node.get("id")
            label = node.get("label", "")
            if node_id is not None:
                attributes = dict(node)
                attributes["label"] = label
                G.add_node(node_id, **attributes)

        # Add edges
        for edge in data.get("links", []):
            source = edge.get("source")
            target = edge.get("target")
            if source is not None and target is not None:
                G.add_edge(source, target, **edge)

        # Simple search: find nodes where label contains question (case-insensitive)
        matches = []
        for node_id, node_data in G.nodes(data=True):
            label = node_data.get("label", "")
            if question.lower() in label.lower():
                matches.append((node_id, label))

        if not matches:
            return f"No nodes found matching question: {question}"

        # Build descriptive output
        lines = [f"Found {len(matches)} node(s) matching '{question}':"]
        for node_id, label in matches:
            lines.append(f"- Node {node_id}: {label}")
            # List a few neighbors
            neighbors = list(G.neighbors(node_id))
            if neighbors:
                neighbor_labels = []
                for nid in neighbors[:5]:  # Limit to 5 neighbors
                    nlabel = G.nodes[nid].get("label", "")
                    neighbor_labels.append(f"{nid} ({nlabel})")
                lines.append(f"  Neighbors: {', '.join(neighbor_labels)}")
                if len(neighbors) > 5:
                    lines.append(f"  ... and {len(neighbors) - 5} more")
            else:
                lines.append("  Neighbors: None")

        return "\n".join(lines)

    def associate_fact_with_text(
        self, fact_text: str, graph_path: Optional[Union[str, Path]] = None
    ) -> List[str]:
        """
        Associate a fact text with graphify node IDs by loading bounded graph.json
        and matching bounded tokens against node id, label, and source_file.
        Returns list of node IDs that appear relevant to the fact text.
        """
        if not isinstance(fact_text, str):
            raise TypeError("fact_text must be a string")
        if not fact_text.strip():
            return []

        # Validate fact_text length
        if len(fact_text) > MAX_TEXT_LENGTH:
            raise ValueError(f"fact_text length exceeds maximum of {MAX_TEXT_LENGTH}")

        # Determine graph path
        if graph_path is None:
            graph_path = self._find_default_graph()
        else:
            graph_path = Path(graph_path).resolve()

        if not graph_path.is_file():
            # If graph not available, return empty list (degrade gracefully) with warning
            print(f"Warning: Graph file not found at {graph_path}. Returning empty association.", file=sys.stderr)
            return []

        # Check graph file size
        try:
            file_size = graph_path.stat().st_size
            if file_size > MAX_GRAPH_FILE_SIZE:
                raise GraphifyBridgeError(
                    f"Graph file size {file_size} exceeds maximum of {MAX_GRAPH_FILE_SIZE} bytes"
                )
        except OSError as e:
            raise GraphifyBridgeError(f"Unable to read graph file size: {e}")

        # Load graph data
        try:
            data = json.loads(graph_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Failed to load graph file {graph_path}: {e}. Returning empty association.", file=sys.stderr)
            return []

        # Extract tokens from fact_text: split by non-alphanumeric, keep alphanumeric sequences
        tokens = re.findall(r"[A-Za-z0-9_]+", fact_text)
        # Also add lowercase versions for case-insensitive matching
        tokens_lower = [t.lower() for t in tokens if len(t) > 2]
        # Remove duplicates while preserving order (case-insensitive)
        seen = set()
        unique_tokens = []
        for t in tokens_lower:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(t)

        # Match tokens against node id, label, source_file
        matched_node_ids = set()
        for node in data.get("nodes", []):
            node_id = node.get("id")
            label = node.get("label", "")
            source_file = node.get("source_file", "")
            # Convert to string for matching
            id_str = str(node_id) if node_id is not None else ""
            label_str = str(label)
            source_str = str(source_file)

            node_tokens = set(re.findall(r"[A-Za-z0-9_]+", f"{id_str} {label_str} {source_str}".lower()))
            if node_id is not None and set(unique_tokens).intersection(node_tokens):
                matched_node_ids.add(str(node_id))

        # Limit to MAX_NODE_IDS_PER_CALL
        if len(matched_node_ids) > MAX_NODE_IDS_PER_CALL:
            # Take first MAX_NODE_IDS_PER_CALL (sorted for determinism)
            matched_node_ids = set(sorted(matched_node_ids)[:MAX_NODE_IDS_PER_CALL])

        return sorted(list(matched_node_ids))
    def get_nodes_for_fact(self, fact_node_ids: List[str], graph_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Retrieve detailed node information for a list of node IDs associated with a fact.

        Args:
            fact_node_ids: List of node IDs (as strings) from the fact record.
            graph_path: Path to graph.json. If None, uses default search.

        Returns:
            Dictionary mapping node ID to node data (from graph.json).
        """
        if not isinstance(fact_node_ids, list):
            raise TypeError("fact_node_ids must be a list")
        if len(fact_node_ids) > MAX_NODE_IDS_PER_CALL:
            raise ValueError(f"fact_node_ids cannot exceed {MAX_NODE_IDS_PER_CALL} entries")
        if any(not isinstance(node_id, str) or not node_id or len(node_id) > 256 for node_id in fact_node_ids):
            raise ValueError("fact_node_ids must contain non-empty strings of at most 256 characters")
        if not fact_node_ids:
            return {}

        try:
            graph_path = self._find_default_graph() if graph_path is None else Path(graph_path).resolve()
            if not graph_path.is_file() or graph_path.stat().st_size > MAX_GRAPH_FILE_SIZE:
                raise GraphifyBridgeError("Graph file is missing or exceeds the 10 MiB limit")
            data = json.loads(graph_path.read_text(encoding="utf-8"))
        except (GraphifyBridgeError, json.JSONDecodeError, OSError) as e:
            # Degrade gracefully: return empty dict
            return {}

        # Build lookup: node ID -> node data
        nodes_by_id = {}
        for node in data.get("nodes", []):
            nid = str(node.get("id"))
            if nid in fact_node_ids:
                nodes_by_id[nid] = node

        return nodes_by_id


def _demo() -> None:
    """Demonstration of the bridge functionality (not run in production)."""
    print("Graphify Bridge Demo")
    print("=" * 50)

    bridge = GraphifyBridge()

    # Example: update scratch project graph
    scratch_project = Path(r"C:\Users\shubh\ai-memory-system\graph\scratch-project")
    if scratch_project.is_dir():
        print(f"Updating graph for scratch project: {scratch_project}")
        try:
            graph_path = bridge.update_graph(scratch_project)
            print(f"Graph updated successfully: {graph_path}")
        except GraphifyBridgeError as e:
            print(f"Warning: Graph update failed: {e}")
            # Try to find existing graph
            try:
                graph_path = bridge._find_default_graph()
                print(f"Using existing graph: {graph_path}")
            except GraphifyBridgeError:
                print("No graph available for demo.")
                return
    else:
        print(f"Scratch project not found: {scratch_project}")
        # Try to find any graph
        try:
            graph_path = bridge._find_default_graph()
            print(f"Using existing graph: {graph_path}")
        except GraphifyBridgeError:
            print("No graph available for demo.")
            return

    # Example query
    question = "What does the add function do?"
    print(f"\nQuerying graph with: '{question}'")
    try:
        answer = bridge.query_graph(question, graph_path=graph_path)
        print("Answer:")
        print(answer)
    except GraphifyBridgeError as e:
        print(f"Query failed: {e}")

    # Example association
    fact_text = "The add function returns the sum of two numbers."
    print(f"\nAssociating fact with graph: '{fact_text}'")
    try:
        node_ids = bridge.associate_fact_with_text(fact_text, graph_path=graph_path)
        print(f"Associated node IDs: {node_ids}")
        if node_ids:
            nodes = bridge.get_nodes_for_fact(node_ids, graph_path=graph_path)
            print("Node details:")
            for nid, node in nodes.items():
                print(f"  {nid}: {node.get('label', 'No label')}")
    except GraphifyBridgeError as e:
        print(f"Association failed: {e}")


if __name__ == "__main__":
    # Run demo only when executed directly
    _demo()
