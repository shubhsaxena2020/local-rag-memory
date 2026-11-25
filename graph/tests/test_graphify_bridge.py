import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from graph.graphify_bridge import GraphifyBridge, GraphifyBridgeError, MAX_GRAPH_FILE_SIZE


class GraphifyBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.graph = self.root / "graph.json"
        self.graph.write_text(json.dumps({"nodes": [
            {"id": "src_calculator_add", "label": "add()", "source_file": "src/calculator.py"},
            {"id": "src_other", "label": "other()", "source_file": "src/other.py"},
        ], "links": []}), encoding="utf-8")

    def bridge(self):
        bridge = GraphifyBridge.__new__(GraphifyBridge)
        bridge.graphify_cmd = "graphify"
        bridge.graphify_available = True
        return bridge

    def test_query_passes_plain_question_argv(self):
        bridge = self.bridge()
        with patch("graph.graphify_bridge.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "found add", "")
            self.assertEqual(bridge.query_graph("What does add do?", self.graph), "found add")
        args = run.call_args.args[0]
        self.assertEqual(args[2], "What does add do?")
        self.assertNotIn('"What does add do?"', args)

    def test_update_uses_subprocess_cwd_without_changing_process(self):
        bridge = self.bridge()
        project = self.root / "project"
        (project / "graphify-out").mkdir(parents=True)
        (project / "graphify-out" / "graph.json").write_text('{"nodes":[]}', encoding="utf-8")
        before = Path.cwd()
        with patch("graph.graphify_bridge.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            result = bridge.update_graph(project)
        self.assertEqual(result, project.resolve() / "graphify-out" / "graph.json")
        self.assertEqual(Path.cwd(), before)
        self.assertEqual(run.call_args.kwargs["cwd"], str(project.resolve()))

    def test_missing_cli_falls_back_with_warning(self):
        bridge = self.bridge()
        bridge.graphify_available = False
        with patch("sys.stderr") as stderr:
            output = bridge.query_graph("add", self.graph)
        self.assertIn("add()", output)
        self.assertTrue(stderr.write.called)

    def test_node_association_uses_real_graph_fields(self):
        bridge = self.bridge()
        self.assertEqual(bridge.associate_fact_with_text("The add function in calculator", self.graph), ["src_calculator_add"])

    def test_validation_and_graph_size_bounds(self):
        bridge = self.bridge()
        with self.assertRaises(ValueError):
            bridge.query_graph("q", self.graph, algorithm="invalid")
        with self.assertRaises(ValueError):
            bridge.query_graph("q", self.graph, budget=0)
        oversized = self.root / "large.json"
        oversized.write_bytes(b" " * (MAX_GRAPH_FILE_SIZE + 1))
        with self.assertRaises(GraphifyBridgeError):
            bridge.associate_fact_with_text("x", oversized)
        with self.assertRaises(ValueError):
            bridge.get_nodes_for_fact([str(i) for i in range(101)], self.graph)

    def test_missing_update_cli_fails_clearly(self):
        bridge = self.bridge()
        bridge.graphify_available = False
        with self.assertRaisesRegex(GraphifyBridgeError, "not available"):
            bridge.update_graph(self.root)


if __name__ == "__main__":
    unittest.main()
