import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from traffic_optimizer.core import logger
from traffic_optimizer.quantum.quantum_optimizer import build_qubo, solve_bruteforce, solve_annealing, QUBO
from traffic_optimizer.traffic.traffic_network import TrafficNetwork

ROOT = Path(__file__).resolve().parents[1]


class OptimizerTests(unittest.TestCase):
    def test_emergency_road_serves_both_endpoints(self):
        net = TrafficNetwork(4, seed=7)
        for road in net.roads.values():
            road.queue = road.density = 0
        for route, expected in [(["I1", "I3"], 0), (["I1", "I2"], 1)]:
            with self.subTest(route=route):
                result = solve_bruteforce(build_qubo(net, route))
                for node in route:
                    self.assertEqual(result.assignment[node], expected)

    def test_annealing_matches_small_known_optimum(self):
        qubo = QUBO(["a", "b"], {0: -5, 1: 3}, {(0, 1): 1}, 4)
        result = solve_annealing(qubo, seed=2)
        self.assertEqual(result.assignment, {"a": 1, "b": 0})
        self.assertEqual(result.energy, solve_bruteforce(qubo).energy)

    def test_empty_qubo(self):
        self.assertEqual(solve_annealing(QUBO([], {}, {}, 3)).energy, 3)


class LoggerTests(unittest.TestCase):
    def test_retention_and_valid_json(self):
        with tempfile.TemporaryDirectory() as folder:
            json_path = str(Path(folder) / "events.json")
            csv_path = str(Path(folder) / "events.csv")
            with patch.object(logger, "JSON_PATH", json_path), patch.object(logger, "CSV_PATH", csv_path), patch.object(logger, "_MAX_JSON", 3), patch.object(logger, "_events", []):
                logger._ensure_csv_header()
                for i in range(6):
                    logger.log_event("TEST", "North", vehicles=i)
                self.assertEqual(len(logger._events), 3)
                saved = json.loads(Path(json_path).read_text())
                self.assertEqual([e["data"]["vehicles"] for e in saved], [3, 4, 5])
                self.assertEqual(len(Path(csv_path).read_text().splitlines()), 7)


class DashboardTests(unittest.TestCase):
    def test_network_rerun(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.title)
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(app.title)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is optional")
    def test_live_idle_rerun(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "live_app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.title)
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(app.title)


if __name__ == "__main__":
    unittest.main()
