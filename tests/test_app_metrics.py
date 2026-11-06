import unittest
from collections import Counter, deque
from unittest.mock import patch

import app


class DashboardMetricTests(unittest.TestCase):
    def test_dashboard_template_renders_new_metrics(self) -> None:
        response = app.app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"rocChart", response.data)
        self.assertIn(b"speedControl", response.data)

    def test_streaming_roc_auc_uses_bounded_histograms(self) -> None:
        histogram = {
            "Normal": {
                "positive": [0] * 90 + [1] + [0] * 9,
                "negative": [1] + [0] * 99,
            },
            "DDoS": {
                "positive": [0] * 80 + [1] + [0] * 19,
                "negative": [0] * 20 + [1] + [0] * 79,
            },
        }
        matrix = {
            "Normal": Counter({"Normal": 1, "DDoS": 0}),
            "DDoS": Counter({"Normal": 0, "DDoS": 1}),
        }
        with patch.dict(
            app.state,
            {
                "categories": ["Normal", "DDoS"],
                "confusion_matrix": matrix,
                "roc_histogram": histogram,
                "performance_samples": deque([(10, 0.5)]),
            },
        ):
            result = app._evaluation_snapshot()

        self.assertEqual(result["roc_auc_ovr_macro"], 1.0)
        self.assertEqual(len(result["roc_curve"]), 21)
        self.assertEqual(result["performance"]["throughput_fps"], 20.0)
        self.assertEqual(result["performance"]["latency_ms_per_flow"], 50.0)

    def test_neighbor_api_returns_recent_graph_layers(self) -> None:
        recent_events = deque(
            [
                {"source": "a", "destination": "b"},
                {"source": "b", "destination": "c"},
            ],
            maxlen=100,
        )
        with patch.dict(app.state, {"events": recent_events}):
            response = app.app.test_client().get("/api/neighbors?node=a&hops=2")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["layers"], [
            {"hop": 1, "nodes": ["b"]},
            {"hop": 2, "nodes": ["c"]},
        ])

    def test_speed_api_validates_and_updates_interval(self) -> None:
        with (
            patch.object(app, "_start_worker"),
            patch.dict(app.state, {"model_status": "waiting"}),
        ):
            client = app.app.test_client()
            invalid = client.post(
                "/api/control",
                json={"action": "speed", "interval_seconds": 0},
            )
            accepted = client.post(
                "/api/control",
                json={"action": "speed", "interval_seconds": 0.1},
            )

        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.get_json()["interval_seconds"], 0.1)


if __name__ == "__main__":
    unittest.main()
