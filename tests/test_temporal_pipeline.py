import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from temporal_graph import DuckDBTemporalGraphLoader
from temporal_model import TemporalGraphClassifier


class TemporalGraphPipelineTests(unittest.TestCase):
    def test_builds_timestamped_endpoint_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "flows.csv"
            with dataset.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    (
                        "IPV4_SRC_ADDR",
                        "IPV4_DST_ADDR",
                        "L4_SRC_PORT",
                        "L4_DST_PORT",
                        "FLOW_START_MILLISECONDS",
                        "IN_BYTES",
                        "Attack",
                    )
                )
                writer.writerows(
                    (
                        ("10.0.0.1", "10.0.0.2", 1234, 443, 1_700_000_000_000, 100, "BENIGN"),
                        ("10.0.0.3", "10.0.0.2", 4321, 443, 1_700_000_000_500, 200, "DDoS"),
                        ("10.0.0.2", "10.0.0.4", 443, 53, 1_700_000_011_000, 300, "PortScan"),
                    )
                )

            snapshots = list(
                DuckDBTemporalGraphLoader(
                    dataset, window_seconds=10, fetch_size=1
                ).iter_snapshots()
            )

        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0].edge_index.shape, (2, 2))
        self.assertEqual(snapshots[0].edge_features.shape, (2, 1))
        self.assertEqual(snapshots[0].labels, ("Normal", "DDoS"))
        self.assertIn("10.0.0.1:1234", snapshots[0].nodes)
        self.assertTrue(np.isfinite(snapshots[0].edge_features).all())
        self.assertAlmostEqual(
            float(snapshots[0].edge_features[0, 0]), np.log1p(100), places=5
        )

    def test_requires_real_endpoints_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "flows.csv"
            dataset.write_text(
                "Destination Port,Label\n443,BENIGN\n80,DDoS\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "source IP"):
                list(DuckDBTemporalGraphLoader(dataset).iter_snapshots())

    def test_temporal_classifier_updates_memory_across_windows(self) -> None:
        model = TemporalGraphClassifier(2, 1, 3, hidden_dim=8)
        node_ids = ("10.0.0.1", "10.0.0.2")
        node_features = torch.zeros((2, 2))
        edge_index = torch.tensor([[0], [1]])
        edge_features = torch.ones((1, 1))

        first = model(node_ids, node_features, edge_index, edge_features)
        first_memory = model.memory["10.0.0.2"].clone()
        second = model(node_ids, node_features, edge_index, edge_features)

        self.assertEqual(first.shape, (1, 3))
        self.assertEqual(second.shape, (1, 3))
        self.assertFalse(torch.equal(first_memory, model.memory["10.0.0.2"]))
        model.reset_memory()
        self.assertEqual(model.memory, {})


if __name__ == "__main__":
    unittest.main()
