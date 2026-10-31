import csv
import tempfile
import unittest
from pathlib import Path

from gnn_detector import StreamingGNNDetector


class StreamingGNNDetectorTests(unittest.TestCase):
    def test_normalizes_alias_labels_and_non_finite_features(self) -> None:
        rows = [
            ("BENIGN", 53, "12", "Infinity"),
            ("benign", 53, "14", "100"),
            ("BENIGN", 80, "16", "200"),
            ("benign", 80, "NaN", "300"),
            ("DDoS", 443, "20", "400"),
            ("DDoS", 443, "22", "500"),
            ("PortScan", 22, "24", "600"),
            ("PortScan", 22, "26", "700"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "flows.csv"
            with dataset.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    (
                        "Label",
                        "Destination Port",
                        "Total Length of Fwd Packets",
                        "Flow Bytes/s",
                    )
                )
                writer.writerows(rows)

            detector = StreamingGNNDetector(dataset, training_fraction=0.75)
            try:
                summary = detector.fit(epochs=1, batch_size=4)
                predictions = [
                    event
                    for batch in detector.iter_predictions(batch_size=2)
                    for event in batch
                ]
            finally:
                if detector.connection is not None:
                    detector.connection.close()

        self.assertEqual(summary["categories"], ["Normal", "DDoS", "PortScan"])
        self.assertEqual(summary["training_rows"], 5)
        self.assertEqual(summary["replay_rows"], 3)
        self.assertEqual(len(predictions), 3)
        for event in predictions:
            self.assertAlmostEqual(sum(event["confidence_scores"].values()), 1.0, places=3)
            self.assertGreaterEqual(event["bytes"], 0)


if __name__ == "__main__":
    unittest.main()
