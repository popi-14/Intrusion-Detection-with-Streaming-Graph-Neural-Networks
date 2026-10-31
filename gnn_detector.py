"""Temporal GraphSAGE inference for the NF-UNSW-NB15-V2 flow dataset."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import duckdb
import numpy as np
import torch
from torch import nn


PROTOCOL_NAMES = {1: "ICMP", 6: "TCP", 17: "UDP"}
FLOW_COLUMNS = (
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "Label",
    "Attack",
)


class GraphSAGE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.message = nn.Linear(4, 12)
        self.edge_classifier = nn.Sequential(
            nn.Linear(24 + 5, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(
        self, source_nodes: torch.Tensor, target_nodes: torch.Tensor, flows: torch.Tensor
    ) -> torch.Tensor:
        source_embedding = torch.relu(self.message(source_nodes))
        target_embedding = torch.relu(self.message(target_nodes))
        return self.edge_classifier(
            torch.cat((source_embedding, target_embedding, flows), dim=1)
        ).squeeze(1)


class StreamingGNNDetector:
    """Train on an initial chronological split, then replay the remaining flows."""

    def __init__(self, path: str | Path, training_fraction: float = 0.8) -> None:
        if not 0 < training_fraction < 1:
            raise ValueError("TRAINING_FRACTION phải nằm trong khoảng 0 và 1.")
        self.path = str(path)
        self.training_fraction = training_fraction
        self.model = GraphSAGE()
        self.connection: duckdb.DuckDBPyConnection | None = None
        self.reader: Any = None
        self.training_rows = 0
        self.model_training_rows = 0
        self.calibration_rows = 0
        self.replay_rows = 0
        self.threshold = 0.5
        self.node_ids: dict[tuple[int, int], int] = {}
        self.node_keys: list[tuple[int, int]] = []
        self.node_attributes = np.empty((0, 2), dtype=np.float32)
        self.neighbor_sums = np.empty((0, 2), dtype=np.float32)
        self.neighbor_counts = np.empty(0, dtype=np.int64)
        self.node_degrees = np.empty(0, dtype=np.int64)
        self.edge_pairs: set[tuple[int, int]] = set()
        self.active_node_ids: set[int] = set()

    @staticmethod
    def _flow_features(row: tuple[Any, ...]) -> np.ndarray:
        return np.asarray(
            (
                min(1.0, math.log1p(max(0, row[3])) / 18.0),
                min(1.0, math.log1p(max(0, row[4])) / 18.0),
                min(1.0, math.log1p(max(0, row[5])) / 11.0),
                min(1.0, math.log1p(max(0, row[6])) / 11.0),
                min(1.0, math.log1p(max(0, row[7])) / 18.0),
            ),
            dtype=np.float32,
        )

    def _node_context(self, node_id: int) -> np.ndarray:
        count = self.neighbor_counts[node_id]
        if count == 0:
            return np.zeros(2, dtype=np.float32)
        return self.neighbor_sums[node_id] / count

    def _observe_edge(self, source_id: int, target_id: int) -> bool:
        edge = (source_id, target_id)
        is_new = edge not in self.edge_pairs
        self.edge_pairs.add(edge)
        self.active_node_ids.update((source_id, target_id))
        self.node_degrees[source_id] += 1
        self.node_degrees[target_id] += 1
        self.neighbor_sums[target_id] += self.node_attributes[source_id]
        self.neighbor_counts[target_id] += 1
        return is_new

    def _read_training_rows(
        self, progress: Callable[[int, int], None] | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.connection is not None
        total_rows = self.connection.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [self.path]
        ).fetchone()[0]
        self.training_rows = int(total_rows * self.training_fraction)
        self.replay_rows = total_rows - self.training_rows
        if self.training_rows < 2 or self.replay_rows < 1:
            raise ValueError("Dataset cần ít nhất 3 flow để chia train và replay.")

        node_rows = self.connection.execute(
            """
            SELECT port, protocol
            FROM (
                SELECT L4_SRC_PORT AS port, PROTOCOL AS protocol
                FROM read_parquet(?)
                UNION
                SELECT L4_DST_PORT AS port, PROTOCOL AS protocol
                FROM read_parquet(?)
            )
            ORDER BY port, protocol
            """,
            [self.path, self.path],
        ).fetchall()
        self.node_ids = {
            (int(port), int(protocol)): index
            for index, (port, protocol) in enumerate(node_rows)
        }
        self.node_keys = [(int(port), int(protocol)) for port, protocol in node_rows]
        self.node_attributes = np.asarray(
            [
                (port / 65535.0, protocol / 255.0)
                for port, protocol in self.node_ids
            ],
            dtype=np.float32,
        )
        self.neighbor_sums = np.zeros_like(self.node_attributes)
        self.neighbor_counts = np.zeros(len(node_rows), dtype=np.int64)
        self.node_degrees = np.zeros(len(node_rows), dtype=np.int64)

        query = (
            f"SELECT {', '.join(FLOW_COLUMNS)}, file_row_number "
            "FROM read_parquet(?, file_row_number=true)"
        )
        self.reader = self.connection.execute(query, [self.path])
        node_features = np.empty((self.training_rows, 2, 4), dtype=np.float32)
        flow_features = np.empty((self.training_rows, 5), dtype=np.float32)
        labels = np.empty(self.training_rows, dtype=np.float32)
        processed = 0

        while processed < self.training_rows:
            rows = self.reader.fetchmany(min(8192, self.training_rows - processed))
            if not rows:
                raise ValueError("Parquet kết thúc trước số dòng train đã khai báo.")
            for row in rows:
                if int(row[10]) != processed:
                    raise ValueError("DuckDB không giữ thứ tự dòng gốc của Parquet.")
                source_id = self.node_ids[(int(row[0]), int(row[2]))]
                target_id = self.node_ids[(int(row[1]), int(row[2]))]
                source_offset = processed
                node_features[source_offset, 0, :2] = self.node_attributes[source_id]
                node_features[source_offset, 0, 2:] = self._node_context(source_id)
                node_features[source_offset, 1, :2] = self.node_attributes[target_id]
                node_features[source_offset, 1, 2:] = self._node_context(target_id)
                flow_features[source_offset] = self._flow_features(row)
                labels[source_offset] = float(row[8])
                self._observe_edge(source_id, target_id)
                processed += 1
            if progress is not None:
                progress(processed, self.training_rows)

        return node_features, flow_features, labels

    def fit(
        self,
        epochs: int = 2,
        batch_size: int = 4096,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, int]:
        if not Path(self.path).is_file():
            raise FileNotFoundError(f"Không tìm thấy dataset Parquet: {self.path}")

        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        torch.manual_seed(42)
        self.connection = duckdb.connect()
        node_features, flow_features, labels = self._read_training_rows(progress)
        self.model_training_rows = int(self.training_rows * 0.875)
        self.calibration_rows = self.training_rows - self.model_training_rows
        training_labels = labels[: self.model_training_rows]
        positives = int(training_labels.sum())
        negatives = self.model_training_rows - positives
        if positives == 0 or negatives == 0:
            raise ValueError("Tập train phải có cả flow benign và flow bất thường.")

        source_tensor = torch.from_numpy(node_features[: self.model_training_rows, 0, :])
        target_tensor = torch.from_numpy(node_features[: self.model_training_rows, 1, :])
        flow_tensor = torch.from_numpy(flow_features[: self.model_training_rows])
        label_tensor = torch.from_numpy(training_labels)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        positive_weight = torch.tensor([min(100.0, negatives / positives)])
        loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

        self.model.train()
        for _ in range(epochs):
            order = torch.randperm(self.model_training_rows)
            for start in range(0, self.model_training_rows, batch_size):
                indices = order[start : start + batch_size]
                optimizer.zero_grad()
                logits = self.model(
                    source_tensor[indices], target_tensor[indices], flow_tensor[indices]
                )
                loss = loss_function(logits, label_tensor[indices])
                loss.backward()
                optimizer.step()

        validation_source = torch.from_numpy(
            node_features[self.model_training_rows :, 0, :]
        )
        validation_target = torch.from_numpy(
            node_features[self.model_training_rows :, 1, :]
        )
        validation_flows = torch.from_numpy(flow_features[self.model_training_rows :])
        validation_labels = labels[self.model_training_rows :]
        validation_scores: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, self.calibration_rows, batch_size):
                logits = self.model(
                    validation_source[start : start + batch_size],
                    validation_target[start : start + batch_size],
                    validation_flows[start : start + batch_size],
                )
                validation_scores.append(torch.sigmoid(logits).numpy())
        scores = np.concatenate(validation_scores)
        order = np.argsort(scores)[::-1]
        sorted_scores = scores[order]
        sorted_labels = validation_labels[order]
        true_positives = np.cumsum(sorted_labels)
        false_positives = np.cumsum(1 - sorted_labels)
        false_negatives = int(validation_labels.sum()) - true_positives
        denominator = 2 * true_positives + false_positives + false_negatives
        f1_scores = np.divide(
            2 * true_positives,
            denominator,
            out=np.zeros_like(true_positives, dtype=np.float64),
            where=denominator != 0,
        )
        thresholds = np.flatnonzero(
            np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
        )
        best = thresholds[np.argmax(f1_scores[thresholds])]
        self.threshold = float(sorted_scores[best])

        del node_features, flow_features, labels
        del source_tensor, target_tensor, flow_tensor, label_tensor
        del validation_source, validation_target, validation_flows
        self.model.eval()
        return {
            "training_rows": self.model_training_rows,
            "training_anomalies": positives,
            "calibration_rows": self.calibration_rows,
            "threshold": self.threshold,
            "nodes": len(self.node_ids),
            "edges": len(self.edge_pairs),
            "replay_rows": self.replay_rows,
        }

    def iter_predictions(self, batch_size: int = 128) -> Iterator[list[dict[str, Any]]]:
        if self.connection is None or self.reader is None:
            raise RuntimeError("GNN chưa được huấn luyện.")

        remaining = self.replay_rows
        sequence = 0
        while remaining:
            rows = self.reader.fetchmany(min(batch_size, remaining))
            if not rows:
                raise ValueError("Parquet kết thúc trước khi replay hoàn tất.")
            remaining -= len(rows)
            source_features = np.empty((len(rows), 4), dtype=np.float32)
            target_features = np.empty((len(rows), 4), dtype=np.float32)
            flow_features = np.empty((len(rows), 5), dtype=np.float32)
            source_ids: list[int] = []
            target_ids: list[int] = []
            new_edges: list[bool] = []

            for index, row in enumerate(rows):
                if int(row[10]) != self.training_rows + sequence + index:
                    raise ValueError("DuckDB không giữ thứ tự dòng gốc của Parquet.")
                source_id = self.node_ids[(int(row[0]), int(row[2]))]
                target_id = self.node_ids[(int(row[1]), int(row[2]))]
                source_ids.append(source_id)
                target_ids.append(target_id)
                source_features[index, :2] = self.node_attributes[source_id]
                source_features[index, 2:] = self._node_context(source_id)
                target_features[index, :2] = self.node_attributes[target_id]
                target_features[index, 2:] = self._node_context(target_id)
                flow_features[index] = self._flow_features(row)
                new_edges.append(self._observe_edge(source_id, target_id))

            with torch.inference_mode():
                probabilities = torch.sigmoid(
                    self.model(
                        torch.from_numpy(source_features),
                        torch.from_numpy(target_features),
                        torch.from_numpy(flow_features),
                    )
                ).tolist()

            batch: list[dict[str, Any]] = []
            for index, row in enumerate(rows):
                sequence += 1
                score = float(probabilities[index])
                attack = str(row[9])
                batch.append(
                    {
                        "id": sequence,
                        "row": self.training_rows + sequence,
                        "timestamp": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                        "source": self.node_label(source_ids[index]),
                        "destination": self.node_label(target_ids[index]),
                        "protocol": self.protocol_label(int(row[2])),
                        "destination_port": int(row[1]),
                        "source_port": int(row[0]),
                        "packets": int(row[5]) + int(row[6]),
                        "bytes": int(row[3]) + int(row[4]),
                        "category": (
                            "Anomaly" if score >= self.threshold else "Normal Traffic"
                        ),
                        "score": round(score, 4),
                        "status": "ALERT" if score >= self.threshold else "NORMAL",
                        "attack": attack,
                        "label": int(row[8]),
                        "new_edge": new_edges[index],
                    }
                )
            yield batch

    def active_nodes(self) -> set[str]:
        return {self.node_label(node_id) for node_id in self.active_node_ids}

    def top_node_labels(self, limit: int) -> list[str]:
        ranked = sorted(
            self.active_node_ids,
            key=lambda node_id: self.node_degrees[node_id],
            reverse=True,
        )
        return [self.node_label(node_id) for node_id in ranked[:limit]]

    def node_label(self, node_id: int) -> str:
        port, protocol = self.node_keys[node_id]
        return f"{port}/{self.protocol_label(protocol)}"

    @staticmethod
    def protocol_label(protocol: int) -> str:
        return PROTOCOL_NAMES.get(protocol, f"IP-{protocol}")
