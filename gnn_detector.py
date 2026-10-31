"""Temporal GraphSAGE training and streaming inference for network-flow datasets."""

from __future__ import annotations

import glob
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import duckdb
import numpy as np
import torch
from torch import nn


UNKNOWN_SOURCE_PORT = 65536
UNKNOWN_PROTOCOL = -1
PROTOCOL_NAMES = {1: "ICMP", 6: "TCP", 17: "UDP"}
TRAINING_BATCH_SIZE = 4096
TRAINING_EPOCHS = 3
FLOW_BASE_COLUMNS = (
    "source_port",
    "destination_port",
    "protocol",
)
FLOW_FEATURE_SPECS = (
    ("in_bytes", 18.0, ("IN_BYTES", "Total Length of Fwd Packets")),
    ("out_bytes", 18.0, ("OUT_BYTES", "Total Length of Bwd Packets")),
    ("in_packets", 11.0, ("IN_PKTS", "Total Fwd Packets")),
    ("out_packets", 11.0, ("OUT_PKTS", "Total Backward Packets")),
    ("duration", 18.0, ("FLOW_DURATION_MILLISECONDS", "Flow Duration")),
    ("bytes_per_second", 25.0, ("SRC_TO_DST_SECOND_BYTES", "Flow Bytes/s")),
    ("packets_per_second", 20.0, ("Flow Packets/s",)),
    ("fwd_length_max", 7.0, ("Fwd Packet Length Max",)),
    ("fwd_length_mean", 7.0, ("Fwd Packet Length Mean",)),
    ("bwd_length_max", 7.0, ("Bwd Packet Length Max",)),
    ("bwd_length_mean", 7.0, ("Bwd Packet Length Mean",)),
    ("flow_iat_mean", 25.0, ("Flow IAT Mean",)),
    ("flow_iat_std", 25.0, ("Flow IAT Std",)),
    ("fwd_iat_mean", 25.0, ("Fwd IAT Mean",)),
    ("bwd_iat_mean", 25.0, ("Bwd IAT Mean",)),
    ("syn_count", 6.0, ("SYN Flag Count",)),
    ("ack_count", 6.0, ("ACK Flag Count",)),
    ("fin_count", 6.0, ("FIN Flag Count",)),
    ("rst_count", 6.0, ("RST Flag Count",)),
    ("packet_length_mean", 7.0, ("Packet Length Mean",)),
    ("packet_length_std", 7.0, ("Packet Length Std",)),
    ("packet_length_variance", 14.0, ("Packet Length Variance",)),
    ("down_up_ratio", 6.0, ("Down/Up Ratio",)),
    ("average_packet_size", 7.0, ("Average Packet Size",)),
)
FLOW_FEATURE_START = len(FLOW_BASE_COLUMNS)
FLOW_FEATURE_SCALES = tuple(spec[1] for spec in FLOW_FEATURE_SPECS)
FLOW_FEATURE_COLUMNS = tuple(spec[0] for spec in FLOW_FEATURE_SPECS)
ATTACK_COLUMN_INDEX = FLOW_FEATURE_START + len(FLOW_FEATURE_SPECS)
STREAM_ROW_INDEX = ATTACK_COLUMN_INDEX + 1
FEATURE_INDEX = {
    name: FLOW_FEATURE_START + index
    for index, name in enumerate(FLOW_FEATURE_COLUMNS)
}
FLOW_COLUMNS = (
    *FLOW_BASE_COLUMNS,
    *FLOW_FEATURE_COLUMNS,
    "attack",
    "stream_row",
)


class GraphSAGE(nn.Module):
    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.message = nn.Linear(4, 32)
        self.edge_classifier = nn.Sequential(
            nn.Linear(64 + len(FLOW_FEATURE_SCALES), 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, class_count),
        )

    def forward(
        self, source_nodes: torch.Tensor, target_nodes: torch.Tensor, flows: torch.Tensor
    ) -> torch.Tensor:
        source_embedding = torch.relu(self.message(source_nodes))
        target_embedding = torch.relu(self.message(target_nodes))
        return self.edge_classifier(
            torch.cat((source_embedding, target_embedding, flows), dim=1)
        )


class StreamingGNNDetector:
    """Train a multi-class model, then replay a deterministic holdout stream."""

    def __init__(self, path: str | Path, training_fraction: float = 0.8) -> None:
        if not 0 < training_fraction < 1:
            raise ValueError("TRAINING_FRACTION phải nằm trong khoảng 0 và 1.")
        self.path = str(path)
        self.training_fraction = training_fraction
        self.model: GraphSAGE | None = None
        self.connection: duckdb.DuckDBPyConnection | None = None
        self.training_reader: Any = None
        self.replay_reader: Any = None
        self.replay_query: str | None = None
        self.training_rows = 0
        self.replay_rows = 0
        self.categories: list[str] = []
        self.available_categories: list[str] = []
        self.training_class_counts: dict[str, int] = {}
        self.category_indices: dict[str, int] = {}
        self.raw_label_categories: dict[str, str] = {}
        self.active_class_mask: torch.Tensor | None = None
        self.source_files: list[str] = []
        self.source_relation = ""
        self.input_columns: dict[str, str] = {}
        self.node_ids: dict[tuple[int, int], int] = {}
        self.node_keys: list[tuple[int, int]] = []
        self.node_attributes = np.empty((0, 2), dtype=np.float32)
        self.neighbor_sums = np.empty((0, 2), dtype=np.float32)
        self.neighbor_counts = np.empty(0, dtype=np.int64)
        self.node_degrees = np.empty(0, dtype=np.int64)
        self.edge_pairs: set[tuple[int, int]] = set()
        self.active_node_ids: set[int] = set()

    @staticmethod
    def _attack_category(value: Any) -> str:
        """Keep attack classes distinct while normalizing the benign and bad-byte labels."""
        if value is None:
            raise ValueError("Dataset có flow không chứa nhãn tấn công.")
        category = str(value).strip().replace("\ufeff", "").replace("\ufffd", "-")
        category = " ".join(category.split())
        if not category:
            raise ValueError("Dataset có flow chứa nhãn tấn công rỗng.")
        if category.casefold() in {"benign", "normal", "normal traffic"}:
            return "Normal"
        return category

    @staticmethod
    def _flow_features(row: tuple[Any, ...]) -> np.ndarray:
        def scaled_log(value: Any, divisor: float) -> float:
            number = float(value or 0)
            if not math.isfinite(number):
                return 0.0
            return min(1.0, math.log1p(max(0.0, number)) / divisor)

        return np.asarray(
            [
                scaled_log(row[index], scale)
                for index, scale in zip(
                    range(FLOW_FEATURE_START, ATTACK_COLUMN_INDEX),
                    FLOW_FEATURE_SCALES,
                )
            ],
            dtype=np.float32,
        )

    def _resolve_source_files(self) -> None:
        supplied_path = Path(self.path)
        if supplied_path.is_dir():
            parquet_files = sorted(supplied_path.glob("*.parquet"))
            csv_files = sorted(supplied_path.glob("*.csv"))
            files = parquet_files or csv_files
        else:
            files = [Path(file) for file in sorted(glob.glob(self.path))]
            if not files and supplied_path.is_file():
                files = [supplied_path]
        self.source_files = [str(file) for file in files if file.is_file()]
        if not self.source_files:
            raise FileNotFoundError(f"Không tìm thấy file Parquet/CSV: {self.path}")

        suffixes = {Path(file).suffix.casefold() for file in self.source_files}
        if len(suffixes) != 1 or not suffixes.issubset({".csv", ".parquet"}):
            raise ValueError("Dataset phải gồm các file cùng định dạng CSV hoặc Parquet.")
        if suffixes == {".csv"}:
            self.source_relation = (
                "read_csv_auto(?, union_by_name=true, sample_size=2000, "
                "parallel=false, filename=true)"
            )
        else:
            self.source_relation = "read_parquet(?, union_by_name=true, filename=true)"

    @staticmethod
    def _quoted_column(column: str) -> str:
        return '"' + column.replace('"', '""') + '"'

    @staticmethod
    def _quoted_literal(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def _read_schema(self) -> None:
        assert self.connection is not None
        schema_rows = self.connection.execute(
            f"DESCRIBE SELECT * FROM {self.source_relation}", [self.source_files]
        ).fetchall()
        columns = {str(row[0]).strip().casefold(): str(row[0]) for row in schema_rows}

        def resolve(*names: str) -> str | None:
            return next((columns[name.casefold()] for name in names if name.casefold() in columns), None)

        label_column = resolve("Attack", "Label")
        destination_column = resolve("L4_DST_PORT", "Destination Port")
        if label_column is None:
            raise ValueError("Dataset cần cột nhãn 'Attack' hoặc 'Label'.")
        if destination_column is None:
            raise ValueError(
                "Dataset cần 'Destination Port' hoặc 'L4_DST_PORT' để tạo đồ thị."
            )
        self.input_columns = {
            "label": label_column,
            "destination_port": destination_column,
        }
        optional_columns = {
            "source_port": resolve("L4_SRC_PORT", "Source Port"),
            "protocol": resolve("PROTOCOL", "Protocol"),
        }
        optional_columns.update(
            {
                name: resolve(*candidates)
                for name, _, candidates in FLOW_FEATURE_SPECS
            }
        )
        self.input_columns.update(
            {name: column for name, column in optional_columns.items() if column}
        )

    def _source_expression(self, field: str, fallback: int) -> str:
        column = self.input_columns.get(field)
        if column is None:
            return str(fallback)
        return f"COALESCE(TRY_CAST({self._quoted_column(column)} AS BIGINT), {fallback})"

    def _numeric_expression(self, field: str) -> str:
        column = self.input_columns.get(field)
        if column is None:
            return "0.0"
        value = f"TRY_CAST({self._quoted_column(column)} AS DOUBLE)"
        return f"CASE WHEN isfinite({value}) THEN {value} ELSE 0.0 END"

    def _label_expression(self) -> str:
        column = self._quoted_column(self.input_columns["label"])
        value = f"CAST({column} AS VARCHAR)"
        mappings = [
            (raw_label, category)
            for raw_label, category in self.raw_label_categories.items()
            if raw_label != category
        ]
        if not mappings:
            return value

        cases = " ".join(
            f"WHEN {self._quoted_literal(raw_label)} "
            f"THEN {self._quoted_literal(category)}"
            for raw_label, category in mappings
        )
        return f"CASE {value} {cases} ELSE {value} END"

    def _standardized_query(self, train: bool | None = None) -> str:
        source_port = self._source_expression("source_port", UNKNOWN_SOURCE_PORT)
        destination_port = self._source_expression("destination_port", 0)
        protocol = self._source_expression("protocol", UNKNOWN_PROTOCOL)
        label = self._label_expression()
        features = ", ".join(
            (
                f"{source_port} AS source_port",
                f"{destination_port} AS destination_port",
                f"{protocol} AS protocol",
                *(
                    f"{self._numeric_expression(name)} AS {name}"
                    for name in FLOW_FEATURE_COLUMNS
                ),
                f"{label} AS attack",
                "row_number() OVER () - 1 AS stream_row",
            )
        )
        query = (
            f"WITH scanned AS (SELECT {features} FROM {self.source_relation}), "
            "ranked AS ("
            "SELECT *, row_number() OVER (PARTITION BY attack ORDER BY stream_row) "
            "AS class_row, count(*) OVER (PARTITION BY attack) AS class_total "
            "FROM scanned) "
            "SELECT "
            + ", ".join(FLOW_COLUMNS)
            + " FROM ranked"
        )
        if train is not None:
            class_split = (
                "class_row <= CASE WHEN class_total <= 1 THEN class_total "
                "ELSE LEAST(class_total - 1, GREATEST(1, "
                "CAST(FLOOR(class_total * ?) AS BIGINT))) END"
            )
            query += f" WHERE {class_split}" if train else f" WHERE NOT ({class_split})"
            query += " ORDER BY stream_row"
        return query

    def _read_class_counts(self) -> dict[str, int]:
        assert self.connection is not None
        label = self._quoted_column(self.input_columns["label"])
        raw_counts = self.connection.execute(
            f"SELECT CAST({label} AS VARCHAR), count(*) "
            f"FROM {self.source_relation} GROUP BY 1",
            [self.source_files],
        ).fetchall()
        counts: dict[str, int] = {}
        for raw_label, count in raw_counts:
            category = self._attack_category(raw_label)
            raw_value = str(raw_label)
            self.raw_label_categories[raw_value] = category
            counts[category] = counts.get(category, 0) + int(count)
        if len(counts) < 2:
            raise ValueError("Dataset cần ít nhất hai lớp nhãn để huấn luyện.")
        return counts

    def _initialize_nodes(self) -> None:
        assert self.connection is not None
        source_port = self._source_expression("source_port", UNKNOWN_SOURCE_PORT)
        destination_port = self._source_expression("destination_port", 0)
        protocol = self._source_expression("protocol", UNKNOWN_PROTOCOL)
        rows = self.connection.execute(
            "SELECT port, protocol FROM ("
            f"SELECT {source_port} AS port, {protocol} AS protocol "
            f"FROM {self.source_relation} UNION "
            f"SELECT {destination_port} AS port, {protocol} AS protocol "
            f"FROM {self.source_relation}) ORDER BY port, protocol",
            [self.source_files, self.source_files],
        ).fetchall()
        self.node_keys = [(int(port), int(protocol)) for port, protocol in rows]
        self.node_ids = {key: index for index, key in enumerate(self.node_keys)}
        self.node_attributes = np.asarray(
            [
                (port / 65536.0, protocol / 255.0)
                for port, protocol in self.node_keys
            ],
            dtype=np.float32,
        )
        self.neighbor_sums = np.zeros_like(self.node_attributes)
        self.neighbor_counts = np.zeros(len(rows), dtype=np.int64)
        self.node_degrees = np.zeros(len(rows), dtype=np.int64)

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
        class_counts = self._read_class_counts()
        self.categories = sorted(
            class_counts, key=lambda category: (category != "Normal", category.casefold())
        )
        self.category_indices = {
            category: index for index, category in enumerate(self.categories)
        }
        self.training_class_counts = {
            category: (
                count
                if count <= 1
                else min(count - 1, max(1, int(count * self.training_fraction)))
            )
            for category, count in class_counts.items()
        }
        self.training_rows = sum(self.training_class_counts.values())
        self.replay_rows = sum(class_counts.values()) - self.training_rows
        if self.training_rows < 2 or self.replay_rows < 1:
            raise ValueError("Dataset cần ít nhất 3 flow để chia train và replay.")

        self._initialize_nodes()
        training_query = self._standardized_query(train=True)
        self.replay_query = self._standardized_query(train=False)
        self.training_reader = self.connection.execute(
            training_query, [self.source_files, self.training_fraction]
        )
        node_features = np.empty((self.training_rows, 2, 4), dtype=np.float32)
        flow_features = np.empty(
            (self.training_rows, len(FLOW_FEATURE_SCALES)), dtype=np.float32
        )
        labels = np.empty(self.training_rows, dtype=np.int64)
        processed = 0

        while processed < self.training_rows:
            rows = self.training_reader.fetchmany(
                min(8192, self.training_rows - processed)
            )
            if not rows:
                raise ValueError("CSV/Parquet kết thúc trước số dòng train đã khai báo.")
            for row in rows:
                source_id = self.node_ids[(int(row[0]), int(row[2]))]
                target_id = self.node_ids[(int(row[1]), int(row[2]))]
                node_features[processed, 0, :2] = self.node_attributes[source_id]
                node_features[processed, 0, 2:] = self._node_context(source_id)
                node_features[processed, 1, :2] = self.node_attributes[target_id]
                node_features[processed, 1, 2:] = self._node_context(target_id)
                flow_features[processed] = self._flow_features(row)
                attack = self._attack_category(row[ATTACK_COLUMN_INDEX])
                labels[processed] = self.category_indices[attack]
                self._observe_edge(source_id, target_id)
                processed += 1
            if progress is not None:
                progress(processed, self.training_rows)

        return node_features, flow_features, labels

    def fit(
        self,
        epochs: int = TRAINING_EPOCHS,
        batch_size: int = TRAINING_BATCH_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        self._resolve_source_files()
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        torch.manual_seed(42)
        self.connection = duckdb.connect()
        self._read_schema()
        node_features, flow_features, labels = self._read_training_rows(progress)
        class_counts = np.bincount(labels, minlength=len(self.categories))
        active_classes = class_counts > 0
        self.available_categories = [
            category
            for index, category in enumerate(self.categories)
            if active_classes[index]
        ]
        self.active_class_mask = torch.from_numpy(active_classes)

        self.model = GraphSAGE(len(self.categories))
        source_tensor = torch.from_numpy(node_features[:, 0, :])
        target_tensor = torch.from_numpy(node_features[:, 1, :])
        flow_tensor = torch.from_numpy(flow_features)
        label_tensor = torch.from_numpy(labels)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        class_weights = torch.ones(len(self.categories), dtype=torch.float32)
        active_counts = torch.from_numpy(class_counts[active_classes]).float()
        active_weights = torch.sqrt(active_counts.max() / active_counts).clamp(max=30.0)
        active_weights /= active_weights.mean()
        class_weights[self.active_class_mask] = active_weights
        loss_function = nn.CrossEntropyLoss(weight=class_weights)

        self.model.train()
        for _ in range(epochs):
            order = torch.randperm(self.training_rows)
            for start in range(0, self.training_rows, batch_size):
                indices = order[start : start + batch_size]
                optimizer.zero_grad()
                logits = self.model(
                    source_tensor[indices], target_tensor[indices], flow_tensor[indices]
                )
                logits = logits.masked_fill(
                    ~self.active_class_mask, torch.finfo(logits.dtype).min
                )
                loss = loss_function(logits, label_tensor[indices])
                loss.backward()
                optimizer.step()

        del node_features, flow_features, labels
        del source_tensor, target_tensor, flow_tensor, label_tensor
        self.model.eval()
        return {
            "training_rows": self.training_rows,
            "training_class_counts": self.training_class_counts,
            "categories": self.categories,
            "available_categories": self.available_categories,
            "nodes": len(self.node_ids),
            "edges": len(self.edge_pairs),
            "replay_rows": self.replay_rows,
        }

    def iter_predictions(self, batch_size: int = 128) -> Iterator[list[dict[str, Any]]]:
        if (
            self.connection is None
            or self.model is None
            or self.active_class_mask is None
            or self.replay_query is None
        ):
            raise RuntimeError("GNN chưa được huấn luyện.")

        self.replay_reader = self.connection.execute(
            self.replay_query, [self.source_files, self.training_fraction]
        )
        remaining = self.replay_rows
        sequence = 0
        while remaining:
            rows = self.replay_reader.fetchmany(min(batch_size, remaining))
            if not rows:
                raise ValueError("CSV/Parquet kết thúc trước khi replay hoàn tất.")
            remaining -= len(rows)
            source_features = np.empty((len(rows), 4), dtype=np.float32)
            target_features = np.empty((len(rows), 4), dtype=np.float32)
            flow_features = np.empty(
                (len(rows), len(FLOW_FEATURE_SCALES)), dtype=np.float32
            )
            source_ids: list[int] = []
            target_ids: list[int] = []
            new_edges: list[bool] = []

            for index, row in enumerate(rows):
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
                logits = self.model(
                    torch.from_numpy(source_features),
                    torch.from_numpy(target_features),
                    torch.from_numpy(flow_features),
                )
                logits = logits.masked_fill(
                    ~self.active_class_mask, torch.finfo(logits.dtype).min
                )
                probabilities = torch.softmax(logits, dim=1).numpy()

            batch: list[dict[str, Any]] = []
            for index, row in enumerate(rows):
                sequence += 1
                class_probabilities = probabilities[index]
                predicted_index = int(np.argmax(class_probabilities))
                category = self.categories[predicted_index]
                score = float(class_probabilities[predicted_index])
                batch.append(
                    {
                        "id": sequence,
                        "row": sequence,
                        "timestamp": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                        "source": self.node_label(source_ids[index]),
                        "destination": self.node_label(target_ids[index]),
                        "protocol": self.protocol_label(int(row[2])),
                        "destination_port": int(row[1]),
                        "source_port": int(row[0]),
                        "packets": int(row[FEATURE_INDEX["in_packets"]])
                        + int(row[FEATURE_INDEX["out_packets"]]),
                        "bytes": int(row[FEATURE_INDEX["in_bytes"]])
                        + int(row[FEATURE_INDEX["out_bytes"]]),
                        "category": category,
                        "score": round(score, 4),
                        "confidence_scores": {
                            name: round(float(class_probabilities[class_index]), 4)
                            for class_index, name in enumerate(self.categories)
                        },
                        "status": "ALERT" if category != "Normal" else "NORMAL",
                        "attack": category,
                        "true_attack": self._attack_category(
                            row[ATTACK_COLUMN_INDEX]
                        ),
                        "raw_attack": str(row[ATTACK_COLUMN_INDEX]),
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
        if port == UNKNOWN_SOURCE_PORT and protocol == UNKNOWN_PROTOCOL:
            return "Unknown Source"
        protocol_name = (
            self.protocol_label(protocol)
            if protocol != UNKNOWN_PROTOCOL
            else "Unknown"
        )
        return f"{port}/{protocol_name}"

    @staticmethod
    def protocol_label(protocol: int) -> str:
        return PROTOCOL_NAMES.get(protocol, f"IP-{protocol}" if protocol >= 0 else "Unknown")
