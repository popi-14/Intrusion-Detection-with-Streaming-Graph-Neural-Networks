"""Build bounded, time-ordered graph snapshots from flow CSV/Parquet datasets."""

from __future__ import annotations

import glob
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import duckdb
import numpy as np


@dataclass(frozen=True)
class TemporalGraphSnapshot:
    """One time window of directed network-flow edges."""

    window_start: datetime
    nodes: tuple[str, ...]
    edge_index: np.ndarray
    edge_features: np.ndarray
    labels: tuple[str, ...]
    feature_names: tuple[str, ...]


class DuckDBTemporalGraphLoader:
    """Stream snapshots from datasets containing endpoints and flow timestamps.

    The loader accepts CIC-/NF-UNSW-style column names, keeps at most one
    configured time window in Python memory, and uses no PyG binary extensions.
    """

    TIMESTAMP_COLUMNS = (
        "timestamp",
        "flow_start_milliseconds",
        "flow_start",
        "start_time",
        "datetime",
        "ts",
    )
    SOURCE_COLUMNS = (
        "ipv4_src_addr",
        "src_ip",
        "source_ip",
        "source ip",
        "src_addr",
        "src",
    )
    DESTINATION_COLUMNS = (
        "ipv4_dst_addr",
        "dst_ip",
        "destination_ip",
        "destination ip",
        "dst_addr",
        "dst",
    )
    SOURCE_PORT_COLUMNS = ("l4_src_port", "src_port", "source_port")
    DESTINATION_PORT_COLUMNS = ("l4_dst_port", "dst_port", "destination_port")
    LABEL_COLUMNS = ("attack", "label", "class")
    EXCLUDED_FEATURES = {
        *TIMESTAMP_COLUMNS,
        *SOURCE_COLUMNS,
        *DESTINATION_COLUMNS,
        *SOURCE_PORT_COLUMNS,
        *DESTINATION_PORT_COLUMNS,
        *LABEL_COLUMNS,
        "protocol",
        "proto",
        "ip_protocol",
    }

    def __init__(
        self,
        path: str | Path,
        window_seconds: int = 10,
        feature_limit: int = 24,
        fetch_size: int = 8192,
    ) -> None:
        if window_seconds < 1:
            raise ValueError("window_seconds phải lớn hơn hoặc bằng 1.")
        if feature_limit < 1:
            raise ValueError("feature_limit phải lớn hơn hoặc bằng 1.")
        if fetch_size < 1:
            raise ValueError("fetch_size phải lớn hơn hoặc bằng 1.")
        self.path = str(path)
        self.window_seconds = window_seconds
        self.feature_limit = feature_limit
        self.fetch_size = fetch_size

    @staticmethod
    def _quote_identifier(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    def _files(self) -> list[str]:
        path = Path(self.path)
        if path.is_dir():
            parquet = sorted(path.glob("*.parquet"))
            files = parquet or sorted(path.glob("*.csv"))
        else:
            files = [Path(item) for item in sorted(glob.glob(self.path))]
            if not files and path.is_file():
                files = [path]
        resolved = [str(item) for item in files if item.is_file()]
        if not resolved:
            raise FileNotFoundError(f"Không tìm thấy file CSV/Parquet: {self.path}")
        suffixes = {Path(item).suffix.casefold() for item in resolved}
        if len(suffixes) != 1 or not suffixes.issubset({".csv", ".parquet"}):
            raise ValueError("Dataset phải gồm các file cùng định dạng CSV hoặc Parquet.")
        return resolved

    def _source_relation(self, files: list[str]) -> str:
        if Path(files[0]).suffix.casefold() == ".parquet":
            return "read_parquet(?, union_by_name=true)"
        return (
            "read_csv_auto(?, union_by_name=true, sample_size=2000, "
            "parallel=false)"
        )

    @staticmethod
    def _resolve(columns: dict[str, tuple[str, str]], candidates: tuple[str, ...]) -> str | None:
        return next((columns[name][0] for name in candidates if name in columns), None)

    def _snapshot_query(
        self,
        relation: str,
        columns: dict[str, tuple[str, str]],
    ) -> tuple[str, tuple[str, ...]]:
        timestamp = self._resolve(columns, self.TIMESTAMP_COLUMNS)
        source = self._resolve(columns, self.SOURCE_COLUMNS)
        destination = self._resolve(columns, self.DESTINATION_COLUMNS)
        label = self._resolve(columns, self.LABEL_COLUMNS)
        missing = [
            name
            for name, value in (
                ("timestamp", timestamp),
                ("source IP", source),
                ("destination IP", destination),
                ("attack label", label),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "Temporal graph cần cột " + ", ".join(missing) + ". "
                "Dataset CIC-IDS2017 hiện tại không có IP nguồn/đích hoặc timestamp."
            )

        source_port = self._resolve(columns, self.SOURCE_PORT_COLUMNS)
        destination_port = self._resolve(columns, self.DESTINATION_PORT_COLUMNS)
        excluded = {name.casefold() for name in self.EXCLUDED_FEATURES}
        numeric_types = (
            "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
            "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT",
            "FLOAT", "DOUBLE", "DECIMAL",
        )
        feature_columns = [
            original
            for original, type_name in columns.values()
            if original.casefold() not in excluded
            and any(type_name.upper().startswith(numeric) for numeric in numeric_types)
        ][: self.feature_limit]
        if not feature_columns:
            raise ValueError("Dataset cần ít nhất một cột số làm edge feature.")

        def endpoint(ip_column: str, port_column: str | None) -> str:
            ip = f"TRIM(CAST({self._quote_identifier(ip_column)} AS VARCHAR))"
            if port_column is None:
                return ip
            port = (
                f"TRY_CAST({self._quote_identifier(port_column)} AS BIGINT)"
            )
            return f"({ip} || ':' || COALESCE(CAST({port} AS VARCHAR), '?'))"

        timestamp_value = self._quote_identifier(timestamp)
        numeric_timestamp = f"TRY_CAST({timestamp_value} AS DOUBLE)"
        text_timestamp = f"CAST({timestamp_value} AS VARCHAR)"
        parsed_text_timestamp = (
            f"COALESCE(TRY_CAST({text_timestamp} AS TIMESTAMP), "
            f"TRY_STRPTIME({text_timestamp}, '%m/%d/%Y %H:%M:%S'), "
            f"TRY_STRPTIME({text_timestamp}, '%m/%d/%Y %H:%M'), "
            f"TRY_STRPTIME({text_timestamp}, '%m/%d/%Y %I:%M:%S %p'), "
            f"TRY_STRPTIME({text_timestamp}, '%d/%m/%Y %H:%M:%S'), "
            f"TRY_STRPTIME({text_timestamp}, '%d/%m/%Y %H:%M'), "
            f"TRY_STRPTIME({text_timestamp}, '%Y-%m-%d %H:%M:%S'))"
        )
        parsed_timestamp = (
            "CASE "
            f"WHEN {numeric_timestamp} IS NOT NULL THEN to_timestamp(CASE "
            f"WHEN {numeric_timestamp} >= 1000000000000000 THEN {numeric_timestamp} / 1000000.0 "
            f"WHEN {numeric_timestamp} >= 100000000000 THEN {numeric_timestamp} / 1000.0 "
            f"ELSE {numeric_timestamp} END) "
            f"ELSE {parsed_text_timestamp} END"
        )
        selected_features = [
            "CASE "
            f"WHEN isfinite(TRY_CAST({self._quote_identifier(name)} AS DOUBLE)) "
            f"THEN LN(1.0 + GREATEST(0.0, "
            f"TRY_CAST({self._quote_identifier(name)} AS DOUBLE))) "
            "ELSE 0.0 END "
            f"AS {self._quote_identifier(f'feature_{index}')}"
            for index, name in enumerate(feature_columns)
        ]
        feature_names = tuple(feature_columns)
        feature_sql = ", " + ", ".join(selected_features) if selected_features else ""
        source_sql = endpoint(source, source_port)
        destination_sql = endpoint(destination, destination_port)
        query = (
            "WITH normalized AS (SELECT "
            f"{parsed_timestamp} AS event_time, "
            f"{source_sql} AS source_node, "
            f"{destination_sql} AS destination_node, "
            f"TRIM(CAST({self._quote_identifier(label)} AS VARCHAR)) AS attack"
            f"{feature_sql} FROM {relation}), "
            "windowed AS (SELECT "
            "CAST(FLOOR(epoch(event_time) / ?) AS BIGINT) AS window_id, "
            "epoch(event_time) AS event_epoch, source_node, destination_node, attack"
            + (
                ", " + ", ".join(
                    self._quote_identifier(f"feature_{index}")
                    for index in range(len(feature_columns))
                )
                if feature_columns
                else ""
            )
            + " FROM normalized WHERE event_time IS NOT NULL "
            "AND source_node <> '' AND destination_node <> '' AND attack <> '') "
            "SELECT * FROM windowed ORDER BY window_id, event_epoch"
        )
        return query, feature_names

    @staticmethod
    def _normalize_label(label: str) -> str:
        normalized = " ".join(label.replace("\ufffd", "-").split())
        if normalized.casefold() in {"benign", "normal", "normal traffic"}:
            return "Normal"
        return normalized

    def iter_snapshots(self) -> Iterator[TemporalGraphSnapshot]:
        """Yield nonempty snapshots in timestamp order."""
        files = self._files()
        relation = self._source_relation(files)
        connection = duckdb.connect()
        try:
            schema = connection.execute(
                f"DESCRIBE SELECT * FROM {relation}", [files]
            ).fetchall()
            columns = {
                str(row[0]).strip().casefold(): (str(row[0]), str(row[1]))
                for row in schema
            }
            query, feature_names = self._snapshot_query(relation, columns)
            cursor = connection.execute(query, [files, self.window_seconds])
            row_width = 5 + len(feature_names)
            current_window: int | None = None
            nodes: dict[str, int] = {}
            edges: list[tuple[int, int]] = []
            features: list[list[float]] = []
            labels: list[str] = []

            def snapshot(window_id: int) -> TemporalGraphSnapshot:
                edge_index = np.asarray(edges, dtype=np.int64).T
                if not edges:
                    edge_index = np.empty((2, 0), dtype=np.int64)
                edge_features = np.asarray(features, dtype=np.float32).reshape(
                    len(edges), len(feature_names)
                )
                return TemporalGraphSnapshot(
                    window_start=datetime.fromtimestamp(
                        window_id * self.window_seconds, tz=timezone.utc
                    ),
                    nodes=tuple(nodes),
                    edge_index=edge_index,
                    edge_features=edge_features,
                    labels=tuple(labels),
                    feature_names=feature_names,
                )

            while rows := cursor.fetchmany(self.fetch_size):
                for row in rows:
                    window_id = int(row[0])
                    if current_window is not None and window_id != current_window:
                        yield snapshot(current_window)
                        nodes, edges, features, labels = {}, [], [], []
                    current_window = window_id
                    source_name, destination_name = str(row[2]), str(row[3])
                    source_id = nodes.setdefault(source_name, len(nodes))
                    destination_id = nodes.setdefault(destination_name, len(nodes))
                    edges.append((source_id, destination_id))
                    labels.append(self._normalize_label(str(row[4])))
                    feature_start = 5
                    features.append(
                        [float(value) if math.isfinite(float(value)) else 0.0
                         for value in row[feature_start:row_width]]
                    )

            if current_window is not None:
                yield snapshot(current_window)
        finally:
            connection.close()
