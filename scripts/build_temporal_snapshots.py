"""Export temporal graph snapshots as compressed NPZ files and a JSON manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from temporal_graph import DuckDBTemporalGraphLoader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV/Parquet file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--window-seconds", type=int, default=10)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    loader = DuckDBTemporalGraphLoader(
        args.input, window_seconds=args.window_seconds
    )
    for index, snapshot in enumerate(loader.iter_snapshots()):
        filename = f"snapshot-{index:08d}.npz"
        np.savez_compressed(
            output / filename,
            edge_index=snapshot.edge_index,
            edge_features=snapshot.edge_features,
            nodes=np.asarray(snapshot.nodes, dtype=str),
            labels=np.asarray(snapshot.labels, dtype=str),
        )
        manifest.append(
            {
                "file": filename,
                "window_start": snapshot.window_start.isoformat(),
                "nodes": len(snapshot.nodes),
                "edges": len(snapshot.labels),
                "feature_names": list(snapshot.feature_names),
                "labels": sorted(set(snapshot.labels)),
            }
        )
        print(f"{filename}: {len(snapshot.nodes)} nodes, {len(snapshot.labels)} flows")

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(manifest)} windows to {output}")


if __name__ == "__main__":
    main()
