"""Flask dashboard for replaying and evaluating the NF-UNSW-NB15-V2 dataset."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from gnn_detector import StreamingGNNDetector

app = Flask(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

CATEGORIES = ("Anomaly", "Normal Traffic")
DATASET_PATH = os.environ.get("DATASET_PATH", "/data/NF-UNSW-NB15-V2.parquet")
TRAINING_FRACTION = float(os.environ.get("TRAINING_FRACTION", "0.8"))
REPLAY_BATCH_SIZE = max(1, int(os.environ.get("REPLAY_BATCH_SIZE", "128")))
REPLAY_INTERVAL_SECONDS = max(
    0.01, float(os.environ.get("REPLAY_INTERVAL_SECONDS", "0.25"))
)

condition = threading.Condition(threading.RLock())
worker_guard = threading.Lock()
worker_started = False
detector: StreamingGNNDetector | None = None
state: dict[str, Any] = {
    "version": 0,
    "paused": False,
    "model_status": "waiting",
    "model_progress": 0.0,
    "model_error": None,
    "training_rows": 0,
    "training_anomalies": 0,
    "calibration_rows": 0,
    "threshold": 0.5,
    "replay_rows": 0,
    "replay_total": 0,
    "total_events": 0,
    "alert_count": 0,
    "nodes": set(),
    "edge_count": 0,
    "graph_seed_nodes": [],
    "counts": Counter({category: 0 for category in CATEGORIES}),
    "events": deque(maxlen=100),
    "node_risk": {},
    "true_positives": 0,
    "true_negatives": 0,
    "false_positives": 0,
    "false_negatives": 0,
}


def _snapshot_locked() -> dict[str, Any]:
    events = list(state["events"])
    graph_node_ids = set(state["graph_seed_nodes"])
    for event in events[:45]:
        graph_node_ids.update((event["source"], event["destination"]))
    graph_nodes = [
        {"id": node, "risk": round(state["node_risk"].get(node, 0.0), 3)}
        for node in sorted(graph_node_ids)
    ]
    graph_edges = [
        {
            "source": event["source"],
            "target": event["destination"],
            "score": event["score"],
            "alert": event["status"] == "ALERT",
        }
        for event in events[:45]
    ]
    return {
        "version": state["version"],
        "paused": state["paused"],
        "model": {
            "name": "Temporal GraphSAGE",
            "status": state["model_status"],
            "progress": round(state["model_progress"], 3),
            "error": state["model_error"],
            "training_rows": state["training_rows"],
            "training_anomalies": state["training_anomalies"],
            "calibration_rows": state["calibration_rows"],
            "threshold": state["threshold"],
            "replay_rows": state["replay_rows"],
            "replay_total": state["replay_total"],
            "evaluation": {
                "true_positives": state["true_positives"],
                "true_negatives": state["true_negatives"],
                "false_positives": state["false_positives"],
                "false_negatives": state["false_negatives"],
            },
        },
        "metrics": {
            "nodes": len(state["nodes"]),
            "edges": state["edge_count"],
            "alerts": state["alert_count"],
            "flows": state["total_events"],
        },
        "counts": {category: state["counts"][category] for category in CATEGORIES},
        "events": events,
        "graph": {"nodes": graph_nodes, "edges": graph_edges},
        "series": [
            {"label": event["id"], "score": event["score"]}
            for event in list(reversed(events))[-24:]
        ],
    }


def _set_progress(processed: int, total: int) -> None:
    with condition:
        state["training_rows"] = total
        state["model_progress"] = processed / total
        state["version"] += 1
        condition.notify_all()


def _publish_batch(batch: list[dict[str, Any]]) -> None:
    with condition:
        for event in batch:
            is_alert = event["status"] == "ALERT"
            is_actual_anomaly = event["label"] == 1
            state["total_events"] += 1
            state["replay_rows"] += 1
            state["counts"][event["category"]] += 1
            if is_alert:
                state["alert_count"] += 1
            if is_alert and is_actual_anomaly:
                state["true_positives"] += 1
            elif is_alert:
                state["false_positives"] += 1
            elif is_actual_anomaly:
                state["false_negatives"] += 1
            else:
                state["true_negatives"] += 1
            state["nodes"].update((event["source"], event["destination"]))
            if event["new_edge"]:
                state["edge_count"] += 1
            state["node_risk"][event["source"]] = max(
                state["node_risk"].get(event["source"], 0.0), event["score"]
            )
            state["node_risk"][event["destination"]] = max(
                state["node_risk"].get(event["destination"], 0.0), event["score"]
            )
            event["received_at"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            event.pop("new_edge", None)
            state["events"].appendleft(event)
        state["version"] += 1
        condition.notify_all()


def _stream_worker() -> None:
    global detector
    try:
        with condition:
            state["model_status"] = "training"
            state["version"] += 1
            condition.notify_all()

        model = StreamingGNNDetector(DATASET_PATH, TRAINING_FRACTION)
        summary = model.fit(progress=_set_progress)
        detector = model
        with condition:
            state["training_rows"] = summary["training_rows"]
            state["training_anomalies"] = summary["training_anomalies"]
            state["calibration_rows"] = summary["calibration_rows"]
            state["threshold"] = summary["threshold"]
            state["replay_total"] = summary["replay_rows"]
            state["nodes"].update(model.active_nodes())
            state["edge_count"] = summary["edges"]
            state["graph_seed_nodes"] = model.top_node_labels(40)
            state["model_progress"] = 1.0
            state["model_status"] = "replaying"
            state["version"] += 1
            condition.notify_all()

        predictions = model.iter_predictions(REPLAY_BATCH_SIZE)
        while True:
            with condition:
                condition.wait_for(
                    lambda: not state["paused"]
                    or state["model_status"] in {"complete", "error"}
                )
                condition.wait(timeout=REPLAY_INTERVAL_SECONDS)
                if state["paused"]:
                    continue
            batch = next(predictions, None)
            if batch is None:
                with condition:
                    state["model_status"] = "complete"
                    state["paused"] = True
                    state["version"] += 1
                    condition.notify_all()
                return
            _publish_batch(batch)
    except Exception as error:
        logger.exception("Dataset GNN worker failed")
        with condition:
            state["model_status"] = "error"
            state["model_error"] = str(error)
            state["version"] += 1
            condition.notify_all()


def _start_worker() -> None:
    global worker_started
    with worker_guard:
        if worker_started:
            return
        worker_started = True
        threading.Thread(
            target=_stream_worker, name="dataset-gnn-worker", daemon=True
        ).start()


@app.get("/")
def dashboard() -> str:
    return render_template("index.html")


@app.get("/healthz")
def health_check() -> Any:
    with condition:
        return jsonify({"status": "ok", "model_status": state["model_status"]})


@app.get("/api/state")
def get_state() -> Any:
    _start_worker()
    with condition:
        return jsonify(_snapshot_locked())


@app.get("/api/stream")
def stream_state() -> Response:
    _start_worker()

    @stream_with_context
    def generate():
        last_version = -1
        while True:
            with condition:
                changed = condition.wait_for(
                    lambda: state["version"] != last_version, timeout=15
                )
                snapshot = _snapshot_locked()
                last_version = snapshot["version"]
            if changed:
                yield f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
            else:
                yield ": keep-alive\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/control")
def control_stream() -> Any:
    body = request.get_json(silent=True) or {}
    action = body.get("action")
    if action not in {"pause", "resume"}:
        return jsonify({"error": "action phải là 'pause' hoặc 'resume'."}), 400

    _start_worker()
    with condition:
        if state["model_status"] in {"complete", "error"} and action == "resume":
            return jsonify({"error": "Dataset replay đã kết thúc hoặc gặp lỗi."}), 409
        state["paused"] = action == "pause"
        state["version"] += 1
        condition.notify_all()
        return jsonify({"paused": state["paused"], "version": state["version"]})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        threaded=True,
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
