"""Flask dashboard for streaming multi-class intrusion detection."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections import Counter, deque
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from gnn_detector import StreamingGNNDetector

app = Flask(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

DATASET_PATH = os.environ.get("DATASET_PATH", "/data")
TRAINING_FRACTION = float(os.environ.get("TRAINING_FRACTION", "0.8"))
REPLAY_BATCH_SIZE = max(1, int(os.environ.get("REPLAY_BATCH_SIZE", "128")))
REPLAY_INTERVAL_SECONDS = max(
    0.01, float(os.environ.get("REPLAY_INTERVAL_SECONDS", "0.25"))
)
ROC_BINS = 100

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
    "categories": [],
    "training_class_counts": {},
    "available_categories": [],
    "replay_rows": 0,
    "replay_total": 0,
    "replay_interval_seconds": REPLAY_INTERVAL_SECONDS,
    "total_events": 0,
    "alert_count": 0,
    "nodes": set(),
    "edge_count": 0,
    "graph_seed_nodes": [],
    "counts": Counter(),
    "events": deque(maxlen=100),
    "node_risk": {},
    "confusion_matrix": {},
    "roc_histogram": {},
    "performance_samples": deque(maxlen=30),
}


def _evaluation_snapshot() -> dict[str, Any]:
    categories = state["categories"]
    matrix = state["confusion_matrix"]
    total = sum(sum(matrix[actual].values()) for actual in categories)
    correct = sum(matrix[category][category] for category in categories)
    by_class: dict[str, dict[str, float | int]] = {}
    for category in categories:
        true_positive = matrix[category][category]
        support = sum(matrix[category].values())
        predicted = sum(matrix[actual][category] for actual in categories)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        by_class[category] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    roc_points: list[list[tuple[float, float]]] = []
    class_auc: dict[str, float] = {}
    for category in categories:
        histogram = state["roc_histogram"].get(category)
        if histogram is None:
            continue
        positives = histogram["positive"]
        negatives = histogram["negative"]
        positive_total = sum(positives)
        negative_total = sum(negatives)
        if not positive_total or not negative_total:
            continue
        true_positive = false_positive = 0
        points = [(0.0, 0.0)]
        for index in range(ROC_BINS - 1, -1, -1):
            true_positive += positives[index]
            false_positive += negatives[index]
            points.append(
                (
                    false_positive / negative_total,
                    true_positive / positive_total,
                )
            )
        area = sum(
            (x2 - x1) * (y1 + y2) / 2
            for (x1, y1), (x2, y2) in zip(points, points[1:])
        )
        class_auc[category] = round(area, 4)
        roc_points.append(points)

    def interpolate(points: list[tuple[float, float]], fpr: float) -> float:
        previous_fpr, previous_tpr = points[0]
        for current_fpr, current_tpr in points[1:]:
            if fpr <= current_fpr:
                if current_fpr == previous_fpr:
                    return current_tpr
                ratio = (fpr - previous_fpr) / (current_fpr - previous_fpr)
                return previous_tpr + ratio * (current_tpr - previous_tpr)
            previous_fpr, previous_tpr = current_fpr, current_tpr
        return 1.0

    macro_curve = []
    if roc_points:
        for index in range(21):
            false_positive_rate = index / 20
            true_positive_rate = sum(
                interpolate(points, false_positive_rate) for points in roc_points
            ) / len(roc_points)
            macro_curve.append(
                {
                    "fpr": round(false_positive_rate, 4),
                    "tpr": round(true_positive_rate, 4),
                }
            )
    performance_samples = list(state["performance_samples"])
    processed = sum(count for count, _ in performance_samples)
    elapsed = sum(duration for _, duration in performance_samples)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "by_class": by_class,
        "roc_auc_ovr_macro": (
            round(sum(class_auc.values()) / len(class_auc), 4) if class_auc else None
        ),
        "roc_auc_by_class": class_auc,
        "roc_curve": macro_curve,
        "confusion_matrix": {
            actual: {predicted: matrix[actual][predicted] for predicted in categories}
            for actual in categories
        },
        "performance": {
            "throughput_fps": round(processed / elapsed, 2) if elapsed else 0.0,
            "latency_ms_per_flow": (
                round(elapsed * 1000 / processed, 3) if processed else 0.0
            ),
            "window_batches": len(performance_samples),
        },
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
            "category": event["category"],
            "alert": event["status"] == "ALERT",
        }
        for event in events[:45]
    ]
    return {
        "version": state["version"],
        "paused": state["paused"],
        "categories": list(state["categories"]),
        "model": {
            "name": "GraphSAGE with replay context",
            "status": state["model_status"],
            "progress": round(state["model_progress"], 3),
            "error": state["model_error"],
            "training_rows": state["training_rows"],
            "training_class_counts": state["training_class_counts"],
            "available_categories": state["available_categories"],
            "replay_rows": state["replay_rows"],
            "replay_total": state["replay_total"],
            "replay_interval_seconds": state["replay_interval_seconds"],
            "evaluation": _evaluation_snapshot(),
        },
        "metrics": {
            "nodes": len(state["nodes"]),
            "edges": state["edge_count"],
            "alerts": state["alert_count"],
            "flows": state["total_events"],
        },
        "counts": {
            category: state["counts"][category] for category in state["categories"]
        },
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


def _publish_batch(batch: list[dict[str, Any]], duration_seconds: float) -> None:
    with condition:
        matrix = state["confusion_matrix"]
        for event in batch:
            category = event["category"]
            actual_category = event["true_attack"]
            is_alert = category != "Normal"
            state["total_events"] += 1
            state["replay_rows"] += 1
            state["counts"][category] += 1
            state["alert_count"] += int(is_alert)
            matrix[actual_category][category] += 1
            for label, score in event["confidence_scores"].items():
                histogram = state["roc_histogram"][label]
                score_bin = min(ROC_BINS - 1, max(0, int(float(score) * ROC_BINS)))
                histogram[
                    "positive" if actual_category == label else "negative"
                ][score_bin] += 1
            state["nodes"].update((event["source"], event["destination"]))
            if event["new_edge"]:
                state["edge_count"] += 1
            if is_alert:
                risk = event["score"]
                state["node_risk"][event["source"]] = max(
                    state["node_risk"].get(event["source"], 0.0), risk
                )
                state["node_risk"][event["destination"]] = max(
                    state["node_risk"].get(event["destination"], 0.0), risk
                )
            event["received_at"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            event.pop("new_edge", None)
            state["events"].appendleft(event)
        state["version"] += 1
        state["performance_samples"].append((len(batch), duration_seconds))
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
            state["categories"] = summary["categories"]
            state["training_class_counts"] = summary["training_class_counts"]
            state["available_categories"] = summary["available_categories"]
            state["counts"] = Counter({category: 0 for category in summary["categories"]})
            state["confusion_matrix"] = {
                actual: Counter(
                    {predicted: 0 for predicted in summary["categories"]}
                )
                for actual in summary["categories"]
            }
            state["roc_histogram"] = {
                category: {
                    "positive": [0] * ROC_BINS,
                    "negative": [0] * ROC_BINS,
                }
                for category in summary["categories"]
            }
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
                condition.wait(timeout=state["replay_interval_seconds"])
                if state["paused"]:
                    continue
            batch_started = perf_counter()
            batch = next(predictions, None)
            if batch is None:
                with condition:
                    state["model_status"] = "complete"
                    state["paused"] = True
                    state["version"] += 1
                    condition.notify_all()
                return
            _publish_batch(batch, perf_counter() - batch_started)
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
    if not isinstance(body, dict):
        return jsonify({"error": "Request body phải là JSON object."}), 400
    action = body.get("action")
    if action not in {"pause", "resume", "speed"}:
        return jsonify({"error": "action phải là 'pause', 'resume' hoặc 'speed'."}), 400

    if action == "speed":
        interval = body.get("interval_seconds")
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or not math.isfinite(interval)
            or not 0.01 <= interval <= 10
        ):
            return jsonify(
                {"error": "interval_seconds phải nằm trong khoảng 0.01 đến 10."}
            ), 400

    _start_worker()
    with condition:
        if state["model_status"] in {"complete", "error"} and action != "pause":
            return jsonify({"error": "Dataset replay đã kết thúc hoặc gặp lỗi."}), 409
        if action == "speed":
            state["replay_interval_seconds"] = float(interval)
        else:
            state["paused"] = action == "pause"
        state["version"] += 1
        condition.notify_all()
        return jsonify(
            {
                "paused": state["paused"],
                "interval_seconds": state["replay_interval_seconds"],
                "version": state["version"],
            }
        )


@app.get("/api/neighbors")
def inspect_neighbors() -> Any:
    node = request.args.get("node", "").strip()
    try:
        hops = int(request.args.get("hops", "2"))
    except ValueError:
        return jsonify({"error": "hops phải là số nguyên từ 1 đến 3."}), 400
    if not node:
        return jsonify({"error": "Cần truyền node."}), 400
    if not 1 <= hops <= 3:
        return jsonify({"error": "hops phải nằm trong khoảng 1 đến 3."}), 400

    with condition:
        adjacency: dict[str, set[str]] = {}
        for event in state["events"]:
            source, destination = event["source"], event["destination"]
            adjacency.setdefault(source, set()).add(destination)
            adjacency.setdefault(destination, set()).add(source)
    if node not in adjacency:
        return jsonify({"error": "Không tìm thấy node trong các flow replay gần đây."}), 404

    seen = {node}
    frontier = {node}
    layers = []
    for distance in range(1, hops + 1):
        next_frontier = {
            neighbor
            for current in frontier
            for neighbor in adjacency.get(current, ())
            if neighbor not in seen
        }
        seen.update(next_frontier)
        layers.append(
            {"hop": distance, "nodes": sorted(next_frontier)}
        )
        frontier = next_frontier
    return jsonify(
        {"node": node, "hops": hops, "layers": layers, "scope": "last 100 replay flows"}
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        threaded=True,
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
