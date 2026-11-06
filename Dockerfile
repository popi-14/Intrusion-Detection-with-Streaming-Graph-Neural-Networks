FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt requirements-docker.txt ./
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements-docker.txt \
    && python -c "import duckdb, flask, numpy, torch"

COPY app.py ./app.py
COPY gnn_detector.py ./gnn_detector.py
COPY temporal_graph.py ./temporal_graph.py
COPY temporal_model.py ./temporal_model.py
COPY scripts ./scripts
COPY templates ./templates

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]
