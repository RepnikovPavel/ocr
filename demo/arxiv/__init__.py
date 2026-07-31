"""Arxiv quant/algo-trading pipeline: source, DAG executor, persistence.

Submodules:
    db       — papers, runs, steps tables (shared demo.db)
    source   — arxiv API client + metadata parsing
    pipeline — DAG executor (download -> submit -> wait -> bundle -> store)

The runner lives in demo/scripts/arxiv_pipeline.py (host venv with boto3);
the container exposes status over /api/v1/arxiv/* via demo/arxiv_api.py.
"""
