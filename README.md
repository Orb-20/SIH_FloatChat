# FloatChat-ARGO

Scaffold for the SIH PoC: AI-powered conversational interface for ARGO ocean data discovery.

> Created: 2025-08-28T10:54:06.431899Z

## Structure

- `floatchat_argo/` — shared python package (config, utils)
- `data/` — raw/interim/processed parquet partitions
- `db/` — SQL schema & migrations
- `scripts/` — ETL scripts (ingest NetCDF → parquet → PostgreSQL; build vector index)
- `services/api/` — FastAPI backend (RAG + SQL translation)
- `services/frontend/` — Streamlit dashboard & chat UI
- `configs/` — env, docker-compose, app configs
- `docs/` — diagrams and documentation
- `tests/` — unit tests

## Quickstart

1. Use Python **3.11** (recommended for best library compatibility).
2. Create a virtualenv and install requirements:
   ```bash
   python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r configs/requirements.txt
   ```
3. Copy environment:
   ```bash
   cp .env.example .env  # Windows: copy .env.example .env
   ```
4. (Optional) Start infra with Docker:
   ```bash
   docker compose -f configs/docker-compose.yml up -d
   ```

Then proceed with **Step 1** in our chat: environment setup & schema creation.
