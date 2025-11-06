# AeroStylist (FastAPI + Parlant + Mem0/Qdrant + DeepInfra)

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U -r requirements.txt
cp .env.example .env  # edit with your keys
python -m app.main
```

Endpoint: `POST /chat` → `{ "customer_id": "vedant", "message": "..." }`

## Notes

* Parlant agent + guidelines + tools are started in the FastAPI lifespan hook (startup), and shut down cleanly on exit. (FastAPI lifespan is the recommended modern approach).
* Mem0 stores memories in **Qdrant** (single collection, fixed dims).
* Catalog search uses **DeepInfra**: Qwen3-Embedding-4B for ANN and Qwen3-Reranker for re-ranking.

