import os, json, numpy as np, httpx

DI_OPENAI_BASE = "https://api.deepinfra.com/v1/openai"
DI_INFER_BASE  = "https://api.deepinfra.com/v1/inference"

EMB_MODEL_CATALOG = "Qwen/Qwen3-Embedding-4B"
RERANK_MODEL      = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")

async def embed_catalog(texts: list[str]) -> list[list[float]]:
    token = os.getenv("DEEPINFRA_TOKEN")
    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
        r = await client.post(
            f"{DI_OPENAI_BASE}/embeddings",
            headers={"Authorization": f"Bearer {token}", "Content-Type":"application/json"},
            json={"model": EMB_MODEL_CATALOG, "input": texts, "encoding_format":"float"}
        )
        r.raise_for_status()
        data = r.json()["data"]
    arr = np.asarray([row["embedding"] for row in data], dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True); norms[norms==0]=1.0
    return (arr/norms).tolist()

async def rerank_qwen(query: str, docs: list[str], top_k: int = 8) -> list[int]:
    if not docs:
        return []
    token = os.getenv("DEEPINFRA_TOKEN")
    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
        r = await client.post(
            f"{DI_INFER_BASE}/{RERANK_MODEL}",
            headers={"Authorization": f"Bearer {token}", "Content-Type":"application/json"},
            json={"queries":[query], "documents": docs}
        )
        r.raise_for_status()
    scores = r.json().get("scores", [])
    if scores and isinstance(scores[0], list):
        scores = scores[0]
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:min(top_k, len(docs))]
    return order
