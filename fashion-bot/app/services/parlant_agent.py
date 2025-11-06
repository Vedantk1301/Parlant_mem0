import os, asyncio, json, httpx
import parlant.sdk as p
from qdrant_client.http import models as rest
from .mem0_qdrant import build_mem0_qdrant
from .deepinfra import embed_catalog, rerank_qwen

# Build shared services (Mem0 + Qdrant client)
mem, qdr = build_mem0_qdrant()
CATALOG_COLLECTION = os.getenv("CATALOG_COLLECTION", "fashion_qwen4b_text")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

@p.tool
async def analyze_query(context: p.ToolContext, text: str) -> p.ToolResult:
    """
    Intent classifier + spelling normalization → strict JSON.
    """
    sys = ("You correct spelling, normalize fashion queries, and split complex asks "
           "into small subqueries. Return STRICT JSON with keys: "
           "intent, normalized_query, subqueries[], entities{}, followups[].")
    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type":"application/json"},
            json={"model":"gpt-5-nano","response_format":{"type":"json_object"},
                  "messages":[{"role":"system","content":sys},{"role":"user","content":text}]}
        )
        r.raise_for_status()
    data = json.loads(r.json()["choices"][0]["message"]["content"])
    return p.ToolResult(data)

@p.tool
async def search_catalog(context: p.ToolContext, query: str, top_k: int = 12) -> p.ToolResult:
    vec = (await embed_catalog([query]))[0]
    def _q():
        return qdr.query_points(
            collection_name=CATALOG_COLLECTION,
            query=vec, limit=top_k,
            with_payload=["product_id","title","category_leaf","brand","commerce","price_inr"],
            search_params=rest.SearchParams(hnsw_ef=500),
        )
    res = await asyncio.to_thread(_q)
    hits=[]
    for pnt in (res.points or []):
        pl = pnt.payload or {}
        hits.append({
            "title": pl.get("title"),
            "brand": pl.get("brand"),
            "category": pl.get("category_leaf"),
            "in_stock": (pl.get("commerce") or {}).get("in_stock"),
            "price_inr": pl.get("price_inr"),
            "score": float(pnt.score),
            "product_id": pl.get("product_id"),
        })
    if hits:
        order = await rerank_qwen(query, [f"{h['title']} | {h['category']} | {h['brand']}" for h in hits], top_k=min(8,len(hits)))
        hits = [hits[i] for i in order]
    return p.ToolResult({"query": query, "items": hits})

@p.tool
async def save_user_memory(context: p.ToolContext, statement: str) -> p.ToolResult:
    """
    Persist stable user preference. Use infer=True for normal flow.
    """
    uid = context.customer.id or "guest"
    mem.add(messages=[{"role":"user","content": statement}], user_id=uid, metadata={"domain":"fashion"}, infer=True)
    return p.ToolResult("ok")

@p.tool
async def summarize_style_profile(context: p.ToolContext) -> p.ToolResult:
    uid = context.customer.id or "guest"
    top = mem.search("summarize stable fashion preferences", user_id=uid, limit=8)
    lines = [r.get("text") or r.get("memory") for r in top.get("results", []) if r]
    return p.ToolResult("\n".join(lines) if lines else "No prior preferences yet.")

async def mem0_retriever(context: p.RetrieverContext) -> p.RetrieverResult:
    msg = context.interaction.last_customer_message and context.interaction.last_customer_message.content
    if not msg: return p.RetrieverResult(None)
    uid = context.customer.id or "guest"
    res = mem.search(msg, user_id=uid, limit=4)
    found = [r.get("text") or r.get("memory") for r in res.get("results", []) if r]
    return p.RetrieverResult({"memories": found})

async def boot_parlant_server() -> tuple[p.Server, str]:
    # Parlant reads OPENAI_API_KEY from env
    server = p.Server(nlp_service=p.NLPServices.openai)
    await server.__aenter__()

    agent = await server.create_agent(
        name="AeroStylist",
        description=("Friendly fashion stylist. Ask for missing constraints "
                     "(occasion, budget, climate, palette, size). Offer 3–5 picks with 1-line why. "
                     "Never claim stock unless provided.")
    )
    await agent.attach_retriever(mem0_retriever, id="mem0")

    g1 = await agent.create_guideline(
        condition="on every new customer message",
        action=("Call analyze_query(text=<message>). If intent in {product_search,outfit_advice,fragrance}: "
                "compose a concise query from normalized_query + entities; call search_catalog(query=...). "
                "Present 3–5 options with brief why & price (INR) if available. "
                "Ask one follow-up if key info is missing."),
        tools=[analyze_query, search_catalog],
    )
    await g1.reevaluate_after(analyze_query)

    await agent.create_guideline(
        condition="the customer states a stable preference (size, color, fit, budget, brand, climate)",
        action="Acknowledge and call save_user_memory(statement=the exact preference).",
        tools=[save_user_memory],
    )

    await agent.create_guideline(
        condition="the customer asks what you know about their tastes",
        action="Call summarize_style_profile and weave a friendly summary.",
        tools=[summarize_style_profile],
    )

    return server, agent.id
