# fashion_pro_fixed.py
import os
import sys
import json
import asyncio
import re
from datetime import datetime
import parlant.sdk as p
import httpx
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ---------- Windows: Selector loop + silence Proactor WinError 10054 ----------
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

def _install_windows_exception_filter():
    loop = asyncio.get_running_loop()
    default_handler = loop.get_exception_handler()

    def _handler(loop_, context):
        exc = context.get("exception")
        # Drop spurious "existing connection was forcibly closed" spam
        if isinstance(exc, ConnectionResetError):
            try:
                if exc.errno == 10054:
                    return
            except Exception:
                pass
        if default_handler:
            default_handler(loop_, context)
        else:
            loop_.default_exception_handler(context)

    loop.set_exception_handler(_handler)


# ---------- Config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CATALOG_COLLECTION = os.getenv("CATALOG_COLLECTION", "fashion_qwen4b_text")

# Lazy services (so boot is instant)
_mem = None
_qdr = None
_embed_catalog = None   # await _embed_catalog([text]) -> [vector]
_rerank_qwen = None     # await _rerank_qwen(query, candidates, top_k) -> indices
_services_lock = asyncio.Lock()

def _log(msg: str):
    print(f"[fashion_pro] {msg}", flush=True)


async def _ensure_services():
    """Load mem0, Qdrant client, and embed/rerank lazily on first tool use."""
    global _mem, _qdr, _embed_catalog, _rerank_qwen
    if _mem and _qdr and _embed_catalog and _rerank_qwen:
        return
    async with _services_lock:
        if _mem and _qdr and _embed_catalog and _rerank_qwen:
            return
        _log("Loading mem0 + Qdrant + embed/rerank services (lazy)...")
        try:
            from services.mem0_qdrant import build_mem0_qdrant
        except Exception:
            from .services.mem0_qdrant import build_mem0_qdrant  # if used as a package
        try:
            from services.deepinfra import embed_catalog as _ec, rerank_qwen as _rq
        except Exception:
            from .services.deepinfra import embed_catalog as _ec, rerank_qwen as _rq
        _mem, _qdr = build_mem0_qdrant()
        _embed_catalog = _ec
        _rerank_qwen = _rq
        _log("✓ Services ready")


# ----------------------- Light, local intent heuristics -----------------------
_COLOR_WORDS = {
    "black","white","navy","blue","light blue","royal blue","red","maroon","green","olive",
    "beige","tan","brown","grey","gray","charcoal","cream","khaki","pink","purple","lavender",
    "teal","mustard","orange"
}
_OCCASION_WORDS = {"date","date night","wedding","interview","party","office","work","gym","beach","festival","concert","meeting","brunch","dinner"}
_FRAGRANCE_WORDS = {"perfume","cologne","fragrance","edt","edp","parfum","sillage","projection"}
_PRODUCT_WORDS = {"shirt","t-shirt","tee","chinos","jeans","trousers","sneakers","shoes","loafer","jacket","blazer","hoodie","kurta","saree","lehenga","dress","watch","belt"}
_PREF_TRIGGERS = {"i love","i like","i prefer","my size","favorite","favourite","budget","under","below","max"}
_OFFTOPIC_HINTS = {"flight","visa","hotel","api","python","bitcoin","stock","tax","physics","chemistry","politics","weather in","news"}

def _is_phrase_in(text, words):
    t = " " + text.lower() + " "
    return any((" " + w + " ") in t for w in words)

def _has_any(text, words):
    tl = text.lower()
    return any(w in tl for w in words)

def _looks_like_preference(text):
    tl = text.lower()
    if _has_any(tl, _PREF_TRIGGERS):
        return True
    # Simple color-only preference like "Black" / "I love black"
    if any(c in tl for c in _COLOR_WORDS):
        if any(k in tl for k in ["love","like","prefer","favourite","favorite"]):
            return True
    # Size/budget numeric hints
    if re.search(r"\b(xs|s|m|l|xl|xxl)\b", tl): return True
    if re.search(r"(?:₹|rs\.?|inr|\$|usd)\s*\d+", tl): return True
    if re.search(r"\bunder\s*\d+", tl): return True
    return False

def _local_analyze(text: str):
    tl = text.strip().lower()
    if not tl:
        return {"intent":"product_search","normalized_query":text,"subqueries":[],"entities":{},"followups":[]}
    if _has_any(tl, _OFFTOPIC_HINTS):
        return {"intent":"off_topic","normalized_query":text,"subqueries":[],"entities":{},"followups":["What event are you shopping for?"]}
    if _looks_like_preference(text):
        return {"intent":"preference_statement","normalized_query":text,"subqueries":[],"entities":{},"followups":[]}
    if _is_phrase_in(tl, _OCCASION_WORDS) or "what to wear" in tl or "what should i wear" in tl or "outfit" in tl:
        return {"intent":"outfit_advice","normalized_query":text,"subqueries":[],"entities":{},"followups":["Occasion? Venue? Palette? Budget? Size?"]}
    if _has_any(tl, _FRAGRANCE_WORDS):
        return {"intent":"fragrance","normalized_query":text,"subqueries":[],"entities":{},"followups":[]}
    if _has_any(tl, _PRODUCT_WORDS) or "recommend" in tl or "buy" in tl or "suggest" in tl:
        return {"intent":"product_search","normalized_query":text,"subqueries":[],"entities":{},"followups":[]}
    # Unknown → let OpenAI try (if available) else product_search
    return None


# ----------------------- Tools (renamed to avoid built-in collisions) -----------------------

@p.tool
async def analyze_query(context: p.ToolContext, text: str) -> p.ToolResult:
    """
    Hybrid analyzer:
    1) local heuristic (fast, no network) for common cases
    2) OpenAI fallback if ambiguous
    3) robust fallback to product_search if anything fails
    """
    local = _local_analyze(text)
    if local is not None:
        return p.ToolResult(data=local)

    if not OPENAI_API_KEY:
        return p.ToolResult(data={
            "intent": "product_search",
            "normalized_query": text,
            "subqueries": [],
            "entities": {},
            "followups": []
        })

    sys_prompt = (
        "You correct spelling, normalize fashion queries, and split complex asks "
        "into small subqueries. Return STRICT JSON with keys: "
        "intent, normalized_query, subqueries[], entities{}, followups[]. "
        "Valid intents: product_search, outfit_advice, fragrance, preference_statement, off_topic."
    )
    try:
        async with httpx.AsyncClient(timeout=45, trust_env=False) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": text},
                    ],
                },
            )
            r.raise_for_status()
            payload = r.json()
            content = payload["choices"][0]["message"]["content"]
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("non-dict JSON")
            return p.ToolResult(data=data)
    except Exception as e:
        _log(f"analyze_query fallback due to: {e!r}")
        return p.ToolResult(data={
            "intent": "product_search",
            "normalized_query": text,
            "subqueries": [],
            "entities": {},
            "followups": []
        })


def _safe_rerank(order, n):
    """Clamp/validate reranker output indices; fall back to natural order if bad."""
    try:
        if not order:
            return list(range(n))
        order = [i for i in order if isinstance(i, int) and 0 <= i < n]
        if not order:
            return list(range(n))
        seen = set(order)
        tail = [i for i in range(n) if i not in seen]
        return order + tail
    except Exception:
        return list(range(n))


@p.tool
async def catalog_search(context: p.ToolContext, query: str, top_k: int = 12) -> p.ToolResult:
    """Qdrant search + rerank; services load lazily."""
    await _ensure_services()
    vec = (await _embed_catalog([query]))[0]

    def _q():
        from qdrant_client.http import models as rest
        return _qdr.query_points(
            collection_name=CATALOG_COLLECTION,
            query=vec,
            limit=top_k,
            with_payload=["product_id", "title", "category_leaf", "brand", "commerce", "price_inr"],
            search_params=rest.SearchParams(hnsw_ef=500),
        )

    res = await asyncio.to_thread(_q)
    hits = []
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

    # Robust rerank path with safe fallback
    try:
        if hits:
            order = await _rerank_qwen(
                query,
                [f"{h['title']} | {h['category']} | {h['brand']}" for h in hits],
                top_k=min(8, len(hits))
            )
            order = _safe_rerank(order, len(hits))
            hits = [hits[i] for i in order]
    except Exception as e:
        _log(f"rerank fallback due to: {e!r}")
        # Keep hits as-is (Qdrant score order)

    return p.ToolResult(data={"query": query, "items": hits})


@p.tool
async def save_preference_to_mem0(context: p.ToolContext, statement: str) -> p.ToolResult:
    """
    Avoid collision with built-in names. Use p.Customer.current for id.
    """
    await _ensure_services()
    cust = getattr(p.Customer, "current", None)
    uid = getattr(cust, "id", None) or getattr(cust, "name", None) or "guest"
    _mem.add(messages=[{"role": "user", "content": statement}],
             user_id=uid, metadata={"domain": "fashion"}, infer=True)
    return p.ToolResult(data="ok")


@p.tool
async def summarize_style_profile_mem0(context: p.ToolContext) -> p.ToolResult:
    """Renamed to avoid collisions; use p.Customer.current."""
    await _ensure_services()
    cust = getattr(p.Customer, "current", None)
    uid = getattr(cust, "id", None) or getattr(cust, "name", None) or "guest"
    top = _mem.search("summarize stable fashion preferences", user_id=uid, limit=8)
    lines = [r.get("text") or r.get("memory") for r in top.get("results", []) if r]
    return p.ToolResult(data="\n".join(lines) if lines else "No prior preferences yet.")


async def mem0_retriever(context: p.RetrieverContext) -> p.RetrieverResult:
    """Retriever uses p.Customer.current; no ToolContext.customer access."""
    await _ensure_services()
    msg = context.interaction.last_customer_message and context.interaction.last_customer_message.content
    if not msg:
        return p.RetrieverResult(None)
    cust = getattr(p.Customer, "current", None)
    uid = getattr(cust, "id", None) or getattr(cust, "name", None) or "guest"
    res = _mem.search(msg, user_id=uid, limit=4)
    found = [r.get("text") or r.get("memory") for r in res.get("results", []) if r]
    return p.RetrieverResult({"memories": found})


# --------- Small glossary & a tiny journey (keeps entity cache fast) ---------

@p.tool
async def get_trending_items(context: p.ToolContext) -> p.ToolResult:
    return p.ToolResult([
        "Oxford Shirt – White",
        "Slim Chinos – Navy",
        "Minimal Sneakers – White",
        "Crewneck Tee – Black",
    ])

@p.tool
async def place_order(context: p.ToolContext, item: str) -> p.ToolResult:
    ts = datetime.now().strftime("%Y%m%d")
    return p.ToolResult(data=f"Order placed for '{item}'. Confirmation: ORD-{ts}-001")


async def add_domain_glossary(agent: p.Agent) -> None:
    await agent.create_term(name="Return Policy",
                            description="Items can be returned within 14 days if unworn and tagged.")
    await agent.create_term(name="Size Guide",
                            description="Use the size chart for waist/chest/length measurements.")
    await agent.create_term(name="Fabric Care",
                            description="Care instructions for cotton, wool, leather, and denim.")


async def create_shopping_journey(server: p.Server, agent: p.Agent) -> p.Journey:
    journey = await agent.create_journey(
        title="Shop an Outfit",
        description="Helps the customer pick an outfit and place an order.",
        conditions=["The customer wants to buy clothes or an outfit"],
    )
    t0 = await journey.initial_state.transition_to(
        chat_state="Ask what occasion or item they are shopping for"
    )
    t1 = await t0.target.transition_to(tool_state=get_trending_items)
    t2 = await t1.target.transition_to(
        chat_state="List trending items and ask which one they want"
    )
    t3 = await t2.target.transition_to(
        chat_state="Confirm size/color and address before placing order",
        condition="The customer picks an item",
    )
    t4 = await t3.target.transition_to(
        tool_state=place_order,
        condition="The customer confirms the details",
    )
    t5 = await t4.target.transition_to(
        chat_state="Confirm the order has been placed and provide the confirmation number"
    )
    await t5.target.transition_to(state=p.END_JOURNEY)
    return journey


# ----------------------- Main (exact healthcare.py pattern) -----------------------

async def main() -> None:
    if sys.platform.startswith("win"):
        _install_windows_exception_filter()

    # Explicit session_store=local to keep everything in-process
    async with p.Server(session_store="local") as server:
        agent = await server.create_agent(
            name="MuseBot",
            description=(
                "Your playful fashion co-pilot. Asks smart follow-ups (occasion, venue, weather, "
                "palette, size, budget, vibe), then gives crisp outfit or product picks. "
                "Keeps it helpful and cheeky—within fashion only."
            ),
        )

        # Attach retriever (lazy services ensure no heavy work at boot)
        await agent.attach_retriever(mem0_retriever, id="mem0")

        # Small glossary → quick entity caching
        await add_domain_glossary(agent)

        # Light journey parity with healthcare.py
        await create_shopping_journey(server, agent)

        # Core behavior: include preference handling path explicitly
        await agent.create_guideline(
            condition="on every new customer message",
            action=(
                "Call analyze_query(text=<message>). "
                "If analyze_query.intent == preference_statement: "
                "  Acknowledge and call save_preference_to_mem0(statement=<message>); "
                "  then ask one short follow-up relevant to their stated preference. "
                "Else if intent in [product_search, outfit_advice, fragrance]: "
                "  ask 2–3 short clarifiers if key info is missing (occasion, venue/setting, climate, "
                "  fit/size, color palette, budget, preferred brands); then call catalog_search(query=...). "
                "Present 3–5 options with a one-line 'why' and price (INR) if available; "
                "finish with a single question to move the convo forward."
            ),
            tools=[analyze_query, catalog_search, save_preference_to_mem0],
        )

        # Occasion-specific coaching (date night, etc.)
        await agent.create_guideline(
            condition=(
                "The customer asks what to wear for a specific occasion "
                "(e.g., date night, wedding, interview, party)"
            ),
            action=(
                "Ask for venue (indoor/outdoor), dress code, climate, preferred palette, and budget. "
                "Offer 3 concise outfit directions (top/bottom/shoes/accessories) with why each fits the vibe. "
                "Invite them to pick a lane or tweak details."
            ),
        )

        # Summarize known tastes
        await agent.create_guideline(
            condition="the customer asks what you know about their tastes",
            action="Call summarize_style_profile_mem0 and weave a friendly summary.",
            tools=[summarize_style_profile_mem0],
        )

        # Off-topic guardrail (playful)
        await agent.create_guideline(
            condition="The customer inquires about something unrelated to fashion, outfits, grooming, or fragrances",
            action=(
                "Reply with a brief playful line like: "
                "'I’m stitched for style, not rocket science 😄 — tell me the occasion and vibe, and I’ll dress you right.' "
                "Then ask what they’re shopping for or the event they have."
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
