import os
import sys
import json
import asyncio
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import parlant.sdk as p
import httpx
from dotenv import load_dotenv

# Import debug utilities
try:
    from app.debug_utils import (
        log_debug, timer, debug_tool, debug_api_call,
        perf_tracker, safe_json_loads
    )
except ImportError:
    from debug_utils import (
        log_debug, timer, debug_tool, debug_api_call,
        perf_tracker, safe_json_loads
    )

load_dotenv()

# =============================================
# =============== Configuration ================
# =============================================
class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Location
    FESTIVAL_COUNTRY = os.getenv("FESTIVAL_COUNTRY", "IN")
    DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", "19.0760"))
    DEFAULT_LON = float(os.getenv("DEFAULT_LON", "72.8777"))

    # Qdrant / DeepInfra
    DEEPINFRA_TOKEN = os.getenv("DEEPINFRA_TOKEN", "")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_KEY = os.getenv("QDRANT_KEY")
    CATALOG_COLLECTION = os.getenv("CATALOG_COLLECTION", "fashion_qwen4b_text")
    MEM_COLLECTION = os.getenv("MEM_COLLECTION", "mem0_fashion_qdrant")

    # Search
    DEFAULT_TOP_K = int(os.getenv("SEARCH_TOP_K", "12"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))
    HNSW_EF = int(os.getenv("HNSW_EF", "500"))

    # Agent
    AGENT_NAME = os.getenv("AGENT_NAME", "MuseBot")
    AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-5-nano")  # Changed from nano
    AGENT_DESCRIPTION = os.getenv(
        "AGENT_DESCRIPTION",
        "An intelligent fashion AI assistant specialized in Indian fashion trends."
    )

    # LLM Settings

    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "45.0"))
    
    # Memory
    MEMORY_SEARCH_LIMIT = int(os.getenv("MEMORY_SEARCH_LIMIT", "5"))
    MEMORY_TIMEOUT = float(os.getenv("MEMORY_TIMEOUT", "1.2"))
    
    # Trends
    TRENDS_REFRESH_MIN = int(os.getenv("TRENDS_REFRESH_MIN", "360"))
    TRENDS_LOOKAHEAD_DAYS = int(os.getenv("TRENDS_LOOKAHEAD_DAYS", "45"))
    FALLBACK_TRENDS = os.getenv(
        "FALLBACK_TRENDS",
        "Smart casuals,Versatile basics,Contemporary style"
    ).split(",")


# =============================================
# ============ Utilities & Helpers ============
# =============================================

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


async def _race(coro, timeout: float, fallback=None):
    """Run a coro with timeout; return fallback on TimeoutError."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        log_debug(f"Operation timed out after {timeout}s", level="WARNING")
        return fallback


# =============================================
# ============== Lazy Service Loader ===========
# =============================================
class Services:
    _mem = None
    _qdr = None
    _embed_catalog = None
    _rerank_qwen = None
    _lock = asyncio.Lock()
    _loaded = False

    @classmethod
    async def ensure_loaded(cls):
        if cls._loaded:
            return
        async with cls._lock:
            if cls._loaded:
                return
            
            with timer(f"[{Config.AGENT_NAME}] Loading services"):
                try:
                    from services.mem0_qdrant import build_mem0_qdrant
                    from services.deepinfra import embed_catalog, rerank_qwen
                except ImportError:
                    from .services.mem0_qdrant import build_mem0_qdrant
                    from .services.deepinfra import embed_catalog, rerank_qwen
                
                cls._mem, cls._qdr = build_mem0_qdrant()
                cls._embed_catalog = embed_catalog
                cls._rerank_qwen = rerank_qwen
                cls._loaded = True


# =============================================
# ================ LLM Utilities ==============
# =============================================
@debug_api_call("OpenAI")
async def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    json_mode: bool = False,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Universal LLM caller with structured output support."""
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY required")
    
    model = model or Config.AGENT_MODEL
    timeout = timeout or Config.LLM_TIMEOUT
    
    log_debug(f"LLM Request", level="API", model=model, json_mode=json_mode)
    
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    
    # Add JSON mode for supported models
    if json_mode and model not in ["gpt-5-nano"]:
        payload["response_format"] = {"type": "json_object"}
    
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        r.raise_for_status()
        
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        
        log_debug(f"LLM Response", level="API", length=len(content), tokens=data.get("usage", {}))
        
        if json_mode:
            return safe_json_loads(content, default={"response": content, "parse_error": True})
        
        return {"response": content}


# =============================================
# ================ Trend Service ===============
# =============================================
class TrendService:
    """Fetch & cache dynamic seasonal trends."""

    _cache: Dict[str, Any] = {}
    _lock = asyncio.Lock()
    _bg_task: Optional[asyncio.Task] = None

    @classmethod
    async def start(cls):
        await cls.refresh()
        cls._bg_task = asyncio.create_task(cls._periodic())

    @classmethod
    async def _periodic(cls):
        while True:
            await asyncio.sleep(Config.TRENDS_REFRESH_MIN * 60)
            try:
                await cls.refresh()
            except Exception as e:
                log_debug(f"Trend refresh error: {e}", level="ERROR")

    @classmethod
    async def get(cls) -> Dict[str, Any]:
        async with cls._lock:
            return dict(cls._cache) if cls._cache else cls._get_default()

    @classmethod
    def _get_default(cls) -> Dict[str, Any]:
        now = datetime.now()
        return {
            "current_month": now.strftime("%B"),
            "season": cls._infer_indian_season(now.month),
            "upcoming_festivals": [],
            "weather_summary": "Weather data unavailable",
            "trends": Config.FALLBACK_TRENDS,
            "season_styling_notes": "Versatile pieces for comfortable style",
            "cultural_highlights": [],
            "generated_at": now.isoformat(),
        }

    @classmethod
    async def refresh(cls) -> None:
        with timer("TrendService.refresh"):
            now = datetime.now()
            lookahead = now + timedelta(days=Config.TRENDS_LOOKAHEAD_DAYS)

            # Fetch external data
            holidays_task = asyncio.create_task(cls._fetch_holidays(now.year, lookahead.year))
            weather_task = asyncio.create_task(cls._fetch_weather(Config.DEFAULT_LAT, Config.DEFAULT_LON))
            
            results = await asyncio.gather(holidays_task, weather_task, return_exceptions=True)
            
            holidays = results[0] if not isinstance(results[0], Exception) else []
            weather_summary = results[1] if not isinstance(results[1], Exception) else "Weather unavailable"
            
            upcoming = [h for h in holidays if now.date() <= h["date"].date() <= lookahead.date()]
            upcoming_sorted = sorted(upcoming, key=lambda x: x["date"])[:8]

            season = cls._infer_indian_season(now.month)

            # Generate trends via LLM
            system_prompt = (
                "You are a fashion trend analyzer for India. Return ONLY valid JSON with: "
                '{"trends": ["trend1", ...], "season_styling_notes": "string", "cultural_highlights": ["item1", ...]}. '
                "Provide 4-6 relevant trends."
            )
            
            user_prompt = json.dumps({
                "current_date": now.strftime("%Y-%m-%d"),
                "season": season,
                "weather_summary": weather_summary,
                "upcoming_festivals": [{"name": h["name"], "date": h["date"].strftime("%Y-%m-%d")} for h in upcoming_sorted],
            }, ensure_ascii=False)
            
            try:
                llm_result = await call_llm(system_prompt, user_prompt, json_mode=True, timeout=30.0)
                trends = llm_result.get("trends") or Config.FALLBACK_TRENDS
                season_notes = llm_result.get("season_styling_notes", "")
                cultural = llm_result.get("cultural_highlights", [])
            except Exception as e:
                log_debug(f"LLM trend gen failed: {e}", level="ERROR")
                trends, season_notes, cultural = Config.FALLBACK_TRENDS, "", []

            async with cls._lock:
                cls._cache = {
                    "current_month": now.strftime("%B"),
                    "season": season,
                    "weather_summary": weather_summary,
                    "upcoming_festivals": upcoming_sorted,
                    "trends": trends,
                    "season_styling_notes": season_notes,
                    "cultural_highlights": cultural,
                    "generated_at": now.isoformat(),
                }
                log_debug(f"Trends refreshed: {len(upcoming_sorted)} festivals", level="SUCCESS")

    @staticmethod
    def _infer_indian_season(month: int) -> str:
        season_map = {
            (12, 1, 2): "Winter",
            (3, 4, 5): "Summer",
            (6, 7, 8, 9): "Monsoon",
            (10, 11): "Post-monsoon"
        }
        for months, season in season_map.items():
            if month in months:
                return season
        return "Transitional"

    @staticmethod
    @debug_api_call("Nager.Date Holidays")
    async def _fetch_holidays(year_a: int, year_b: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        base = "https://date.nager.at/api/v3/PublicHolidays"
        
        async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
            for y in sorted({year_a, year_b}):
                try:
                    r = await client.get(f"{base}/{y}/{Config.FESTIVAL_COUNTRY}")
                    r.raise_for_status()
                    
                    content_type = r.headers.get("content-type", "")
                    if "application/json" not in content_type:
                        log_debug(f"Unexpected content-type: {content_type}", level="WARNING")
                        continue
                    
                    data = r.json()
                    for d in data:
                        try:
                            results.append({
                                "name": d.get("localName") or d.get("name", "Holiday"),
                                "date": datetime.fromisoformat(d["date"]),
                            })
                        except (KeyError, ValueError):
                            continue
                            
                except Exception as e:
                    log_debug(f"Holiday fetch error {y}: {e}", level="WARNING")
        
        return results

    @staticmethod
    @debug_api_call("Open-Meteo Weather")
    async def _fetch_weather(lat: float, lon: float) -> str:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
            "forecast_days": 7,
            "timezone": "auto",
        }
        
        async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
            r = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
            r.raise_for_status()
            j = r.json()
        
        daily = j.get("daily", {})
        prec = sum(daily.get("precipitation_sum", []) or [0])
        tmax = max(daily.get("temperature_2m_max", []) or [25])
        tmin = min(daily.get("temperature_2m_min", []) or [15])
        
        return f"Next 7d: {tmin:.0f}–{tmax:.0f}°C, ~{prec:.0f}mm precip"


# =============================================
# ================== Tools ====================
# =============================================

@p.tool
@debug_tool
async def quick_greeting_check(context: p.ToolContext) -> p.ToolResult:
    """Lightning-fast greeting data fetch."""
    await Services.ensure_loaded()

    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or "guest"

    trends_obj = await TrendService.get()

    async def _mem_search():
        return await asyncio.to_thread(
            Services._mem.search,
            "user's name and fashion preferences",
            user_id=user_id,
            limit=Config.MEMORY_SEARCH_LIMIT,
        )

    results = await _race(_mem_search(), timeout=Config.MEMORY_TIMEOUT, fallback=None)

    user_name: Optional[str] = None
    preference: Optional[str] = None

    try:
        if results:
            memories = [r.get("text") or r.get("memory") for r in results.get("results", []) if r]
            for mem in memories:
                if mem and "name is" in mem.lower():
                    m = re.search(r"name is (\w+)", mem, re.IGNORECASE)
                    if m:
                        user_name = m.group(1).capitalize()
                        break
            for mem in memories:
                if mem and any(w in mem.lower() for w in ["love", "prefer", "like", "favorite"]):
                    preference = mem
                    break
    except Exception as e:
        log_debug(f"Memory parse error: {e}", level="WARNING")

    return p.ToolResult(
        data={
            "user_name": user_name,
            "preference": preference,
            "trends": trends_obj.get("trends", []),
            "season": trends_obj.get("season"),
            "upcoming_festivals": trends_obj.get("upcoming_festivals", []),
            "is_returning": user_name is not None,
        }
    )


@p.tool
@debug_tool
async def analyze_catalog_schema(context: p.ToolContext) -> p.ToolResult:
    """Sample catalog for schema."""
    await Services.ensure_loaded()

    def _sample():
        return Services._qdr.scroll(
            collection_name=Config.CATALOG_COLLECTION,
            limit=100,
            with_payload=True,
        )

    results = await asyncio.to_thread(_sample)
    points = results[0] if results else []

    all_colors, all_sizes, all_brands, all_categories = set(), set(), set(), set()
    price_min, price_max = float("inf"), 0

    for point in points:
        payload = point.payload or {}
        commerce = payload.get("commerce", {})
        all_colors.update(commerce.get("colors_in_stock", []) or [])
        all_sizes.update(commerce.get("sizes_in_stock", []) or [])
        if brand := payload.get("brand"):
            all_brands.add(brand)
        if cat := payload.get("category_leaf"):
            all_categories.add(cat)
        price = commerce.get("price")
        if isinstance(price, (int, float)):
            price_min = min(price_min, price)
            price_max = max(price_max, price)

    if price_min == float("inf"):
        price_min = 0

    return p.ToolResult(data={
        "available_colors": sorted(all_colors),
        "available_sizes": sorted(all_sizes),
        "available_brands": sorted(all_brands),
        "available_categories": sorted(all_categories),
        "price_range": {"min": int(price_min), "max": int(price_max)},
    })


@p.tool
@debug_tool
async def get_contextual_knowledge(context: p.ToolContext, query: str) -> p.ToolResult:
    """Get fashion context."""
    now = datetime.now()
    trends_state = await TrendService.get()

    system_prompt = f"""You are a fashion context analyzer for India. Return JSON with:
{{"current_date": "YYYY-MM-DD", "season": "string", "season_styling_notes": "string", 
"cultural_context": ["array"], "weather_considerations": "string", "trending_styles": ["array"], 
"occasion_insights": "string"}}

Today: {now.strftime('%B %d, %Y')}. Context: {json.dumps(trends_state, ensure_ascii=False)}"""

    try:
        return p.ToolResult(data=await call_llm(system_prompt, f'Query: "{query}"', json_mode=True))
    except Exception as e:
        return p.ToolResult(data={
            "current_date": now.strftime("%Y-%m-%d"),
            "season": trends_state.get("season"),
            "error": str(e),
        })


@p.tool
@debug_tool
async def intelligent_query_analyzer(context: p.ToolContext, text: str) -> p.ToolResult:
    """Deep query understanding."""

    system_prompt = """Analyze fashion query. Return JSON:
{"intent": "product_search|outfit_advice|preference_statement|question|off_topic", 
"confidence": 0.0-1.0, "normalized_query": "string", 
"extracted_entities": {"colors": [], "product_types": [], "occasions": [], "styles": [], 
"fits": [], "materials": [], "brands": [], "sizes": [], 
"price_constraints": {"min": null, "max": null, "budget": null}}, 
"implied_filters": {"must_be_in_stock": true/false, "preferred_price_segment": "budget|mid|premium|luxury|null"}, 
"user_sentiment": "excited|casual|confused|frustrated", 
"is_fashion_related": true/false}

Handle spelling errors and Indian English patterns."""

    try:
        return p.ToolResult(data=await call_llm(system_prompt, f'Analyze: "{text}"', json_mode=True, timeout=20.0))
    except Exception as e:
        return p.ToolResult(data={
            "intent": "product_search",
            "normalized_query": text,
            "is_fashion_related": True,
            "error": str(e),
        })


@p.tool
@debug_tool
async def build_smart_filters(
    context: p.ToolContext,
    query_analysis_json: str,
    catalog_schema_json: str,
) -> p.ToolResult:
    """Map query → Qdrant filters."""
    try:
        query_analysis = safe_json_loads(query_analysis_json)
        catalog_schema = safe_json_loads(catalog_schema_json)
    except Exception as e:
        return p.ToolResult(data={"filters": {}, "reasoning": f"Parse error: {e}"})

    system_prompt = """Build Qdrant filters. Return JSON:
{"filters": {"colors_in_stock": [], "sizes_in_stock": [], "price_range": {"min": null, "max": null}, 
"brand": [], "category_leaf": [], "in_stock": true/false}, 
"reasoning": "string", "search_strategy": "narrow|balanced|broad"}"""

    try:
        return p.ToolResult(data=await call_llm(
            system_prompt, 
            f"Query: {json.dumps(query_analysis)}\nSchema: {json.dumps(catalog_schema)}", 
            json_mode=True, 
            timeout=15.0
        ))
    except Exception as e:
        return p.ToolResult(data={"filters": {}, "reasoning": f"Fallback: {e}"})


@p.tool
@debug_tool
async def search_catalog(
    context: p.ToolContext,
    query: str,
    filters_json: str = "{}",
    top_k: int = None,
) -> p.ToolResult:
    """Vector search with reranking."""
    await Services.ensure_loaded()
    top_k = top_k or Config.DEFAULT_TOP_K

    filters = safe_json_loads(filters_json, default={})
    vec = (await Services._embed_catalog([query]))[0]

    from qdrant_client.http import models as rest

    must: List[Any] = []
    if colors := filters.get("colors_in_stock"):
        must.append(rest.FieldCondition(key="commerce.colors_in_stock", match=rest.MatchAny(any=colors)))
    if sizes := filters.get("sizes_in_stock"):
        must.append(rest.FieldCondition(key="commerce.sizes_in_stock", match=rest.MatchAny(any=sizes)))
    if pr := filters.get("price_range"):
        if (mn := pr.get("min")) is not None:
            must.append(rest.FieldCondition(key="commerce.price", range=rest.Range(gte=mn)))
        if (mx := pr.get("max")) is not None:
            must.append(rest.FieldCondition(key="commerce.price", range=rest.Range(lte=mx)))
    if filters.get("in_stock") is True:
        must.append(rest.FieldCondition(key="commerce.in_stock", match=rest.MatchValue(value=True)))

    qdrant_filter = rest.Filter(must=must) if must else None

    def _search():
        return Services._qdr.query_points(
            collection_name=Config.CATALOG_COLLECTION,
            query=vec,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            search_params=rest.SearchParams(hnsw_ef=Config.HNSW_EF),
        )

    results = await asyncio.to_thread(_search)

    items = []
    for point in (results.points or []):
        payload = point.payload or {}
        commerce = payload.get("commerce", {})
        items.append({
            "product_id": payload.get("product_id"),
            "title": payload.get("title"),
            "brand": payload.get("brand"),
            "category": payload.get("category_leaf"),
            "price_inr": commerce.get("price"),
            "in_stock": commerce.get("in_stock"),
            "colors_available": commerce.get("colors_in_stock", []),
            "score": float(point.score),
        })

    if items and len(items) > 1:
        try:
            rerank_texts = [f"{it['title']} {it['brand']}" for it in items]
            indices = await Services._rerank_qwen(query, rerank_texts, top_k=Config.RERANK_TOP_K)
            items = [items[i] for i in indices if i < len(items)]
        except Exception as e:
            log_debug(f"Rerank error: {e}", level="WARNING")

    return p.ToolResult(data={"query": query, "applied_filters": filters, "items": items})


@p.tool
@debug_tool
async def generate_product_presentation(
    context: p.ToolContext,
    products_json: str,
    user_context_json: str,
    query_info_json: str,
) -> p.ToolResult:
    """Generate witty product descriptions."""
    try:
        products = safe_json_loads(products_json)
        user_context = safe_json_loads(user_context_json)
        query_info = safe_json_loads(query_info_json)
    except Exception as e:
        return p.ToolResult(data={"opening_line": "Here's what I found!", "products": [], "error": str(e)})

    system_prompt = """Fashion stylist. Return JSON:
{"presentation_style": "string", "opening_line": "string", 
"products": [{"product_index": 0, "description": "string", "contextual_note": "string"}], 
"closing_question": "string"}"""

    try:
        result = await call_llm(
            system_prompt, 
            f"Context: {json.dumps(user_context)}\nQuery: {json.dumps(query_info)}\nProducts: {json.dumps(products[:8])}", 
            json_mode=True
        )
        return p.ToolResult(data=result)
    except Exception as e:
        return p.ToolResult(data={
            "opening_line": "Here's what I found!",
            "products": [{"product_index": i, "description": p.get("title", "")} for i, p in enumerate(products[:5])],
            "error": str(e),
        })


@p.tool
@debug_tool
async def save_user_preference(context: p.ToolContext, preference: str) -> p.ToolResult:
    """Save preference."""
    await Services.ensure_loaded()
    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or "guest"
    Services._mem.add(
        messages=[{"role": "user", "content": preference}],
        user_id=user_id,
        metadata={"domain": "fashion"},
        infer=True,
    )
    return p.ToolResult(data={"status": "saved"})


@p.tool
@debug_tool
async def get_user_profile(context: p.ToolContext) -> p.ToolResult:
    """Get user profile."""
    await Services.ensure_loaded()
    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or "guest"
    results = Services._mem.search("fashion preferences", user_id=user_id, limit=10)
    memories = [r.get("text") or r.get("memory") for r in results.get("results", []) if r]
    return p.ToolResult(data={"user_id": user_id, "preferences": memories})


@p.tool
@debug_tool
async def save_user_name(context: p.ToolContext, name: str) -> p.ToolResult:
    """Save name."""
    await Services.ensure_loaded()
    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or "guest"
    Services._mem.add(
        messages=[{"role": "user", "content": f"My name is {name}"}],
        user_id=user_id,
        metadata={"type": "name"},
        infer=True,
    )
    return p.ToolResult(data={"status": "saved", "name": name})


@p.tool
@debug_tool
async def prepare_and_search(context: p.ToolContext, text: str) -> p.ToolResult:
    """End-to-end search pipeline."""
    # Parallel phase
    qa_t = intelligent_query_analyzer(context, text)
    ctx_t = get_contextual_knowledge(context, text)
    sch_t = analyze_catalog_schema(context)
    prof_t = get_user_profile(context)

    results = await asyncio.gather(qa_t, ctx_t, sch_t, prof_t, return_exceptions=True)

    def _data(x, default):
        return x.data if isinstance(x, p.ToolResult) else default

    qa = _data(results[0], {"normalized_query": text})
    ctx = _data(results[1], {})
    sch = _data(results[2], {})
    prof = _data(results[3], {})

    # Filters
    filt_r = await build_smart_filters(context, json.dumps(qa), json.dumps(sch))
    filters = filt_r.data if isinstance(filt_r, p.ToolResult) else {"filters": {}}

    # Search
    search_r = await search_catalog(context, qa.get("normalized_query", text), json.dumps(filters.get("filters", {})))
    items = (search_r.data or {}).get("items", []) if isinstance(search_r, p.ToolResult) else []
    
    if not items:
        search_r = await search_catalog(context, qa.get("normalized_query", text), "{}")
        items = (search_r.data or {}).get("items", []) if isinstance(search_r, p.ToolResult) else []

    # Presentation
    pres = await generate_product_presentation(
        context,
        json.dumps(items),
        json.dumps({"profile": prof, "context": ctx}),
        json.dumps(qa),
    )

    return p.ToolResult(
        data={
            "analysis": qa,
            "context": ctx,
            "schema": sch,
            "filters": filters,
            "search": search_r.data if isinstance(search_r, p.ToolResult) else {},
            "presentation": pres.data if isinstance(pres, p.ToolResult) else {},
        }
    )


# =============================================
# ================ Retriever ==================
# =============================================
async def fashion_memory_retriever(context: p.RetrieverContext) -> p.RetrieverResult:
    await Services.ensure_loaded()
    message = context.interaction.last_customer_message
    if not message or not message.content:
        return p.RetrieverResult(None)

    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or "guest"

    results = Services._mem.search(message.content, user_id=user_id, limit=Config.MEMORY_SEARCH_LIMIT)
    memories = [r.get("text") or r.get("memory") for r in results.get("results", []) if r]
    return p.RetrieverResult({"memories": memories} if memories else None)


# =============================================
# ================== Server ===================
# =============================================
async def main() -> None:
    if sys.platform.startswith("win"):
        _install_windows_exception_filter()

    log_debug(f"🚀 {Config.AGENT_NAME} Starting...", level="INFO")
    log_debug(f"Model: {Config.AGENT_MODEL}, Debug: {os.getenv('DEBUG_MODE', 'true')}", level="INFO")

    # Preload services
    with timer(f"[{Config.AGENT_NAME}] Pre-loading services"):
        await Services.ensure_loaded()
        await TrendService.start()
    
    log_debug(f"✅ All services ready!", level="SUCCESS")

    async with p.Server(session_store="local") as server:
        agent = await server.create_agent(
            name=Config.AGENT_NAME,
            description=Config.AGENT_DESCRIPTION,
        )

        await agent.attach_retriever(fashion_memory_retriever, id="fashion_memory")

        # ================= First Message Greeting =================
        await agent.create_guideline(
            condition=(
                "The conversation has exactly ONE message from the customer "
                "AND zero messages from the agent"
            ),
            action=(
                "This is the first interaction! IMMEDIATELY:\n"
                "1. Call quick_greeting_check tool\n"
                "2. If is_returning=true and user_name exists:\n"
                "   - Greet warmly by name\n"
                "   - Mention one trend\n"
                "3. If is_returning=false:\n"
                "   - Introduce as MuseBot\n"
                "   - Mention two trends\n"
                "   - Ask their name\n"
                "4. Then address their original message\n"
                "Keep greeting to MAX 2 sentences."
            ),
            tools=[quick_greeting_check, save_user_name],
        )

        # ================= Name Handling =================
        await agent.create_guideline(
            condition="User shares their name",
            action="Call save_user_name. Respond warmly then continue naturally.",
            tools=[save_user_name],
        )

        # ================= Fashion Query =================
        await agent.create_guideline(
            condition="User sends a fashion-related query or product search",
            action=(
                "Call prepare_and_search to parallelize all analysis and search. "
                "It will auto-broaden if no results."
            ),
            tools=[
                prepare_and_search,
                intelligent_query_analyzer,
                get_contextual_knowledge,
                analyze_catalog_schema,
                build_smart_filters,
                search_catalog,
                generate_product_presentation,
                save_user_preference,
                get_user_profile,
            ],
        )

        # ================= No Results =================
        await agent.create_guideline(
            condition="Search returns no or few results",
            action=(
                "Try:\n"
                "1. Call search_catalog with empty filters\n"
                "2. Call analyze_catalog_schema for alternatives\n"
                "3. Relax filters\n"
                "Be empathetic and suggest alternatives."
            ),
            tools=[analyze_catalog_schema, search_catalog],
        )

        # ================= Preference Learning =================
        await agent.create_guideline(
            condition="User expresses a preference",
            action="Call save_user_preference. Acknowledge briefly.",
            tools=[save_user_preference],
        )

        # ================= Off-Topic =================
        await agent.create_guideline(
            condition="User asks unrelated to fashion",
            action="Gracefully redirect with personality.",
            tools=[],
        )

        log_debug(f"🎨 {Config.AGENT_NAME} ready!", level="SUCCESS")
        log_debug(f"⚡ Parallel pipeline enabled", level="INFO")
        log_debug(f"🌐 Server: http://localhost:8800/chat/", level="INFO")
        log_debug(f"📊 Debug mode: {os.getenv('DEBUG_MODE', 'true')}", level="INFO")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Print performance summary on exit
        perf_tracker.print_summary()