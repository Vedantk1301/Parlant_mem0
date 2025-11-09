import os
import sys
import json
import asyncio
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import parlant.sdk as p
import httpx
from dotenv import load_dotenv

load_dotenv()

# =============================================
# =============== Configuration ================
# =============================================
class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # External data (free, no key)
    FESTIVAL_COUNTRY = os.getenv("FESTIVAL_COUNTRY", "IN")
    DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", "19.0760"))
    DEFAULT_LON = float(os.getenv("DEFAULT_LON", "72.8777"))

    # Qdrant / DeepInfra
    DEEPINFRA_TOKEN = os.getenv("DEEPINFRA_TOKEN", "")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_KEY = os.getenv("QDRANT_KEY")
    CATALOG_COLLECTION = os.getenv("CATALOG_COLLECTION", "fashion_qwen4b_text")
    MEM_COLLECTION = os.getenv("MEM_COLLECTION", "mem0_fashion_qdrant")

    # Search / Rerank
    DEFAULT_TOP_K = int(os.getenv("SEARCH_TOP_K", "12"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))
    HNSW_EF = int(os.getenv("HNSW_EF", "500"))

    # Agent
    AGENT_NAME = os.getenv("AGENT_NAME", "MuseBot")
    AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-5-nano")
    AGENT_DESCRIPTION = os.getenv(
        "AGENT_DESCRIPTION",
        "An intelligent fashion AI assistant. You understand context deeply, "
        "learn from every interaction, and provide personalized recommendations. "
        "You're witty but helpful, fashion-focused, and culturally aware. "
        "You ALWAYS greet users warmly on their first message."
    )

    # Trends refresh cadence
    TRENDS_REFRESH_MIN = int(os.getenv("TRENDS_REFRESH_MIN", "360"))
    TRENDS_LOOKAHEAD_DAYS = int(os.getenv("TRENDS_LOOKAHEAD_DAYS", "45"))

    # LLM Settings
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "45.0"))
    
    # Memory Settings
    MEMORY_SEARCH_LIMIT = int(os.getenv("MEMORY_SEARCH_LIMIT", "5"))
    MEMORY_TIMEOUT = float(os.getenv("MEMORY_TIMEOUT", "1.2"))
    
    # Fallback Trends
    FALLBACK_TRENDS = os.getenv(
        "FALLBACK_TRENDS",
        "Smart casuals,Versatile basics,Contemporary style,Comfortable fits,Seasonal essentials,Classic pieces"
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
            print(f"[{Config.AGENT_NAME}] 🔄 Loading services...", flush=True)
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
            print(f"[{Config.AGENT_NAME}] ✅ Services ready", flush=True)


# =============================================
# ================ LLM Utilities ==============
# =============================================
async def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    json_mode: bool = False,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Universal LLM caller with optional structured output.
    gpt-5-nano doesn't support temperature or response_format parameters.
    """
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY required for LLM calls")
    
    model = model or Config.AGENT_MODEL
    timeout = timeout or Config.LLM_TIMEOUT
    
    # Build base payload
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    
    # Only add temperature if not gpt-5-nano
    if model != "gpt-5-nano" and temperature is not None:
        payload["temperature"] = temperature
    elif temperature is None:
        payload["temperature"] = Config.LLM_TEMPERATURE
    
    # Only add response_format if not gpt-5-nano AND json_mode requested
    if json_mode and model != "gpt-5-nano":
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
        content = r.json()["choices"][0]["message"]["content"]
        
        # Try to parse JSON if requested, even without json_mode
        if json_mode:
            try:
                # Clean up potential markdown code blocks
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"[call_llm] JSON parse failed for {model}: {e}, returning as text")
                return {"response": content, "parse_error": str(e)}
        
        return {"response": content}


# =============================================
# ================ Trend Service ===============
# =============================================
class TrendService:
    """Fetch & cache dynamic seasonal trends, upcoming holidays, and weather-based notes."""

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
                print(f"[TrendService] refresh error: {type(e).__name__}: {e!r}")

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
            "season_styling_notes": "Versatile pieces for comfortable everyday style",
            "cultural_highlights": [],
            "generated_at": now.isoformat(),
        }

    @classmethod
    async def refresh(cls) -> None:
        now = datetime.now()
        month_name = now.strftime("%B")
        lookahead = now + timedelta(days=Config.TRENDS_LOOKAHEAD_DAYS)

        # Fetch external context concurrently
        holidays_task = asyncio.create_task(
            cls._fetch_public_holidays(now.year, lookahead.year)
        )
        weather_task = asyncio.create_task(
            cls._fetch_weather_summary(Config.DEFAULT_LAT, Config.DEFAULT_LON)
        )
        
        results = await asyncio.gather(
            holidays_task, weather_task, return_exceptions=True
        )
        
        holidays = results[0] if not isinstance(results[0], Exception) else []
        weather_summary = results[1] if not isinstance(results[1], Exception) else "Weather data unavailable"
        
        if isinstance(results[0], Exception):
            print(f"[TrendService] holidays error: {type(results[0]).__name__}: {results[0]!r}")
        if isinstance(results[1], Exception):
            print(f"[TrendService] weather error: {type(results[1]).__name__}: {results[1]!r}")

        upcoming = [
            h for h in holidays
            if now.date() <= h["date"].date() <= lookahead.date()
        ]
        upcoming_sorted = sorted(upcoming, key=lambda x: x["date"])[:8]

        season = cls._infer_indian_season(now.month)

        # Synthesize style trends via LLM
        system_prompt = (
            "You are a fashion context synthesizer for India. Analyze the provided context and return ONLY valid JSON "
            "(no markdown, no preamble) with these exact keys: "
            '{"trends": ["trend1", "trend2", ...], "season_styling_notes": "string", "cultural_highlights": ["item1", "item2", ...]}. '
            "Provide 4-6 trends relevant to the season and upcoming events."
        )
        
        user_prompt = json.dumps(
            {
                "current_date": now.strftime("%Y-%m-%d"),
                "month": month_name,
                "season": season,
                "weather_summary": weather_summary,
                "upcoming_festivals": [
                    {"name": h["name"], "date": h["date"].strftime("%Y-%m-%d")}
                    for h in upcoming_sorted
                ],
            },
            ensure_ascii=False,
        )
        
        try:
            llm_result = await call_llm(
                system_prompt, 
                user_prompt, 
                model=Config.AGENT_MODEL,
                json_mode=True,
                timeout=30.0
            )
            trends = llm_result.get("trends") or Config.FALLBACK_TRENDS
            season_notes = llm_result.get("season_styling_notes", "")
            cultural = llm_result.get("cultural_highlights", [])
        except Exception as e:
            print(f"[TrendService] LLM trend gen error: {type(e).__name__}: {e!r}")
            trends = Config.FALLBACK_TRENDS
            season_notes = "Comfortable and versatile styles for the season"
            cultural = []

        async with cls._lock:
            cls._cache = {
                "current_month": month_name,
                "season": season,
                "weather_summary": weather_summary,
                "upcoming_festivals": upcoming_sorted,
                "trends": trends,
                "season_styling_notes": season_notes,
                "cultural_highlights": cultural,
                "generated_at": datetime.now().isoformat(),
            }
            print(
                f"[TrendService] ✅ refreshed trends for {month_name}; "
                f"{len(upcoming_sorted)} festivals upcoming"
            )

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
    async def _fetch_public_holidays(year_a: int, year_b: int) -> List[Dict[str, Any]]:
        """Fetch holidays from Nager.Date API with better error handling."""
        base = "https://date.nager.at/api/v3/PublicHolidays"
        results: List[Dict[str, Any]] = []
        
        async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
            for y in sorted({year_a, year_b}):
                url = f"{base}/{y}/{Config.FESTIVAL_COUNTRY}"
                try:
                    r = await client.get(url)
                    r.raise_for_status()
                    
                    # Check if response is valid JSON
                    content_type = r.headers.get("content-type", "")
                    if "application/json" not in content_type:
                        print(f"[TrendService] Unexpected content-type for holidays: {content_type}")
                        continue
                    
                    data = r.json()
                    
                    if not isinstance(data, list):
                        print(f"[TrendService] Unexpected response format for holidays")
                        continue
                    
                    for d in data:
                        try:
                            results.append({
                                "name": d.get("localName") or d.get("name", "Holiday"),
                                "date": datetime.fromisoformat(d["date"]),
                            })
                        except (KeyError, ValueError) as e:
                            continue
                            
                except httpx.HTTPStatusError as e:
                    print(f"[TrendService] HTTP {e.response.status_code} fetching holidays for {y}")
                except Exception as e:
                    print(f"[TrendService] holiday fetch {y} error: {type(e).__name__}: {e!r}")
        
        return results

    @staticmethod
    async def _fetch_weather_summary(lat: float, lon: float) -> str:
        """Fetch weather from Open-Meteo with better error handling."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
            "forecast_days": 7,
            "timezone": "auto",
        }
        
        try:
            async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
                r = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
                r.raise_for_status()
                j = r.json()
            
            daily = j.get("daily", {})
            prec = sum(daily.get("precipitation_sum", []) or [0])
            tmax = max(daily.get("temperature_2m_max", []) or [25])
            tmin = min(daily.get("temperature_2m_min", []) or [15])
            monsoonish = prec >= 30
            
            return (
                f"Next 7d: {tmin:.0f}–{tmax:.0f}°C, precipitation total ~{prec:.0f}mm; "
                + ("monsoon-like showers likely" if monsoonish else "mostly dry")
            )
        except Exception as e:
            print(f"[TrendService] weather fetch error: {type(e).__name__}: {e!r}")
            return "Weather forecast unavailable"


# =============================================
# ================== Tools ====================
# =============================================

@p.tool
async def quick_greeting_check(context: p.ToolContext) -> p.ToolResult:
    """
    ⚡ Lightning-fast greeting data fetch.
    Combines cached trends + best-effort memory probe in parallel.
    """
    await Services.ensure_loaded()

    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or getattr(customer, "name", None) or "guest"

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
            memories = [
                r.get("text") or r.get("memory") for r in results.get("results", []) if r
            ]
            for mem in memories:
                if mem and "name is" in mem.lower():
                    m = re.search(r"name is (\w+)", mem, re.IGNORECASE)
                    if m:
                        user_name = m.group(1).capitalize()
                        break
            for mem in memories:
                if mem and any(
                    w in mem.lower() for w in ["love", "prefer", "like", "favorite", "fan of"]
                ):
                    preference = mem
                    break
    except Exception as e:
        print(f"[quick_greeting] parse error: {type(e).__name__}: {e!r}")

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
async def analyze_catalog_schema(context: p.ToolContext) -> p.ToolResult:
    """Sample catalog to derive available fields and dynamic filter space."""
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

    schema = {
        "available_colors": sorted(all_colors),
        "available_sizes": sorted(all_sizes),
        "available_brands": sorted(all_brands),
        "available_categories": sorted(all_categories),
        "price_range": {"min": int(price_min), "max": int(price_max)},
        "filterable_fields": [
            "commerce.colors_in_stock",
            "commerce.sizes_in_stock",
            "commerce.price",
            "commerce.in_stock",
            "brand",
            "category_leaf",
        ],
    }

    return p.ToolResult(data=schema)


@p.tool
async def get_contextual_knowledge(context: p.ToolContext, query: str) -> p.ToolResult:
    """LLM-driven seasonal, cultural and weather-aware context (India-focused)."""
    now = datetime.now()
    trends_state = await TrendService.get()

    system_prompt = f"""You are a fashion context analyzer. Given the current date and a user query, 
provide relevant contextual information in JSON format (no markdown, no preamble) for India with these exact keys:
{{"current_date": "YYYY-MM-DD", "season": "string", "season_styling_notes": "string", 
"cultural_context": ["array"], "weather_considerations": "string", "trending_styles": ["array"], 
"occasion_insights": "string"}}

Today is {now.strftime('%B %d, %Y, %A')}. Use this context: {json.dumps(trends_state, ensure_ascii=False)}"""

    user_prompt = f'User query: "{query}"\nProvide contextual fashion knowledge relevant to this query and the current date.'

    try:
        result = await call_llm(system_prompt, user_prompt, json_mode=True)
        return p.ToolResult(data=result)
    except Exception as e:
        return p.ToolResult(
            data={
                "current_date": now.strftime("%Y-%m-%d"),
                "season": trends_state.get("season", "Unknown"),
                "weather_considerations": trends_state.get("weather_summary", ""),
                "trending_styles": trends_state.get("trends", []),
                "error": str(e),
            }
        )


@p.tool
async def intelligent_query_analyzer(context: p.ToolContext, text: str) -> p.ToolResult:
    """Deep query understanding using LLM reasoning."""

    system_prompt = """You are an expert fashion query analyzer. Analyze the user's message and return ONLY valid JSON 
(no markdown, no preamble) with these exact keys:
{"intent": "product_search|outfit_advice|preference_statement|question|off_topic", "confidence": 0.0-1.0, 
"normalized_query": "string", "extracted_entities": {"colors": [], "product_types": [], "occasions": [], 
"styles": [], "fits": [], "materials": [], "brands": [], "sizes": [], 
"price_constraints": {"min": null, "max": null, "budget": null}}, 
"implied_filters": {"must_be_in_stock": true/false, "preferred_price_segment": "budget|mid|premium|luxury|null"}, 
"user_sentiment": "excited|casual|confused|frustrated", "suggested_clarifications": [], 
"is_fashion_related": true/false, "off_topic_reason": "string or null"}

Be smart about spelling errors, abbreviations, and Indian English patterns."""

    user_prompt = f'Analyze this query: "{text}"'

    try:
        result = await call_llm(system_prompt, user_prompt, json_mode=True, timeout=20.0)
        return p.ToolResult(data=result)
    except Exception as e:
        return p.ToolResult(
            data={
                "intent": "product_search",
                "confidence": 0.5,
                "normalized_query": text,
                "extracted_entities": {},
                "is_fashion_related": True,
                "error": str(e),
            }
        )


@p.tool
async def build_smart_filters(
    context: p.ToolContext,
    query_analysis_json: str,
    catalog_schema_json: str,
) -> p.ToolResult:
    """Map query entities → precise Qdrant filters using LLM + schema."""
    try:
        query_analysis = json.loads(query_analysis_json)
        catalog_schema = json.loads(catalog_schema_json)
    except json.JSONDecodeError as e:
        return p.ToolResult(
            data={"filters": {}, "reasoning": f"JSON parse error: {e}", "search_strategy": "broad"}
        )

    system_prompt = """You are a search filter constructor. Given query analysis + catalog schema, 
produce Qdrant filters in JSON format (no markdown, no preamble) with these possible keys:
{"filters": {"colors_in_stock": [], "sizes_in_stock": [], "price_range": {"min": null, "max": null}, 
"brand": [], "category_leaf": [], "in_stock": true/false}, 
"reasoning": "string", "search_strategy": "narrow|balanced|broad"}

Only include filters that make sense based on the query. Don't be overly restrictive."""

    user_prompt = f"Query analysis:\n{json.dumps(query_analysis, indent=2)}\n\nSchema:\n{json.dumps(catalog_schema, indent=2)}\n\nBuild filters."

    try:
        result = await call_llm(system_prompt, user_prompt, json_mode=True, timeout=15.0)
        return p.ToolResult(data=result)
    except Exception as e:
        return p.ToolResult(data={"filters": {}, "reasoning": f"Fallback: {e}", "search_strategy": "broad"})


@p.tool
async def search_catalog(
    context: p.ToolContext,
    query: str,
    filters_json: str = "{}",
    top_k: int = None,
) -> p.ToolResult:
    """Vector search in Qdrant with dynamic filters; rerank via Qwen on DeepInfra."""
    await Services.ensure_loaded()
    top_k = top_k or Config.DEFAULT_TOP_K

    try:
        filters = json.loads(filters_json) if filters_json else {}
    except json.JSONDecodeError:
        filters = {}

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
    if cats := filters.get("category_leaf"):
        must.append(rest.FieldCondition(key="category_leaf", match=rest.MatchAny(any=cats)))
    if brands := filters.get("brand"):
        must.append(rest.FieldCondition(key="brand", match=rest.MatchAny(any=brands)))

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
            "discount_pct": commerce.get("discount_pct"),
            "in_stock": commerce.get("in_stock"),
            "colors_available": commerce.get("colors_in_stock", []) or [],
            "sizes_available": commerce.get("sizes_in_stock", []) or [],
            "url": payload.get("url"),
            "image": payload.get("primary_image"),
            "score": float(point.score),
        })

    if items and len(items) > 1:
        try:
            rerank_texts = [f"{it['title']} by {it['brand']} - {it['category']}" for it in items]
            rerank_indices = await Services._rerank_qwen(
                query, rerank_texts, top_k=min(Config.RERANK_TOP_K, len(items))
            )
            items = [items[i] for i in rerank_indices if 0 <= i < len(items)]
        except Exception as e:
            print(f"[search_catalog] Rerank error: {type(e).__name__}: {e!r}")

    return p.ToolResult(
        data={
            "query": query,
            "applied_filters": filters,
            "total_found": len(items),
            "items": items[:Config.RERANK_TOP_K],
        }
    )


@p.tool
async def generate_product_presentation(
    context: p.ToolContext,
    products_json: str,
    user_context_json: str,
    query_info_json: str,
) -> p.ToolResult:
    """Witty, contextual product blurbs generated by LLM."""
    try:
        products = json.loads(products_json)
        user_context = json.loads(user_context_json)
        query_info = json.loads(query_info_json)
    except json.JSONDecodeError as e:
        return p.ToolResult(
            data={
                "presentation_style": "casual",
                "opening_line": "Here's what I found!",
                "products": [],
                "closing_question": "Which one interests you?",
                "error": f"JSON parse error: {e}",
            }
        )

    system_prompt = """You are a witty, fashion-savvy stylist. Return JSON (no markdown, no preamble) with these exact keys:
{"presentation_style": "string", "opening_line": "string", 
"products": [{"product_index": 0, "description": "string", "contextual_note": "string"}], 
"closing_question": "string"}

Tie notes to season/festival/occasion if relevant. Be enthusiastic but natural."""

    user_prompt = (
        f"User query info: {json.dumps(query_info, indent=2)}\n\n"
        f"User context: {json.dumps(user_context, indent=2)}\n\n"
        f"Products to present: {json.dumps(products[:8], indent=2)}\n"
        "Create the presentation."
    )

    try:
        result = await call_llm(system_prompt, user_prompt, json_mode=True)
        return p.ToolResult(data=result)
    except Exception as e:
        return p.ToolResult(
            data={
                "presentation_style": "casual",
                "opening_line": "Here's what I found!",
                "products": [
                    {"product_index": i, "description": p.get("title", ""), "contextual_note": ""}
                    for i, p in enumerate(products[:5])
                ],
                "closing_question": "Which one catches your eye?",
                "error": str(e),
            }
        )


@p.tool
async def save_user_preference(context: p.ToolContext, preference: str) -> p.ToolResult:
    """Persist user preference into memory."""
    await Services.ensure_loaded()

    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or getattr(customer, "name", None) or "guest"

    Services._mem.add(
        messages=[{"role": "user", "content": preference}],
        user_id=user_id,
        metadata={"domain": "fashion", "timestamp": datetime.now().isoformat()},
        infer=True,
    )

    return p.ToolResult(data={"status": "saved", "user_id": user_id})


@p.tool
async def get_user_profile(context: p.ToolContext) -> p.ToolResult:
    """Retrieve the user's fashion profile from memory."""
    await Services.ensure_loaded()

    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or getattr(customer, "name", None) or "guest"

    results = Services._mem.search(
        "What are this user's fashion preferences, sizes, budget, and favorite styles?",
        user_id=user_id,
        limit=10,
    )

    memories = [r.get("text") or r.get("memory") for r in results.get("results", []) if r]

    return p.ToolResult(data={"user_id": user_id, "preferences": memories, "has_profile": len(memories) > 0})


@p.tool
async def save_user_name(context: p.ToolContext, name: str) -> p.ToolResult:
    """Persist user's name for personalization."""
    await Services.ensure_loaded()
    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or "guest"
    Services._mem.add(
        messages=[{"role": "user", "content": f"My name is {name}"}],
        user_id=user_id,
        metadata={"domain": "personal", "type": "name"},
        infer=True,
    )
    return p.ToolResult(data={"status": "saved", "name": name})


@p.tool
async def prepare_and_search(context: p.ToolContext, text: str) -> p.ToolResult:
    """
    One-stop pipeline that parallelizes analysis, context, schema & profile,
    then builds filters, searches, and generates the presentation.
    """

    # 1) Run heavy-but-independent steps together
    qa_t = intelligent_query_analyzer(context, text)
    ctx_t = get_contextual_knowledge(context, text)
    sch_t = analyze_catalog_schema(context)
    prof_t = get_user_profile(context)

    results = await asyncio.gather(
        qa_t, ctx_t, sch_t, prof_t, return_exceptions=True
    )

    # unwrap or default
    def _data(x, default):
        return x.data if isinstance(x, p.ToolResult) else default

    qa = _data(results[0], {"normalized_query": text, "extracted_entities": {}})
    ctx = _data(results[1], {})
    sch = _data(results[2], {"available_colors": [], "available_sizes": [], "available_brands": [], "available_categories": [], "price_range": {"min": 0, "max": 0}})
    prof = _data(results[3], {"preferences": [], "has_profile": False})

    # 2) Filters via LLM
    filt_r = await build_smart_filters(context, json.dumps(qa), json.dumps(sch))
    filters = filt_r.data if isinstance(filt_r, p.ToolResult) else {"filters": {}}

    # 3) Search (may be narrow). If zero, broaden.
    search_r = await search_catalog(
        context, qa.get("normalized_query", text), json.dumps(filters.get("filters", {}))
    )

    items = (search_r.data or {}).get("items", []) if isinstance(search_r, p.ToolResult) else []
    if not items:
        # broaden search without filters
        search_r = await search_catalog(context, qa.get("normalized_query", text), "{}")
        items = (search_r.data or {}).get("items", []) if isinstance(search_r, p.ToolResult) else []

    # 4) Presentation
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
    user_id = getattr(customer, "id", None) or getattr(customer, "name", None) or "guest"

    results = Services._mem.search(message.content, user_id=user_id, limit=Config.MEMORY_SEARCH_LIMIT)
    memories = [r.get("text") or r.get("memory") for r in results.get("results", []) if r]
    return p.RetrieverResult({"memories": memories} if memories else None)


# =============================================
# ================== Server ===================
# =============================================
async def main() -> None:
    if sys.platform.startswith("win"):
        _install_windows_exception_filter()

    # Preload all services & trend cache before server starts listening
    print(f"[{Config.AGENT_NAME}] 🚀 Pre-loading services...", flush=True)
    await Services.ensure_loaded()
    await TrendService.start()
    print(f"[{Config.AGENT_NAME}] ✅ Services & Trend cache ready!", flush=True)

    async with p.Server(session_store="local") as server:
        agent = await server.create_agent(
            name=Config.AGENT_NAME,
            description=Config.AGENT_DESCRIPTION,
        )

        # Attach memory retriever
        await agent.attach_retriever(fashion_memory_retriever, id="fashion_memory")

        # ================= CRITICAL: First Message Greeting =================
        await agent.create_guideline(
            condition=(
                "The conversation has exactly ONE message from the customer "
                "AND zero messages from the agent"
            ),
            action=(
                "This is the first interaction! IMMEDIATELY do this:\n\n"
                "1. Call quick_greeting_check tool\n"
                "2. Based on the result:\n"
                "   - If is_returning is true and user_name exists:\n"
                "     * Greet warmly by their name with excitement\n"
                "     * Mention one trend from the trends array\n"
                "     * Ask what brings them today\n"
                "   - If is_returning is false (new user):\n"
                "     * Give warm introduction as MuseBot\n"
                "     * Mention two trends from the trends array\n"
                "     * Ask their name casually\n"
                "3. Then naturally address their original message in the next sentence\n"
                "RULES:\n"
                "- Keep greeting to MAX 2 sentences before addressing their message\n"
                "- Use trends data from the tool response\n"
                "- If they said Hello or Hi, acknowledge it casually then ask how you can help\n"
                "- If they asked a product question, answer it after the greeting"
            ),
            tools=[quick_greeting_check, save_user_name],
        )

        # ================= Name Handling =================
        await agent.create_guideline(
            condition="User shares their name",
            action=(
                "Call save_user_name with the name. Respond warmly like: 'Nice to meet you NAME!' "
                "Then continue conversation naturally."
            ),
            tools=[save_user_name],
        )

        # ================= Fashion Query Handling (PARALLEL) =================
        await agent.create_guideline(
            condition="User sends a fashion-related query or product search request",
            action=(
                "Call prepare_and_search with the user's text to parallelize analysis, context, schema, profile, "
                "filter construction, search and presentation in one shot. If no results, the tool will auto-broaden."
            ),
            tools=[
                prepare_and_search,
                # keep individual tools available for follow-ups
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

        # ================= No Results Handling =================
        await agent.create_guideline(
            condition="Search returns no results or very few results",
            action=(
                "Try these strategies:\n"
                "1. Call search_catalog again with empty filters for broader search\n"
                "2. Call analyze_catalog_schema and suggest alternatives (synonyms, nearby categories)\n"
                "3. Relax filters: remove color, raise price cap by ~50%, broaden category\n"
                "Be empathetic and always propose alternatives."
            ),
            tools=[analyze_catalog_schema, search_catalog],
        )

        # ================= Preference Learning =================
        await agent.create_guideline(
            condition="User expresses a preference or style choice",
            action=(
                "Call save_user_preference with the exact statement. Acknowledge briefly, then continue."
            ),
            tools=[save_user_preference],
        )

        # ================= Off-Topic Handling =================
        await agent.create_guideline(
            condition="User asks something unrelated to fashion",
            action=(
                "Gracefully redirect with personality (keep it light, friendly, on-brand)."
            ),
            tools=[],
        )

        print(f"[{Config.AGENT_NAME}] 🎨 Optimized agent ready!")
        print(f"[{Config.AGENT_NAME}] ⚡ Parallel pipeline enabled (prepare_and_search)")
        print(f"[{Config.AGENT_NAME}] 🌐 Server running at http://localhost:8800/chat/")


if __name__ == "__main__":
    asyncio.run(main())