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
import time

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

    # Search - TWO modes now!
    SIMPLE_SEARCH_LIMIT = int(os.getenv("SIMPLE_SEARCH_LIMIT", "15"))  # For specific queries
    DISCOVERY_QUERIES = int(os.getenv("DISCOVERY_QUERIES", "4"))  # For broad/pairing queries
    PRODUCTS_PER_QUERY = int(os.getenv("PRODUCTS_PER_QUERY", "40"))
    FINAL_RERANK_TOP_K = int(os.getenv("FINAL_RERANK_TOP_K", "12"))
    HNSW_EF = int(os.getenv("HNSW_EF", "500"))

    # Agent
    AGENT_NAME = os.getenv("AGENT_NAME", "MuseBot")
    AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-5-nano")
    AGENT_DESCRIPTION = os.getenv(
        "AGENT_DESCRIPTION",
        "A witty, conversational fashion AI that helps you discover perfect outfits 🎨"
    )

    # LLM Settings
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "15.0"))
    
    # Memory
    MEMORY_SEARCH_LIMIT = int(os.getenv("MEMORY_SEARCH_LIMIT", "8"))
    MEMORY_TIMEOUT = float(os.getenv("MEMORY_TIMEOUT", "1.0"))

    # Caching - AGGRESSIVE
    SEARCH_CACHE_TTL_HOURS = int(os.getenv("SEARCH_CACHE_TTL_HOURS", "24"))  # 24h for searches
    WEB_SEARCH_CACHE_TTL_HOURS = int(os.getenv("WEB_SEARCH_CACHE_TTL_HOURS", "12"))
    TREND_CACHE_TTL_HOURS = int(os.getenv("TREND_CACHE_TTL_HOURS", "12"))
    CACHE_DIR = os.getenv("CACHE_DIR", ".")
    SEARCH_CACHE_FILE = os.path.join(CACHE_DIR, "search_cache.json")  # NEW
    WEB_SEARCH_CACHE_FILE = os.path.join(CACHE_DIR, "web_search_cache.json")
    TREND_CACHE_FILE = os.path.join(CACHE_DIR, "trend_cache.json")
    
    # Trends
    TRENDS_REFRESH_MIN = int(os.getenv("TRENDS_REFRESH_MIN", "360"))
    TRENDS_LOOKAHEAD_DAYS = int(os.getenv("TRENDS_LOOKAHEAD_DAYS", "45"))
    FALLBACK_TRENDS = os.getenv(
        "FALLBACK_TRENDS",
        "Smart casuals,Versatile basics,Contemporary style"
    ).split(",")

# Fashion domains
INDIA_FASHION_DOMAINS = [
    "vogue.in", "gqindia.com", "lifestyleasia.com",
    "timesofindia.indiatimes.com", "thehindu.com",
    "moneycontrol.com", "fashionnetwork.com",
    "ndtv.com", "mensxp.com",
]
GLOBAL_FASHION_DOMAINS = [
    "vogue.com", "wwd.com", "businessoffashion.com",
    "highsnobiety.com", "hypebeast.com", "theimpression.com",
    "harpersbazaar.com", "elle.com", "gq.com", "whowhatwear.com",
]

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

def _read_json_file(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _write_json_file(path: str, data: Dict[str, Any]) -> None:
    try:
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass

# =============================================
# ============ NEW: Search Cache ==============
# =============================================
_SEARCH_CACHE_LOCK = asyncio.Lock()

async def get_cached_search(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached search result if fresh (24h TTL)."""
    async with _SEARCH_CACHE_LOCK:
        cache_all = _read_json_file(Config.SEARCH_CACHE_FILE)
        entry = cache_all.get(cache_key)
        if entry and isinstance(entry, dict):
            ts = float(entry.get("ts", 0))
            ttl_secs = Config.SEARCH_CACHE_TTL_HOURS * 3600
            if (time.time() - ts) < ttl_secs:
                return entry.get("val")
    return None

async def save_cached_search(cache_key: str, result: Dict[str, Any]) -> None:
    """Save search result to cache."""
    async with _SEARCH_CACHE_LOCK:
        cache_all = _read_json_file(Config.SEARCH_CACHE_FILE)
        cache_all[cache_key] = {"ts": time.time(), "val": result}
        _write_json_file(Config.SEARCH_CACHE_FILE, cache_all)

# =============================================
# ============ Web Search (Cached) ============
# =============================================
_WEB_SEARCH_CACHE_LOCK = asyncio.Lock()

@debug_api_call("OpenAI Web Search")
async def openai_web_search(
    prompt: str, 
    *,
    allowed_domains: Optional[List[str]] = None,
    country: Optional[str] = None, 
    city: Optional[str] = None,
    timeout: float = 60.0
) -> Dict[str, Any]:
    """Use OpenAI Responses API with web_search tool (12h cached)."""
    
    cache_key = json.dumps(
        {"prompt": prompt, "allowed_domains": allowed_domains, "country": country, "city": city},
        sort_keys=True,
        ensure_ascii=False,
    )
    ttl_secs = Config.WEB_SEARCH_CACHE_TTL_HOURS * 3600
    now = time.time()

    async with _WEB_SEARCH_CACHE_LOCK:
        cache_all = _read_json_file(Config.WEB_SEARCH_CACHE_FILE)
        entry = cache_all.get(cache_key)
        if entry and isinstance(entry, dict) and (now - float(entry.get("ts", 0))) < ttl_secs:
            return entry.get("val", {"text": "", "sources": []})

    if not Config.OPENAI_API_KEY:
        return {"text": "", "sources": []}

    tool: Dict[str, Any] = {"type": "web_search"}
    
    if country or city:
        user_loc: Dict[str, Any] = {"type": "approximate"}
        if country:
            user_loc["country"] = country
        if city:
            user_loc["city"] = city
        tool["user_location"] = user_loc
    
    if allowed_domains:
        tool["filters"] = {"allowed_domains": allowed_domains}

    payload: Dict[str, Any] = {
        "model": Config.AGENT_MODEL,
        "tools": [tool],
        "tool_choice": "auto",
        "input": prompt,
        "reasoning": {"effort": "low"},
    }

    headers = {
        "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=True) as client:
            r = await client.post(
                "https://api.openai.com/v1/responses", 
                headers=headers, 
                json=payload
            )
            r.raise_for_status()
            data = r.json()

        text = data.get("output_text", "").strip()
        sources: List[Dict[str, str]] = []
        
        for item in data.get("output", []):
            if item.get("type") == "web_search_call":
                action = item.get("action", {})
                for source in action.get("sources", []):
                    if source.get("type") == "url":
                        sources.append({
                            "title": source.get("title", ""),
                            "url": source.get("url", "")
                        })
            elif item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        for ann in content.get("annotations", []):
                            if ann.get("type") == "url_citation":
                                sources.append({
                                    "title": ann.get("title", ""),
                                    "url": ann.get("url", "")
                                })

        unique_sources = []
        seen_urls = set()
        for s in sources:
            url = s.get("url")
            if url and url not in seen_urls:
                unique_sources.append(s)
                seen_urls.add(url)

        result = {"text": text, "sources": unique_sources[:8]}

        async with _WEB_SEARCH_CACHE_LOCK:
            cache_all = _read_json_file(Config.WEB_SEARCH_CACHE_FILE)
            cache_all[cache_key] = {"ts": now, "val": result}
            _write_json_file(Config.WEB_SEARCH_CACHE_FILE, cache_all)

        return result
    
    except httpx.ReadTimeout:
        log_debug(f"Web search timed out after {timeout}s", level="WARNING")
        return {"text": "", "sources": []}
    except Exception as e:
        log_debug(f"Web search error: {e}", level="ERROR")
        return {"text": "", "sources": []}

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
@debug_api_call("OpenAI Responses API")
async def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    json_mode: bool = False,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Universal LLM caller using Responses API."""
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY required")
    
    model = model or Config.AGENT_MODEL
    timeout = timeout or Config.LLM_TIMEOUT
    
    log_debug(f"LLM Request", level="API", model=model, json_mode=json_mode)
    
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    payload: Dict[str, Any] = {
        "model": model,
        "input": combined_prompt,
        "reasoning": {"effort": "low"},
    }
    
    headers = {
        "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=True) as client:
            r = await client.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                json=payload,
            )
            r.raise_for_status()
            
            data = r.json()
            content = data.get("output_text", "").strip()
            
            if not content:
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for c in item.get("content", []):
                            if c.get("type") == "output_text":
                                content = c.get("text", "")
                                break
                        if content:
                            break
            
            log_debug(f"LLM Response", level="API", length=len(content))
            
            if json_mode:
                content_clean = content.strip()
                if content_clean.startswith("```json"):
                    content_clean = content_clean[7:]
                if content_clean.startswith("```"):
                    content_clean = content_clean[3:]
                if content_clean.endswith("```"):
                    content_clean = content_clean[:-3]
                content_clean = content_clean.strip()
                
                return safe_json_loads(content_clean, default={"response": content, "parse_error": True})
            
            return {"response": content}
    
    except httpx.ReadTimeout:
        log_debug(f"LLM call timed out after {timeout}s", level="WARNING")
        raise
    except Exception as e:
        log_debug(f"LLM call error: {e}", level="ERROR")
        raise

# =============================================
# ================ Trend Service ===============
# =============================================
class TrendService:
    """Fetch & cache dynamic seasonal trends."""
    _cache: Dict[str, Any] = {}
    _lock = asyncio.Lock()
    _bg_task: Optional[asyncio.Task] = None
    _cache_file_lock = asyncio.Lock()

    @classmethod
    async def start(cls):
        loaded = await cls._load_cache_from_disk()
        if not loaded:
            await cls.refresh()
        cls._bg_task = asyncio.create_task(cls._periodic())

    @classmethod
    async def _periodic(cls):
        while True:
            await asyncio.sleep(Config.TRENDS_REFRESH_MIN * 60)
            try:
                if not await cls._is_cache_fresh():
                    await cls.refresh()
                else:
                    log_debug("Trend cache still fresh; skipping refresh ✅", level="INFO")
            except Exception as e:
                log_debug(f"Trend refresh error: {e}", level="ERROR")

    @classmethod
    async def _is_cache_fresh(cls) -> bool:
        async with cls._lock:
            data = cls._cache
        if not data:
            return False
        try:
            gen_at = data.get("generated_at")
            if not gen_at:
                return False
            gen_ts = datetime.fromisoformat(gen_at.replace("Z", "+00:00")) if "T" in gen_at else datetime.strptime(gen_at, "%Y-%m-%d")
            age = datetime.now(gen_ts.tzinfo) - gen_ts if gen_ts.tzinfo else datetime.now() - gen_ts
            return age.total_seconds() < (Config.TREND_CACHE_TTL_HOURS * 3600)
        except Exception:
            return False
    
    @classmethod
    async def get(cls) -> Dict[str, Any]:
        async with cls._lock:
            if cls._cache:
                return dict(cls._cache)
        return cls._get_default()

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
    async def _load_cache_from_disk(cls) -> bool:
        async with cls._cache_file_lock:
            data = _read_json_file(Config.TREND_CACHE_FILE)
        if not data:
            return False
        try:
            if "upcoming_festivals" in data:
                for f in data["upcoming_festivals"]:
                    if isinstance(f.get("date"), str):
                        try:
                            f["date"] = datetime.strptime(f["date"], "%Y-%m-%d")
                        except Exception:
                            pass
        except Exception:
            pass

        async with cls._lock:
            cls._cache = data

        return await cls._is_cache_fresh()

    @classmethod
    async def _save_cache_to_disk(cls) -> None:
        async with cls._lock:
            data = dict(cls._cache)

        try:
            if "upcoming_festivals" in data:
                safe = []
                for f in data["upcoming_festivals"]:
                    safe.append({
                        "name": f.get("name"),
                        "date": (
                            f["date"].strftime("%Y-%m-%d")
                            if isinstance(f.get("date"), datetime) else f.get("date")
                        ),
                        "hint": f.get("hint"),
                    })
                data["upcoming_festivals"] = safe
        except Exception:
            pass

        async with cls._cache_file_lock:
            _write_json_file(Config.TREND_CACHE_FILE, data)

    @classmethod
    async def refresh(cls) -> None:
        with timer("TrendService.refresh"):
            now = datetime.now()
            lookahead = now + timedelta(days=Config.TRENDS_LOOKAHEAD_DAYS)

            holidays_task = asyncio.create_task(cls._fetch_holidays_web(now, lookahead))
            weather_task = asyncio.create_task(cls._fetch_weather(Config.DEFAULT_LAT, Config.DEFAULT_LON))
            india_trends_task = asyncio.create_task(cls._fetch_trends_india(days=7))
            global_trends_task = asyncio.create_task(cls._fetch_trends_global(days=7))
            
            results = await asyncio.gather(
                holidays_task, weather_task, india_trends_task, global_trends_task, 
                return_exceptions=True
            )
            
            holidays = results[0] if not isinstance(results[0], Exception) else []
            weather_summary = results[1] if not isinstance(results[1], Exception) else "Weather unavailable"
            india_trends = results[2] if not isinstance(results[2], Exception) else ""
            global_trends = results[3] if not isinstance(results[3], Exception) else ""
            
            upcoming = [h for h in holidays if now.date() <= h["date"].date() <= lookahead.date()]
            upcoming_sorted = sorted(upcoming, key=lambda x: x["date"])[:8]

            season = cls._infer_indian_season(now.month)

            system_prompt = (
                "Fashion trend analyzer. Return ONLY JSON: "
                '{"trends": ["4-6 trends"], "season_styling_notes": "brief", '
                '"cultural_highlights": ["2-3 items"]}. Mix ethnic/western/streetwear.'
            )
            
            user_prompt = json.dumps({
                "date": now.strftime("%Y-%m-%d"),
                "season": season,
                "weather": weather_summary,
                "festivals": [{"name": h["name"], "date": h["date"].strftime("%Y-%m-%d")} for h in upcoming_sorted],
                "india_trends": india_trends[:400],
                "global_trends": global_trends[:400],
            }, ensure_ascii=False)
            
            try:
                llm_result = await call_llm(
                    system_prompt, user_prompt, 
                    json_mode=True, timeout=12.0
                )
                trends = llm_result.get("trends") or Config.FALLBACK_TRENDS
                season_notes = llm_result.get("season_styling_notes", "")
                cultural = llm_result.get("cultural_highlights", [])
            except Exception as e:
                log_debug(f"LLM trend gen failed: {e}", level="WARNING")
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
                    "trending_india": india_trends[:300],
                    "trending_global": global_trends[:300],
                    "generated_at": now.isoformat(),
                }
                log_debug(
                    f"Trends refreshed: {len(upcoming_sorted)} festivals, {len(trends)} trends", 
                    level="SUCCESS"
                )

            await cls._save_cache_to_disk()

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
    @debug_api_call("Fetch Holidays")
    async def _fetch_holidays_web(now: datetime, lookahead: datetime) -> List[Dict[str, Any]]:
        days_ahead = (lookahead.date() - now.date()).days
        
        prompt = (
            f"List 8-10 major festivals/holidays in India over next {days_ahead} days from {now.strftime('%Y-%m-%d')}. "
            f"For each: name, exact date (YYYY-MM-DD), brief fashion tip. Simple bullets."
        )
        
        allowed = ["timesofindia.indiatimes.com", "indiatoday.in", "wikipedia.org", "thehindu.com"]

        try:
            out = await openai_web_search(prompt, allowed_domains=allowed, country="IN", timeout=25.0)
            
            text = out.get("text", "").strip()
            if not text:
                return []
            
            results: List[Dict[str, Any]] = []
            lines = text.split('\n')
            for line in lines:
                date_match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2})|(\d{2}[-/]\d{2}[-/]\d{4})', line)
                if date_match:
                    date_str = date_match.group(0).replace('/', '-')
                    try:
                        if date_str.startswith('20'):
                            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        else:
                            date_obj = datetime.strptime(date_str, '%d-%m-%Y')
                        
                        name_part = line[:date_match.start()].strip(' -•*')
                        if name_part:
                            results.append({
                                "name": name_part[:50],
                                "date": date_obj,
                                "hint": line[date_match.end():].strip(' -:')[:100]
                            })
                    except ValueError:
                        continue
            
            return results[:10]
        except Exception as e:
            log_debug(f"Holidays fetch failed: {e}", level="WARNING")
            return []

    @staticmethod
    @debug_api_call("Fetch Weather")
    async def _fetch_weather(lat: float, lon: float) -> str:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
            "forecast_days": 7,
            "timezone": "auto",
        }
        
        try:
            async with httpx.AsyncClient(timeout=15, trust_env=True) as client:
                r = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
                r.raise_for_status()
                j = r.json()
            
            daily = j.get("daily", {})
            prec_list = daily.get("precipitation_sum", [])
            tmax_list = daily.get("temperature_2m_max", [])
            tmin_list = daily.get("temperature_2m_min", [])

            prec = sum(prec_list) if prec_list else 0
            tmax = max(tmax_list) if tmax_list else 25
            tmin = min(tmin_list) if tmin_list else 15

            return f"Next 7d: {tmin:.0f}-{tmax:.0f}°C, ~{prec:.0f}mm precip"
        except Exception as e:
            log_debug(f"Weather fetch failed: {e}", level="WARNING")
            return "Weather data unavailable"

    @staticmethod
    async def _fetch_trends_india(days: int = 7) -> str:
        end = datetime.now()
        start = end - timedelta(days=days)
        window = f"{start.strftime('%b %d')}-{end.strftime('%b %d, %Y')}"
        
        prompt = (
            f"List 6 concise fashion trends in INDIA ({window}). "
            f"Mix: ethnic, western, streetwear, athleisure, formal, footwear, accessories. "
            f"One sentence each."
        )
        
        try:
            out = await openai_web_search(
                prompt, 
                allowed_domains=INDIA_FASHION_DOMAINS, 
                country="IN",
                timeout=25.0
            )
            return out.get("text", "").strip()
        except Exception as e:
            log_debug(f"India trends fetch failed: {e}", level="WARNING")
            return ""

    @staticmethod
    async def _fetch_trends_global(days: int = 7) -> str:
        end = datetime.now()
        start = end - timedelta(days=days)
        window = f"{start.strftime('%b %d')}-{end.strftime('%b %d, %Y')}"
        
        prompt = (
            f"List 6 concise GLOBAL fashion trends ({window}). "
            f"Cover runway, high-street, streetwear, footwear, accessories. "
            f"Mention brands. One sentence each."
        )
        
        try:
            out = await openai_web_search(
                prompt, 
                allowed_domains=GLOBAL_FASHION_DOMAINS,
                timeout=25.0
            )
            return out.get("text", "").strip()
        except Exception as e:
            log_debug(f"Global trends fetch failed: {e}", level="WARNING")
            return ""

# =============================================
# ================ User Profile ===============
# =============================================
class UserProfile:
    """Structured user profile management."""
    
    @staticmethod
    def parse_from_memories(memories: List[str]) -> Dict[str, Any]:
        """Extract structured profile from memories."""
        profile = {
            "name": None,
            "gender": None,
            "preferences": [],
            "past_queries": [],
            "occasions": [],
            "style_tags": [],
        }
        
        for mem in memories:
            if not mem:
                continue
            
            mem_lower = mem.lower()
            
            if "name is" in mem_lower:
                m = re.search(r"name is (\w+)", mem, re.IGNORECASE)
                if m:
                    profile["name"] = m.group(1).capitalize()
            
            if "identify as" in mem_lower or "i am" in mem_lower or "i'm" in mem_lower:
                if any(w in mem_lower for w in ["male", "man", "guy", "boy", "he", "him"]):
                    profile["gender"] = "male"
                elif any(w in mem_lower for w in ["female", "woman", "girl", "she", "her", "lady"]):
                    profile["gender"] = "female"
            
            gender_clues = {
                "male": ["kurta", "sherwani", "bandhgala", "men's", "men ", "mens"],
                "female": ["saree", "lehenga", "salwar", "women's", "women ", "womens", "anarkali"],
            }
            for gender, keywords in gender_clues.items():
                if any(kw in mem_lower for kw in keywords) and profile["gender"] is None:
                    profile["gender"] = gender
            
            if any(w in mem_lower for w in ["love", "prefer", "like", "favorite", "into"]):
                profile["preferences"].append(mem[:100])
            
            if any(w in mem_lower for w in ["wedding", "event", "party", "office", "casual", "formal", "date"]):
                profile["occasions"].append(mem[:80])
            
            style_keywords = ["traditional", "western", "fusion", "ethnic", "contemporary", "modern", "classic"]
            for style in style_keywords:
                if style in mem_lower and style not in profile["style_tags"]:
                    profile["style_tags"].append(style)
        
        return profile


# =============================================
# ========== NEW: Intent Classifier ===========
# =============================================

@debug_tool
async def classify_search_intent(
    user_query: str,
    user_gender: Optional[str] = None,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    🧠 SMART INTENT CLASSIFIER
    
    Determines:
    1. search_type: "specific" | "discovery" | "pairing"
    2. search_queries: List of 1-4 queries to use
    3. confidence: how confident we are
    
    Examples:
    - "blue cotton shirt" → specific (1 query)
    - "what to pair with jeans" → pairing (3-4 queries)
    - "outfit for date night" → discovery (3-4 queries)
    - "recommend something" → discovery (4 queries)
    """
    
    system_prompt = """You are a fashion search intent classifier. Analyze the user query and return ONLY JSON.

OUTPUT FORMAT:
{
  "search_type": "specific" | "discovery" | "pairing",
  "reasoning": "brief explanation",
  "search_queries": ["query1", "query2", ...],
  "product_category": "topwear" | "bottomwear" | "footwear" | "accessories" | "fullset" | "mixed"
}

RULES:
1. search_type = "specific" when:
   - User asks for a SPECIFIC item with clear attributes
   - Examples: "blue cotton shirt", "black leather jacket", "white sneakers"
   - Return 1 search query (the user query itself, maybe refined)

2. search_type = "pairing" when:
   - User wants to match/pair/complement an existing item
   - Examples: "what to pair with jeans", "shoes for formal pants", "top to go with skirt"
   - Return 3-4 COMPLEMENTARY queries based on what they need
   - If pairing with bottomwear → suggest topwear
   - If pairing with topwear → suggest bottomwear/footwear

3. search_type = "discovery" when:
   - User wants recommendations, suggestions, or broad exploration
   - Examples: "outfit for date", "summer collection", "trending clothes", "recommend something"
   - Return 3-4 DIVERSE queries covering different styles/colors/types

PAIRING LOGIC:
- "pair with jeans/trousers/pants" → suggest topwear: shirts, t-shirts, jackets, blazers
- "pair with shirt/top" → suggest bottomwear: trousers, jeans, skirts
- "pair with dress/kurta" → suggest accessories/footwear: shoes, bags, jewelry
- "complete the outfit" → suggest missing pieces

DISCOVERY LOGIC:
- For occasions: provide diverse style options (casual, formal, trendy)
- For seasons: provide weather-appropriate variations
- For trends: provide current + classic options
- For broad requests: provide color/style/type variations

Use gender context when available to make queries more relevant."""

    user_prompt = json.dumps({
        "query": user_query,
        "gender": user_gender or "unknown",
        "context": context or ""
    }, ensure_ascii=False)
    
    try:
        result = await call_llm(
            system_prompt,
            user_prompt,
            json_mode=True,
            timeout=8.0  # Fast classification
        )
        
        search_type = result.get("search_type", "specific")
        queries = result.get("search_queries", [user_query])
        
        # Ensure we have the right number of queries
        if search_type == "specific" and len(queries) > 1:
            queries = queries[:1]
        elif search_type in ["discovery", "pairing"] and len(queries) < 3:
            # Pad with variations if needed
            queries = queries + [user_query] * (3 - len(queries))
        
        return {
            "search_type": search_type,
            "reasoning": result.get("reasoning", ""),
            "search_queries": queries[:4],  # Max 4
            "product_category": result.get("product_category", "mixed"),
        }
    
    except Exception as e:
        log_debug(f"Intent classification failed: {e}", level="WARNING")
        # Safe fallback: treat as specific
        return {
            "search_type": "specific",
            "reasoning": "Classification failed, defaulting to specific search",
            "search_queries": [user_query],
            "product_category": "mixed",
        }


# =============================================
# ======== NEW: Unified Smart Search ==========
# =============================================

@debug_tool
async def execute_vector_search(
    queries: List[str],
    limit_per_query: int = 50,
    rerank_top_k: int = 12
) -> List[Dict[str, Any]]:
    """
    Execute vector search with multiple queries and rerank.
    
    Args:
        queries: List of search queries
        limit_per_query: Products to fetch per query
        rerank_top_k: Final number of products to return
    
    Returns:
        List of reranked products
    """
    await Services.ensure_loaded()
    
    # Embed all queries in parallel
    vectors = await Services._embed_catalog(queries)
    
    # Search Qdrant with all queries in parallel
    from qdrant_client.http import models as rest
    
    async def _search_single(query_text: str, vec: List[float]):
        def _do_search():
            return Services._qdr.query_points(
                collection_name=Config.CATALOG_COLLECTION,
                query=vec,
                limit=limit_per_query,
                with_payload=True,
                search_params=rest.SearchParams(hnsw_ef=Config.HNSW_EF),
            )
        return await asyncio.to_thread(_do_search)
    
    search_tasks = [_search_single(q, v) for q, v in zip(queries, vectors)]
    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
    
    # Collect products
    all_products = []
    seen_product_ids = set()
    
    # For multi-query: take top 3 from each, for single query: take all
    products_per_result = 3 if len(queries) > 1 else limit_per_query
    
    for i, (query_text, result) in enumerate(zip(queries, search_results)):
        if isinstance(result, Exception):
            log_debug(f"Query {i} failed: {result}", level="WARNING")
            continue
        
        for point in (result.points or [])[:products_per_result]:
            payload = point.payload or {}
            product_id = payload.get("product_id")
            
            if product_id in seen_product_ids:
                continue
            seen_product_ids.add(product_id)
            
            commerce = payload.get("commerce", {})
            all_products.append({
                "product_id": product_id,
                "title": payload.get("title"),
                "brand": payload.get("brand"),
                "category": payload.get("category_leaf"),
                "price_inr": commerce.get("price"),
                "in_stock": commerce.get("in_stock"),
                "colors_available": commerce.get("colors_in_stock", []),
                "sizes_available": commerce.get("sizes_in_stock", []),
                "description": payload.get("description", "")[:150],
                "score": float(point.score),
                "from_query": query_text,
            })
    
    # Rerank if multiple products
    if len(all_products) > 1:
        try:
            # Use first query for reranking (most important)
            rerank_query = queries[0]
            rerank_texts = [f"{p['title']} {p['brand']} {p['category']}" for p in all_products]
            indices = await Services._rerank_qwen(
                rerank_query,
                rerank_texts,
                top_k=min(rerank_top_k, len(all_products))
            )
            all_products = [all_products[i] for i in indices if i < len(all_products)]
        except Exception as e:
            log_debug(f"Rerank failed: {e}", level="WARNING")
            all_products = all_products[:rerank_top_k]
    
    return all_products


# =============================================
# ================== Tools ====================
# =============================================

@p.tool
@debug_tool
async def quick_greeting_check_local(context: p.ToolContext) -> p.ToolResult:
    """Lightning-fast greeting data fetch."""
    await Services.ensure_loaded()

    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or "guest"

    trends_task = TrendService.get()
    
    async def _mem_search():
        return await asyncio.to_thread(
            Services._mem.search,
            "user profile: name, gender, preferences",
            user_id=user_id,
            limit=Config.MEMORY_SEARCH_LIMIT,
        )
    
    mem_task = _race(_mem_search(), timeout=Config.MEMORY_TIMEOUT, fallback=None)
    
    trends_obj, mem_results = await asyncio.gather(trends_task, mem_task, return_exceptions=True)
    
    if isinstance(trends_obj, Exception):
        trends_obj = TrendService._get_default()
    if isinstance(mem_results, Exception):
        mem_results = None

    memories = []
    if mem_results:
        memories = [r.get("text") or r.get("memory") for r in mem_results.get("results", []) if r]
    
    profile = UserProfile.parse_from_memories(memories)

    return p.ToolResult(
        data={
            "user_name": profile["name"],
            "gender": profile["gender"] or "unknown",
            "preferences": profile["preferences"][:3],
            "style_tags": profile["style_tags"],
            "trends": trends_obj.get("trends", [])[:3],
            "season": trends_obj.get("season"),
            "upcoming_festivals": [f["name"] for f in trends_obj.get("upcoming_festivals", [])][:2],
            "is_returning": profile["name"] is not None,
        }
    )


@p.tool
@debug_tool
async def save_user_profile_local(
    context: p.ToolContext,
    name: Optional[str] = None,
    gender: Optional[str] = None,
    preference: Optional[str] = None,
) -> p.ToolResult:
    """Save user profile info (name, gender, preferences)."""
    await Services.ensure_loaded()
    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or "guest"
    
    saved_items = []
    
    if name:
        Services._mem.add(
            messages=[{"role": "user", "content": f"My name is {name}"}],
            user_id=user_id,
            metadata={"type": "name"},
            infer=False,
        )
        saved_items.append("name")
    
    if gender and gender.lower() in ["male", "female", "other"]:
        Services._mem.add(
            messages=[{"role": "user", "content": f"I identify as {gender}"}],
            user_id=user_id,
            metadata={"type": "gender"},
            infer=False,
        )
        saved_items.append("gender")
    
    if preference:
        Services._mem.add(
            messages=[{"role": "user", "content": preference}],
            user_id=user_id,
            metadata={"type": "preference"},
            infer=True,
        )
        saved_items.append("preference")
    
    return p.ToolResult(data={
        "status": "saved",
        "saved_items": saved_items,
        "name": name,
        "gender": gender
    })


@p.tool
@debug_tool
async def smart_fashion_search_local(
    context: p.ToolContext,
    user_query: str,
) -> p.ToolResult:
    """
    🎯 INTELLIGENT FASHION SEARCH
    
    Automatically determines search strategy:
    - Specific queries → Single focused search (fast, 15 results)
    - Discovery/Pairing → Multi-query search (diverse, 12 results)
    
    Uses 24h cache for speed.
    """
    await Services.ensure_loaded()
    
    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or "guest"
    
    # Check cache first (24h TTL)
    cache_key = json.dumps({
        "query": user_query,
        "user_id": user_id
    }, sort_keys=True, ensure_ascii=False)
    
    cached_result = await get_cached_search(cache_key)
    if cached_result:
        log_debug(f"✅ Cache HIT for '{user_query}'", level="SUCCESS")
        return p.ToolResult(data=cached_result)
    
    log_debug(f"🔍 Cache MISS for '{user_query}' - computing...", level="INFO")
    
    # Get user profile for context
    async def _get_profile():
        results = await asyncio.to_thread(
            Services._mem.search,
            "user gender and preferences",
            user_id=user_id,
            limit=5,
        )
        memories = [r.get("text") or r.get("memory") for r in results.get("results", []) if r]
        return UserProfile.parse_from_memories(memories)
    
    profile = await _race(_get_profile(), timeout=0.8, fallback={})
    gender = profile.get("gender")
    
    # Classify intent (smart routing)
    intent_result = await classify_search_intent(
        user_query,
        user_gender=gender,
        context=f"User preferences: {profile.get('preferences', [])[:2]}"
    )
    
    search_type = intent_result["search_type"]
    search_queries = intent_result["search_queries"]
    
    log_debug(
        f"🧠 Intent: {search_type}",
        level="INFO",
        queries=search_queries,
        reasoning=intent_result["reasoning"]
    )
    
    # Execute appropriate search
    if search_type == "specific":
        # Fast single-query search
        products = await execute_vector_search(
            queries=search_queries[:1],
            limit_per_query=Config.SIMPLE_SEARCH_LIMIT,
            rerank_top_k=Config.SIMPLE_SEARCH_LIMIT
        )
    else:
        # Multi-query discovery/pairing
        products = await execute_vector_search(
            queries=search_queries,
            limit_per_query=Config.PRODUCTS_PER_QUERY,
            rerank_top_k=Config.FINAL_RERANK_TOP_K
        )
    
    # Save query to memory
    try:
        Services._mem.add(
            messages=[{"role": "user", "content": f"Searched for: {user_query}"}],
            user_id=user_id,
            metadata={"type": "query", "timestamp": datetime.now().isoformat()},
            infer=False,
        )
    except Exception:
        pass
    
    result_data = {
        "original_query": user_query,
        "search_type": search_type,
        "search_queries": search_queries,
        "total_products_found": len(products),
        "products": products,
        "user_gender": gender or "unknown",
        "intent_reasoning": intent_result["reasoning"],
    }
    
    # Cache for 24h
    await save_cached_search(cache_key, result_data)
    
    return p.ToolResult(data=result_data)


@p.tool
@debug_tool
async def generate_conversational_response_local(
    context: p.ToolContext,
    search_results_json: str,
) -> p.ToolResult:
    """Generate witty, emoji-rich, conversational product presentation."""
    try:
        search_data = safe_json_loads(search_results_json)
    except Exception as e:
        return p.ToolResult(data={
            "response": "Oops! 😅 Something went wrong parsing the results. Let me try that again!",
            "error": str(e)
        })
    
    products = search_data.get("products", [])
    user_gender = search_data.get("user_gender", "unknown")
    original_query = search_data.get("original_query", "your search")
    search_type = search_data.get("search_type", "specific")
    
    if not products:
        return p.ToolResult(data={
            "response": (
                "Hmm, couldn't find exactly what you're looking for 🤔\n\n"
                "Mind rephrasing? Or I can show you what's trending right now! ✨"
            ),
            "has_results": False,
        })
    
    # Build gender-aware context
    gender_context = ""
    if user_gender == "male":
        gender_context = "Focus on men's styling. Use masculine language."
    elif user_gender == "female":
        gender_context = "Focus on women's styling. Use feminine language."
    
    search_context = ""
    if search_type == "pairing":
        search_context = "These are PAIRING suggestions - emphasize how they complement the user's existing item."
    elif search_type == "discovery":
        search_context = "These are DISCOVERY results - highlight variety and help them explore options."
    
    system_prompt = (
        f"You're MuseBot 🎨 - witty, friendly fashion AI. {gender_context} {search_context}\n"
        f"Return ONLY JSON:\n"
        f'{{"opening": "catchy 1-liner with emoji", '
        f'"products": [{{"index": 0, "hook": "punchy 1-line sell", "why": "why perfect"}}, ...], '
        f'"closing": "engaging question with emoji"}}\n'
        f"Show 3-4 products max. Be conversational, use emojis naturally. Keep it SHORT!"
    )
    
    # Prepare product summaries (top 5)
    products_summary = []
    for i, prod in enumerate(products[:5]):
        products_summary.append({
            "index": i,
            "title": prod.get("title", ""),
            "brand": prod.get("brand", ""),
            "price": prod.get("price_inr"),
            "category": prod.get("category", ""),
            "from_query": prod.get("from_query", ""),
        })
    
    user_prompt = json.dumps({
        "query": original_query,
        "gender": user_gender,
        "search_type": search_type,
        "products": products_summary,
        "total_available": len(products),
    }, ensure_ascii=False)
    
    try:
        result = await call_llm(
            system_prompt,
            user_prompt,
            json_mode=True,
            timeout=12.0
        )
        
        opening = result.get("opening", "Here's what I found! ✨")
        product_details = result.get("products", [])
        closing = result.get("closing", "What do you think? 😊")
        
        response_parts = [opening, ""]
        
        for pd in product_details:
            idx = pd.get("index", 0)
            if idx >= len(products):
                continue
            
            prod = products[idx]
            hook = pd.get("hook", prod.get("title", ""))
            why = pd.get("why", "")
            price = prod.get("price_inr")
            brand = prod.get("brand", "")
            
            response_parts.append(f"**{hook}** by {brand}")
            if price:
                response_parts.append(f"💰 ₹{price:,.0f}")
            if why:
                response_parts.append(f"_{why}_")
            response_parts.append("")
        
        if len(products) > len(product_details):
            others_count = len(products) - len(product_details)
            response_parts.append(f"_...plus {others_count} more options! Want to see them?_ 👀")
            response_parts.append("")
        
        response_parts.append(closing)
        
        return p.ToolResult(data={
            "response": "\n".join(response_parts),
            "has_results": True,
            "products_shown": len(product_details),
            "products_available": len(products),
        })
        
    except Exception as e:
        log_debug(f"Conversation gen failed: {e}", level="WARNING")
        
        response = f"Found {len(products)} awesome options! 🎉✨\n\n"
        for i, prod in enumerate(products[:3]):
            response += f"{i+1}. **{prod.get('title', 'Product')}** by {prod.get('brand', 'Brand')}"
            if price := prod.get('price_inr'):
                response += f" - ₹{price:,.0f} 💸"
            response += "\n"
        
        response += f"\n_({len(products)} total options!)_\n\n"
        response += f"Which vibe are you feeling? 😎"
        
        return p.ToolResult(data={
            "response": response,
            "has_results": True,
            "error": str(e),
        })


@p.tool
@debug_tool
async def get_user_profile_local(context: p.ToolContext) -> p.ToolResult:
    """Get structured user profile."""
    await Services.ensure_loaded()
    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or "guest"

    def _search():
        return Services._mem.search(
            "user profile: name, gender, preferences, queries",
            user_id=user_id,
            limit=15,
        )

    try:
        results = await asyncio.to_thread(_search)
    except Exception as e:
        log_debug(f"Mem search error: {e}", level="WARNING")
        results = {"results": []}

    memories = [r.get("text") or r.get("memory") for r in results.get("results", []) if r]
    profile = UserProfile.parse_from_memories(memories)
    
    return p.ToolResult(data={
        "user_id": user_id,
        "profile": profile,
    })


# =============================================
# ================ Retriever ==================
# =============================================
async def fashion_memory_retriever(
    context: p.RetrieverContext
) -> p.RetrieverResult:
    """Retrieve user memories."""
    await Services.ensure_loaded()
    message = context.interaction.last_customer_message
    if not message or not message.content:
        return p.RetrieverResult(None)

    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or "guest"

    results = Services._mem.search(
        message.content,
        user_id=user_id,
        limit=Config.MEMORY_SEARCH_LIMIT,
    )
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

        # ================= First Interaction: Warm Welcome =================
        await agent.create_guideline(
            condition=(
                "The conversation has exactly ONE message from the customer "
                "AND zero messages from the agent"
            ),
            action=(
                "This is the FIRST interaction! 🎉\n\n"
                "1. Call quick_greeting_check_local\n"
                "2. Based on results:\n"
                "   - If is_returning=true and user_name exists:\n"
                "     * Warm greeting: 'Hey [name]! 👋 Welcome back!'\n"
                "     * Reference a past preference if any\n"
                "     * Mention 1 trend with emoji\n"
                "   - If is_returning=false (NEW USER):\n"
                "     * Friendly intro: 'Hey there! 👋 I'm MuseBot, your fashion buddy 🎨'\n"
                "     * Mention 2 trends with emojis\n"
                "     * Casually ask: 'What's your name?' 😊\n"
                "3. If gender=unknown, ALSO ask: 'Also, are you looking for men's or women's fashion?' 🤔\n"
                "4. Then naturally address their original message\n\n"
                "Keep it SHORT (3-4 sentences), friendly, emoji-rich! 🌟"
            ),
            tools=[quick_greeting_check_local, save_user_profile_local],
        )

        # ================= Profile Building =================
        await agent.create_guideline(
            condition=(
                "User shares their name, gender, or says things like "
                "'I'm John', 'my name is...', 'I'm a guy', 'I'm female', "
                "'looking for men's fashion', 'women's clothing', etc."
            ),
            action=(
                "Extract and save profile info! 🎯\n"
                "1. Call save_user_profile_local with extracted name and/or gender\n"
                "2. Respond warmly:\n"
                "   - If name: 'Awesome, [Name]! 🌟' or 'Nice to meet you, [Name]! 😊'\n"
                "   - If gender: 'Got it! Let me find you something perfect' 👌\n"
                "3. Then continue helping with their fashion query\n\n"
                "Be conversational, use emojis, acknowledge what they shared!"
            ),
            tools=[save_user_profile_local],
        )

        # ================= Fashion Search: Smart Adaptive =================
        await agent.create_guideline(
            condition="User asks about products, outfits, or fashion recommendations",
            action=(
                "🎯 SMART SEARCH ENGINE\n\n"
                "1. Call smart_fashion_search_local with their query\n"
                "   - It automatically determines intent:\n"
                "     * Specific query (e.g., 'blue cotton shirt') → Single fast search\n"
                "     * Discovery (e.g., 'outfit for date') → Multi-query diverse search\n"
                "     * Pairing (e.g., 'pair with jeans') → Smart complementary search\n"
                "   - Uses 24h cache for speed ⚡\n"
                "   - Returns search_type and reasoning for transparency\n"
                "2. Call generate_conversational_response_local with results\n"
                "3. Present results:\n"
                "   - Show 3-4 products with details\n"
                "   - Use emojis naturally 🎨💰✨\n"
                "   - If search_type='pairing', emphasize complementary nature\n"
                "   - If search_type='discovery', highlight variety\n"
                "   - Mention total available\n"
                "   - Ask engaging follow-up\n"
                "4. If user shared a preference, call save_user_profile_local\n\n"
                "Be specific, helpful, FUN! Trust the smart search engine."
            ),
            tools=[
                smart_fashion_search_local,
                generate_conversational_response_local,
                save_user_profile_local,
                get_user_profile_local,
            ],
        )

        # ================= No Results =================
        await agent.create_guideline(
            condition="Search returns no products or very few results (< 3)",
            action=(
                "Be empathetic and helpful! 💙\n\n"
                "1. Acknowledge: 'Hmm, couldn't find that exact thing 🤔'\n"
                "2. Suggest:\n"
                "   - Try smart_fashion_search_local with broader query\n"
                "   - Offer trending alternatives with emojis\n"
                "   - Ask clarifying questions\n"
                "3. Stay positive: 'But here's what's trending...' ✨\n\n"
                "Keep conversation flowing, be encouraging!"
            ),
            tools=[smart_fashion_search_local, generate_conversational_response_local],
        )

        # ================= Follow-up & Feedback =================
        await agent.create_guideline(
            condition=(
                "User asks for more details, different options, or gives feedback "
                "('show me more', 'what else', 'not quite', 'perfect!', 'love it', etc.)"
            ),
            action=(
                "Engage naturally! 💬✨\n\n"
                "1. If they want more: call smart_fashion_search_local with refined query\n"
                "2. If positive feedback: celebrate! 🎉 'Yay! Glad you love it!'\n"
                "3. If negative: ask what to change: 'Different color? Style? Price range?' 🤔\n"
                "4. Save preferences if mentioned\n\n"
                "Keep it conversational, use emojis, be helpful!"
            ),
            tools=[
                smart_fashion_search_local,
                generate_conversational_response_local,
                save_user_profile_local,
            ],
        )

        # ================= Off-Topic =================
        await agent.create_guideline(
            condition="User asks something unrelated to fashion",
            action=(
                "Politely redirect with personality! 😊\n\n"
                "Respond: 'Haha, I'm all about fashion! 👗👔✨ "
                "But I can help you find something perfect to wear. "
                "What's the occasion or vibe you're going for?' 🎨\n\n"
                "Keep it light, friendly, guide back to fashion."
            ),
            tools=[],
        )

        log_debug(f"🎨 {Config.AGENT_NAME} ready!", level="SUCCESS")
        log_debug(f"🧠 Smart Intent Classification: specific | discovery | pairing", level="INFO")
        log_debug(f"⚡ 24h Search Cache enabled", level="INFO")
        log_debug(f"🔍 Adaptive search: 1 query (specific) or 4 queries (discovery/pairing)", level="INFO")
        log_debug(f"💬 Conversational: witty, emoji-rich, friendly", level="INFO")
        log_debug(f"🌐 Server: http://localhost:8800/chat/", level="INFO")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        perf_tracker.print_summary()