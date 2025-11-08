import os
import sys
import json
import asyncio
import re
from datetime import datetime
from typing import Optional, Dict, Any, List
import parlant.sdk as p
import httpx
from dotenv import load_dotenv

load_dotenv()

# ========== Configuration ==========
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    DEEPINFRA_TOKEN = os.getenv("DEEPINFRA_TOKEN", "")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_KEY = os.getenv("QDRANT_KEY")
    CATALOG_COLLECTION = os.getenv("CATALOG_COLLECTION", "fashion_qwen4b_text")
    MEM_COLLECTION = os.getenv("MEM_COLLECTION", "mem0_fashion_qdrant")
    
    DEFAULT_TOP_K = int(os.getenv("SEARCH_TOP_K", "12"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))
    HNSW_EF = int(os.getenv("HNSW_EF", "500"))
    
    AGENT_NAME = os.getenv("AGENT_NAME", "MuseBot")
    AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-5-nano")

# ========== Season-Aware Trends (No API needed) ==========
SEASON_TRENDS = {
    "November": ["Layering pieces", "Earth tones", "Cozy knits"],
    "December": ["Party wear", "Velvet textures", "Metallic accents"],
    "January": ["Winter essentials", "Warm layers", "Minimalist style"],
    "February": ["Spring preview", "Pastels", "Light fabrics"],
    "March": ["Floral prints", "Light jackets", "Transitional wear"],
    "April": ["Breathable fabrics", "Bright colors", "Cotton basics"],
    "May": ["Summer prep", "Linen", "Light neutrals"],
    "June": ["Monsoon ready", "Quick-dry fabrics", "Waterproof"],
    "July": ["Monsoon fashion", "Bold prints", "Comfortable fits"],
    "August": ["Festive prep", "Traditional wear", "Ethnic fusion"],
    "September": ["Navratri colors", "Ethnic chic", "Statement pieces"],
    "October": ["Diwali special", "Festive wear", "Traditional glam"],
}

def get_current_trends() -> List[str]:
    """Instant trends based on current month - no API needed"""
    month = datetime.now().strftime("%B")
    return SEASON_TRENDS.get(month, ["Smart casuals", "Versatile basics", "Contemporary style"])

# ========== Windows Compatibility ==========
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

# ========== Lazy Service Loader ==========
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

# ========== LLM Utilities ==========

async def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = None,
    temperature: float = 0.3,
    json_mode: bool = True
) -> Dict[str, Any]:
    """Universal LLM caller with structured output"""
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY required for LLM calls")
    
    model = model or Config.AGENT_MODEL
    
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    
    async with httpx.AsyncClient(timeout=45, trust_env=False) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        
        if json_mode:
            return json.loads(content)
        return {"response": content}

# ========== Fast Greeting Tool ==========

@p.tool
async def quick_greeting_check(context: p.ToolContext) -> p.ToolResult:
    """
    Lightning-fast greeting data fetch (< 1 second)
    Combines user profile + trends without blocking
    """
    await Services.ensure_loaded()
    
    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or getattr(customer, "name", None) or "guest"
    
    # Get trends instantly (no API)
    trends = get_current_trends()
    
    # Quick memory check with timeout
    user_name = None
    preference = None
    
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(
                Services._mem.search,
                "user's name and fashion preferences",
                user_id=user_id,
                limit=5
            ),
            timeout=1.5  # Fail fast if slow
        )
        
        memories = [r.get("text") or r.get("memory") for r in results.get("results", []) if r]
        
        # Extract name quickly
        for mem in memories:
            if mem and "name is" in mem.lower():
                match = re.search(r"name is (\w+)", mem, re.IGNORECASE)
                if match:
                    user_name = match.group(1).capitalize()
                    break
        
        # Get one quick preference
        for mem in memories:
            if mem and any(word in mem.lower() for word in ["love", "prefer", "like", "favorite", "fan of"]):
                preference = mem
                break
                
    except (asyncio.TimeoutError, Exception) as e:
        print(f"[quick_greeting] Timeout/error: {e}")
    
    return p.ToolResult(data={
        "user_name": user_name,
        "preference": preference,
        "trends": trends,
        "is_returning": user_name is not None
    })

# ========== Dynamic Knowledge Tools ==========

@p.tool
async def analyze_catalog_schema(context: p.ToolContext) -> p.ToolResult:
    """
    Dynamically discover what fields exist in the catalog by sampling.
    Returns available filters, categories, brands, price ranges, etc.
    """
    await Services.ensure_loaded()
    
    from qdrant_client.http import models as rest
    
    def _sample():
        return Services._qdr.scroll(
            collection_name=Config.CATALOG_COLLECTION,
            limit=100,
            with_payload=True
        )
    
    results = await asyncio.to_thread(_sample)
    points = results[0] if results else []
    
    # Analyze schema dynamically
    all_colors = set()
    all_sizes = set()
    all_brands = set()
    all_categories = set()
    price_min, price_max = float('inf'), 0
    
    for point in points:
        payload = point.payload or {}
        commerce = payload.get("commerce", {})
        
        all_colors.update(commerce.get("colors_in_stock", []))
        all_sizes.update(commerce.get("sizes_in_stock", []))
        if brand := payload.get("brand"):
            all_brands.add(brand)
        if cat := payload.get("category_leaf"):
            all_categories.add(cat)
        
        if price := commerce.get("price"):
            price_min = min(price_min, price)
            price_max = max(price_max, price)
    
    schema = {
        "available_colors": sorted(list(all_colors)),
        "available_sizes": sorted(list(all_sizes)),
        "available_brands": sorted(list(all_brands)),
        "available_categories": sorted(list(all_categories)),
        "price_range": {"min": int(price_min), "max": int(price_max)},
        "filterable_fields": [
            "commerce.colors_in_stock",
            "commerce.sizes_in_stock",
            "commerce.price",
            "commerce.in_stock",
            "brand",
            "category_leaf"
        ]
    }
    
    return p.ToolResult(data=schema)


@p.tool
async def get_contextual_knowledge(context: p.ToolContext, query: str) -> p.ToolResult:
    """
    Dynamically fetch real-time contextual knowledge:
    - Current date, season, weather implications
    - Cultural events, festivals, holidays
    - Fashion trends, styling advice
    
    Uses LLM reasoning instead of hardcoded rules.
    """
    now = datetime.now()
    
    system_prompt = f"""You are a fashion context analyzer. Given the current date and a user query, 
provide relevant contextual information in JSON format:

{{
  "current_date": "YYYY-MM-DD",
  "season": "...",
  "season_styling_notes": "...",
  "cultural_context": ["list of relevant festivals, holidays, or events"],
  "weather_considerations": "...",
  "trending_styles": ["..."],
  "occasion_insights": "..." (if query mentions an occasion)
}}

Be specific to Indian context for festivals and seasons.
Today is {now.strftime("%B %d, %Y, %A")}.
"""
    
    user_prompt = f"""User query: "{query}"

Provide contextual fashion knowledge relevant to this query and the current date."""
    
    try:
        result = await call_llm(system_prompt, user_prompt)
        return p.ToolResult(data=result)
    except Exception as e:
        # Fallback to basic date info
        return p.ToolResult(data={
            "current_date": now.strftime("%Y-%m-%d"),
            "season": "Unknown",
            "error": str(e)
        })


@p.tool
async def intelligent_query_analyzer(context: p.ToolContext, text: str) -> p.ToolResult:
    """
    Deep query understanding using LLM reasoning.
    No hardcoded patterns - fully dynamic analysis.
    """
    
    system_prompt = """You are an expert fashion query analyzer. Analyze the user's message and return:

{
  "intent": "product_search" | "outfit_advice" | "preference_statement" | "question" | "off_topic",
  "confidence": 0.0-1.0,
  "normalized_query": "spelling-corrected version",
  "extracted_entities": {
    "colors": [],
    "product_types": [],
    "occasions": [],
    "styles": [],
    "fits": [],
    "materials": [],
    "brands": [],
    "sizes": [],
    "price_constraints": {"min": null, "max": null, "budget": null}
  },
  "implied_filters": {
    "must_be_in_stock": true/false,
    "preferred_price_segment": "budget/mid/premium/luxury" or null
  },
  "user_sentiment": "excited/casual/confused/frustrated",
  "suggested_clarifications": ["questions to ask if info is missing"],
  "is_fashion_related": true/false,
  "off_topic_reason": "..." (if not fashion related)
}

Be smart about spelling errors, abbreviations, and Indian English patterns."""
    
    user_prompt = f'Analyze this query: "{text}"'
    
    try:
        result = await call_llm(system_prompt, user_prompt, temperature=0.2)
        return p.ToolResult(data=result)
    except Exception as e:
        # Minimal fallback
        return p.ToolResult(data={
            "intent": "product_search",
            "confidence": 0.5,
            "normalized_query": text,
            "extracted_entities": {},
            "is_fashion_related": True,
            "error": str(e)
        })


@p.tool
async def build_smart_filters(
    context: p.ToolContext,
    query_analysis_json: str,
    catalog_schema_json: str
) -> p.ToolResult:
    """
    Dynamically construct Qdrant filters based on query analysis and catalog schema.
    Uses LLM to map natural language to technical filters.
    
    Args:
        query_analysis_json: JSON string of query analysis
        catalog_schema_json: JSON string of catalog schema
    """
    
    try:
        query_analysis = json.loads(query_analysis_json)
        catalog_schema = json.loads(catalog_schema_json)
    except json.JSONDecodeError as e:
        return p.ToolResult(data={
            "filters": {},
            "reasoning": f"JSON parse error: {e}",
            "search_strategy": "broad"
        })
    
    system_prompt = f"""You are a search filter constructor. Given:
1. Query analysis with extracted entities
2. Catalog schema showing available options

Build optimal Qdrant filters in this format:
{{
  "filters": {{
    "colors_in_stock": ["exact color names from schema"],
    "sizes_in_stock": ["exact size codes from schema"],
    "price_range": {{"min": X, "max": Y}},
    "brand": ["exact brand names from schema"],
    "category_leaf": ["exact category names from schema"],
    "in_stock": true
  }},
  "reasoning": "why these filters were chosen",
  "search_strategy": "broad/narrow/balanced"
}}

Available catalog schema:
{json.dumps(catalog_schema, indent=2)}

Match user intent to available options intelligently. If user says "navy", check if schema has "navy" or "navy blue". 
If budget is mentioned, set price range accordingly.
If no exact match, leave filter empty to get broader results."""
    
    user_prompt = f"""Query analysis:
{json.dumps(query_analysis, indent=2)}

Build the optimal filters."""
    
    try:
        result = await call_llm(system_prompt, user_prompt, temperature=0.1)
        return p.ToolResult(data=result)
    except Exception as e:
        # Return empty filters
        return p.ToolResult(data={
            "filters": {},
            "reasoning": f"Fallback: {e}",
            "search_strategy": "broad"
        })


@p.tool
async def search_catalog(
    context: p.ToolContext,
    query: str,
    filters_json: str = "{}",
    top_k: int = None
) -> p.ToolResult:
    """
    Semantic search with dynamic filtering.
    
    Args:
        query: Search query string
        filters_json: JSON string of filters to apply
        top_k: Number of results to return
    """
    await Services.ensure_loaded()
    top_k = top_k or Config.DEFAULT_TOP_K
    
    # Parse filters
    try:
        filters = json.loads(filters_json) if filters_json else {}
    except json.JSONDecodeError:
        filters = {}
    
    # Generate embedding
    vec = (await Services._embed_catalog([query]))[0]
    
    # Build Qdrant filter
    from qdrant_client.http import models as rest
    
    filter_conditions = []
    
    if filters:
        # Dynamic filter construction
        if colors := filters.get("colors_in_stock"):
            filter_conditions.append(
                rest.FieldCondition(
                    key="commerce.colors_in_stock",
                    match=rest.MatchAny(any=colors)
                )
            )
        
        if sizes := filters.get("sizes_in_stock"):
            filter_conditions.append(
                rest.FieldCondition(
                    key="commerce.sizes_in_stock",
                    match=rest.MatchAny(any=sizes)
                )
            )
        
        if price_range := filters.get("price_range"):
            if price_min := price_range.get("min"):
                filter_conditions.append(
                    rest.FieldCondition(
                        key="commerce.price",
                        range=rest.Range(gte=price_min)
                    )
                )
            if price_max := price_range.get("max"):
                filter_conditions.append(
                    rest.FieldCondition(
                        key="commerce.price",
                        range=rest.Range(lte=price_max)
                    )
                )
        
        if filters.get("in_stock") is True:
            filter_conditions.append(
                rest.FieldCondition(
                    key="commerce.in_stock",
                    match=rest.MatchValue(value=True)
                )
            )
        
        if categories := filters.get("category_leaf"):
            filter_conditions.append(
                rest.FieldCondition(
                    key="category_leaf",
                    match=rest.MatchAny(any=categories)
                )
            )
        
        if brands := filters.get("brand"):
            filter_conditions.append(
                rest.FieldCondition(
                    key="brand",
                    match=rest.MatchAny(any=brands)
                )
            )
    
    qdrant_filter = rest.Filter(must=filter_conditions) if filter_conditions else None
    
    # Search
    def _search():
        return Services._qdr.query_points(
            collection_name=Config.CATALOG_COLLECTION,
            query=vec,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            search_params=rest.SearchParams(hnsw_ef=Config.HNSW_EF)
        )
    
    results = await asyncio.to_thread(_search)
    
    # Parse results
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
            "colors_available": commerce.get("colors_in_stock", []),
            "sizes_available": commerce.get("sizes_in_stock", []),
            "url": payload.get("url"),
            "image": payload.get("primary_image"),
            "score": float(point.score)
        })
    
    # Rerank
    if items and len(items) > 1:
        try:
            rerank_texts = [
                f"{item['title']} by {item['brand']} - {item['category']}"
                for item in items
            ]
            rerank_indices = await Services._rerank_qwen(
                query,
                rerank_texts,
                top_k=min(Config.RERANK_TOP_K, len(items))
            )
            items = [items[i] for i in rerank_indices if i < len(items)]
        except Exception as e:
            print(f"[search_catalog] Rerank error: {e}")
    
    return p.ToolResult(data={
        "query": query,
        "applied_filters": filters,
        "total_found": len(items),
        "items": items[:Config.RERANK_TOP_K]
    })


@p.tool
async def generate_product_presentation(
    context: p.ToolContext,
    products_json: str,
    user_context_json: str,
    query_info_json: str
) -> p.ToolResult:
    """
    Generate personalized, contextual product descriptions.
    No hardcoded templates - fully dynamic.
    
    Args:
        products_json: JSON string of product list
        user_context_json: JSON string of user context
        query_info_json: JSON string of query information
    """
    
    try:
        products = json.loads(products_json)
        user_context = json.loads(user_context_json)
        query_info = json.loads(query_info_json)
    except json.JSONDecodeError as e:
        return p.ToolResult(data={
            "presentation_style": "casual",
            "opening_line": "Here's what I found!",
            "products": [],
            "closing_question": "Which one interests you?",
            "error": f"JSON parse error: {e}"
        })
    
    system_prompt = f"""You are a witty, fashion-savvy stylist. Given products and user context, 
create engaging product presentations.

Return JSON:
{{
  "presentation_style": "playful/sophisticated/casual",
  "opening_line": "...",
  "products": [
    {{
      "product_index": 0,
      "description": "one-liner (10-15 words) on why it's perfect",
      "contextual_note": "tie to season/festival/occasion if relevant"
    }}
  ],
  "closing_question": "one engaging question to continue conversation"
}}

Be conversational, use emojis sparingly, and tie to the user's query sentiment."""
    
    user_prompt = f"""User query info: {json.dumps(query_info, indent=2)}

User context: {json.dumps(user_context, indent=2)}

Products to present: {json.dumps(products[:8], indent=2)}

Create the presentation."""
    
    try:
        result = await call_llm(system_prompt, user_prompt, temperature=0.7)
        return p.ToolResult(data=result)
    except Exception as e:
        # Basic fallback
        return p.ToolResult(data={
            "presentation_style": "casual",
            "opening_line": "Here's what I found!",
            "products": [{"product_index": i, "description": p.get("title", "")} for i, p in enumerate(products[:5])],
            "closing_question": "Which one catches your eye?",
            "error": str(e)
        })


# ========== Memory Tools ==========

@p.tool
async def save_user_preference(context: p.ToolContext, preference: str) -> p.ToolResult:
    """Save user preference to long-term memory"""
    await Services.ensure_loaded()
    
    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or getattr(customer, "name", None) or "guest"
    
    Services._mem.add(
        messages=[{"role": "user", "content": preference}],
        user_id=user_id,
        metadata={"domain": "fashion", "timestamp": datetime.now().isoformat()},
        infer=True
    )
    
    return p.ToolResult(data={"status": "saved", "user_id": user_id})


@p.tool
async def get_user_profile(context: p.ToolContext) -> p.ToolResult:
    """Retrieve user's fashion profile from memory"""
    await Services.ensure_loaded()
    
    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or getattr(customer, "name", None) or "guest"
    
    # Dynamic query based on common fashion preferences
    results = Services._mem.search(
        "What are this user's fashion preferences, sizes, budget, and favorite styles?",
        user_id=user_id,
        limit=10
    )
    
    memories = [
        r.get("text") or r.get("memory")
        for r in results.get("results", [])
        if r
    ]
    
    return p.ToolResult(data={
        "user_id": user_id,
        "preferences": memories,
        "has_profile": len(memories) > 0
    })


@p.tool
async def save_user_name(context: p.ToolContext, name: str) -> p.ToolResult:
    """Save user's name for personalization"""
    await Services.ensure_loaded()
    
    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or "guest"
    
    Services._mem.add(
        messages=[{"role": "user", "content": f"My name is {name}"}],
        user_id=user_id,
        metadata={"domain": "personal", "type": "name"},
        infer=True
    )
    
    return p.ToolResult(data={"status": "saved", "name": name})


# ========== Memory Retriever ==========

async def fashion_memory_retriever(context: p.RetrieverContext) -> p.RetrieverResult:
    """Retrieve relevant memories based on current conversation"""
    await Services.ensure_loaded()
    
    message = context.interaction.last_customer_message
    if not message or not message.content:
        return p.RetrieverResult(None)
    
    customer = getattr(p.Customer, "current", None)
    user_id = getattr(customer, "id", None) or getattr(customer, "name", None) or "guest"
    
    results = Services._mem.search(message.content, user_id=user_id, limit=5)
    memories = [
        r.get("text") or r.get("memory")
        for r in results.get("results", [])
        if r
    ]
    
    return p.RetrieverResult({"memories": memories} if memories else None)


# ========== Main Agent Setup ==========

async def main() -> None:
    if sys.platform.startswith("win"):
        _install_windows_exception_filter()
    
    # PRE-LOAD SERVICES BEFORE STARTING SERVER
    print(f"[{Config.AGENT_NAME}] 🚀 Pre-loading services...", flush=True)
    await Services.ensure_loaded()
    print(f"[{Config.AGENT_NAME}] ✅ Services pre-loaded successfully!", flush=True)
    
    async with p.Server(session_store="local") as server:
        agent = await server.create_agent(
            name=Config.AGENT_NAME,
            description=(
                "An intelligent fashion AI assistant. You understand context deeply, "
                "learn from every interaction, and provide personalized recommendations. "
                "You're witty but helpful, fashion-focused, and culturally aware. "
                "You ALWAYS greet users warmly on their first message."
            )
        )
        
        # Attach memory retriever
        await agent.attach_retriever(fashion_memory_retriever, id="fashion_memory")
        
        # ========== CRITICAL: First Message Greeting (NO TEMPLATE VARIABLES) ==========
        
        await agent.create_guideline(
            condition=(
                "The conversation has exactly ONE message from the customer "
                "AND zero messages from the agent"
            ),
            action=(
                "This is the first interaction! IMMEDIATELY do this:\n"
                "\n"
                "1. Call quick_greeting_check tool\n"
                "2. Based on the result:\n"
                "   - If is_returning is true and user_name exists:\n"
                "     * Greet warmly by their name with excitement\n"
                "     * Mention one trend from the trends array\n"
                "     * Ask what brings them today\n"
                "     * Example: 'Hey Arjun! Welcome back! Currently loving: Layering pieces and Earth tones. What is on your mind today?'\n"
                "\n"
                "   - If is_returning is false (new user):\n"
                "     * Give warm introduction as MuseBot\n"
                "     * Mention two trends from the trends array\n"
                "     * Ask their name casually\n"
                "     * Example: 'Hey there! I am MuseBot—your fashion sidekick! Trending now: Layering pieces and Earth tones. What should I call you?'\n"
                "\n"
                "3. Then naturally address their original message in the next sentence\n"
                "\n"
                "RULES:\n"
                "- Keep greeting to MAX 2 sentences before addressing their message\n"
                "- Be warm but brief\n"
                "- Use trends data from the tool response\n"
                "- If they said Hello or Hi, acknowledge it casually then ask how you can help\n"
                "- If they asked a product question, answer it after the greeting"
            ),
            tools=[quick_greeting_check, save_user_name]
        )
        
        # ========== Name Handling ==========
        
        await agent.create_guideline(
            condition="User shares their name",
            action=(
                "Call save_user_name with the name. "
                "Respond warmly like: 'Nice to meet you NAME!' "
                "Then continue conversation naturally."
            ),
            tools=[save_user_name]
        )
        
        # ========== Fashion Query Handling ==========
        
        await agent.create_guideline(
            condition="User sends a fashion-related query or product search request",
            action=(
                "Follow this flow:\n"
                "1. Call intelligent_query_analyzer to understand the query deeply\n"
                "2. If confidence is low or intent unclear, ask ONE clarifying question\n"
                "3. If intent is off_topic, gracefully redirect with personality\n"
                "4. If intent is preference_statement, call save_user_preference and acknowledge\n"
                "5. For product_search or outfit_advice:\n"
                "   a. Call get_contextual_knowledge for seasonal context\n"
                "   b. Call analyze_catalog_schema to know available options\n"
                "   c. Call build_smart_filters with query_analysis and catalog_schema as JSON strings\n"
                "   d. Call search_catalog with query and filters_json from build_smart_filters\n"
                "   e. If results found: call generate_product_presentation\n"
                "   f. If no results: try again with empty filters for broader search\n"
                "6. Always incorporate user preferences from memory\n"
                "\n"
                "JSON conversion for tools:\n"
                "- build_smart_filters needs JSON string parameters\n"
                "- search_catalog needs filters_json as string\n"
                "- generate_product_presentation needs all parameters as JSON strings"
            ),
            tools=[
                intelligent_query_analyzer,
                get_contextual_knowledge,
                analyze_catalog_schema,
                build_smart_filters,
                search_catalog,
                generate_product_presentation,
                save_user_preference,
                get_user_profile
            ]
        )
        
        # ========== No Results Handling ==========
        
        await agent.create_guideline(
            condition="Search returns no results or very few results",
            action=(
                "Try these strategies:\n"
                "1. Call search_catalog again with empty filters for broader search\n"
                "2. If still no results, call analyze_catalog_schema and suggest alternatives\n"
                "3. Try synonyms: shirt to polo, sneakers to shoes, jeans to denim\n"
                "4. Relax filters: remove color, increase price by 50 percent, search by category\n"
                "\n"
                "Be empathetic: say you could not find exact matches but here are similar options\n"
                "Always provide alternatives, never say nothing found without trying broader search"
            ),
            tools=[analyze_catalog_schema, search_catalog]
        )
        
        # ========== Product Presentation ==========
        
        await agent.create_guideline(
            condition="You have search results and need to present products",
            action=(
                "Use generate_product_presentation for personalized descriptions. "
                "For EACH product include:\n"
                "- Product title and brand\n"
                "- Why it fits their needs tied to query or season or user preference\n"
                "- Price in INR with discount if over 10 percent\n"
                "- Available colors list 2 to 3 if many\n"
                "- Available sizes mention range if many\n"
                "\n"
                "Present 5 to 8 products maximum\n"
                "End with ONE engaging question like:\n"
                "- Which one catches your eye\n"
                "- Want to see more in a specific color\n"
                "- Need styling tips for any of these\n"
                "\n"
                "Keep descriptions concise but contextual"
            ),
            tools=[generate_product_presentation, get_contextual_knowledge]
        )
        
        # ========== Preference Learning ==========
        
        await agent.create_guideline(
            condition="User expresses a preference or style choice",
            action=(
                "Call save_user_preference with the exact statement. "
                "Acknowledge naturally like:\n"
                "- Got it! I will remember you love that\n"
                "- Noted! That preference is saved\n"
                "- Perfect! I will keep that in mind\n"
                "\n"
                "Then continue conversation without making a big deal"
            ),
            tools=[save_user_preference]
        )
        
        # ========== Off-Topic Handling ==========
        
        await agent.create_guideline(
            condition="User asks something unrelated to fashion",
            action=(
                "Gracefully redirect with personality:\n"
                "- Haha I am all about fashion! But speaking of style...\n"
                "- That is outside my expertise! I am your fashion guru though—need style help?\n"
                "- Interesting question! But let me talk style—what is your go-to outfit vibe?\n"
                "\n"
                "Keep it light and friendly"
            ),
            tools=[]
        )
        
        print(f"[{Config.AGENT_NAME}] 🎨 Optimized agent ready!")
        print(f"[{Config.AGENT_NAME}] ⚡ Fast greeting enabled")
        print(f"[{Config.AGENT_NAME}] 💡 Services pre-loaded - ready for instant responses")
        print(f"[{Config.AGENT_NAME}] 🌐 Server running at http://localhost:8800/chat/")


if __name__ == "__main__":
    asyncio.run(main())