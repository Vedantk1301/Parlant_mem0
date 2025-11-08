# 🎨 MuseBot - AI Fashion Discovery Assistant

A smart, context-aware fashion chatbot powered by Parlant SDK, with semantic search (Qdrant + Qwen embeddings), long-term memory (Mem0), and Indian seasonal/cultural awareness.

## ✨ Key Features

- **Smart Query Understanding**: Corrects spelling, extracts entities (colors, sizes, price, occasion)
- **Rich Product Filtering**: Leverage all Qdrant metadata (colors, sizes, price ranges, categories, brands)
- **Reranking**: Uses Qwen3-Reranker-0.6B for precision
- **Cultural Context**: Aware of Indian seasons, festivals (Diwali, Holi, etc.)
- **Long-Term Memory**: Remembers user preferences via Mem0
- **Personality**: Witty, playful, fashion-focused—redirects off-topic queries gracefully

---

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-...
DEEPINFRA_TOKEN=...
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_KEY=...

# Optional (defaults shown)
CATALOG_COLLECTION=fashion_qwen4b_text
MEM_COLLECTION=mem0_fashion_qdrant
EMB_MODEL_CATALOG=Qwen/Qwen3-Embedding-4B
RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B
DEEPINFRA_TIMEOUT=90
```

### 2. Install Dependencies

```bash
pip install parlant-sdk mem0ai qdrant-client httpx python-dotenv numpy
```

### 3. Run the Agent

```bash
python main.py
```

---

## 🔧 Architecture

### Services (Lazy-Loaded)
- **Mem0**: Long-term memory (OpenAI embeddings + Qdrant storage)
- **Qdrant**: Vector search for product catalog
- **DeepInfra**: Qwen embeddings & reranking

### Tools

| Tool | Purpose |
|------|---------|
| `get_current_context` | Returns season, date, Indian festivals |
| `analyze_fashion_query` | NLU: intent detection, entity extraction, spelling correction |
| `search_catalog` | Qdrant search with filters (colors, sizes, price, brand, category) |
| `save_user_preference` | Stores preferences in Mem0 |
| `get_user_profile` | Retrieves user's fashion profile |
| `save_user_name` | Captures user name for personalization |

### Memory Retriever
Automatically fetches relevant past preferences during conversations.

---

## 💡 Usage Examples

### Example 1: First-Time User
```
User: Hey!
Bot: Hey! I'm MuseBot, your fashion buddy! What should I call you?
User: Call me Arjun
Bot: Nice to meet you, Arjun! 🎉 What are you shopping for today?
```

### Example 2: Product Search with Filters
```
User: Show me black polos under 2000
Bot: [Calls analyze_fashion_query → extracts: colors=["black"], products=["polo"], price_range={"max": 2000}]
    [Calls search_catalog with filters]
    
    Here's what I found! 🖤
    
    1. 🏐 Ivory Knit Polo by Bonkerscorner
       Perfect for casual Fridays! Breathable knit, timeless style.
       ₹1,499 (12% off) | Colors: Off White | Sizes: XS-XXL
    
    2. ...
    
    Which vibe speaks to you?
```

### Example 3: Outfit Advice
```
User: What should I wear for a date night?
Bot: [Calls get_current_context → "Winter, Diwali season"]
    
    Ooh, date night! 💫 Is it indoor (cozy restaurant) or outdoor? 
    And what's your go-to color palette?
```

### Example 4: Preference Learning
```
User: I love navy and slim fits
Bot: [Calls save_user_preference]
    
    Navy + slim fit = timeless! 👌 Noted. 
    Are you more polo or oxford shirt?
```

### Example 5: Off-Topic Redirect
```
User: What's the weather in Mumbai?
Bot: I'm stitched for style, not weather reports! 😄 
    Tell me—what occasion are you dressing for?
```

---

## 🎯 Advanced: Qdrant Filtering

The `search_catalog` tool supports rich filters based on your product payload:

```python
filters = {
    "colors_in_stock": ["black", "navy"],       # OR condition
    "sizes_in_stock": ["M", "L", "XL"],         # OR condition
    "price_range": {"min": 1000, "max": 3000},  # Range
    "category_leaf": ["polo shirts"],           # Exact match
    "brand": ["bonkerscorner"],                 # OR condition
    "in_stock": True                            # Boolean
}
```

### Example Payload Structure
```json
{
  "product_id": "...",
  "title": "ivory knit polo t-shirt",
  "brand": "bonkerscorner",
  "category_leaf": "slim fit polo shirts",
  "commerce": {
    "price": 1499,
    "discount_pct": 11.77,
    "in_stock": true,
    "colors_in_stock": ["off white"],
    "sizes_in_stock": ["XS", "S", "M", "L", "XL", "XXL"]
  }
}
```

---

## 🧪 Testing Guidelines

### Test Scenarios
1. **Spelling Errors**: "shwo me blak jens under 200"
2. **Off-Topic**: "Book me a flight to Goa"
3. **Preferences**: "I prefer large sizes and bright colors"
4. **Seasonal Context**: "What's good for Diwali?"
5. **Empty Results**: "Neon green kurta with polka dots"

### Expected Behavior
- Corrects spelling via OpenAI
- Redirects off-topic with humor
- Saves preferences to Mem0
- Provides festival-aware suggestions
- Gracefully handles no results

---

## 🛠️ Customization

### Adjust Personality
Edit `Config.AGENT_VIBE` in `main.py`:
```python
AGENT_VIBE = "professional, sophisticated"  # vs. "playful, witty"
```

### Change Search Defaults
```python
DEFAULT_TOP_K = 20      # Fetch more candidates
RERANK_TOP_K = 15       # Show more results
HNSW_EF = 1000          # Higher accuracy (slower)
```

### Add More Festival Logic
Edit `get_current_context` tool:
```python
if month == 4 and day > 10:
    festivals.append("Eid")
```

---

## 🐛 Troubleshooting

### Issue: "No products found"
- Check Qdrant collection name matches `CATALOG_COLLECTION`
- Verify embeddings are generated with same model (Qwen3-Embedding-4B)
- Test direct Qdrant query outside agent

### Issue: Reranking fails
- Reranker will fallback to vector search order
- Check `DEEPINFRA_TOKEN` is valid
- Increase `DEEPINFRA_TIMEOUT` if network is slow

### Issue: Memory not persisting
- Ensure Mem0 collection exists in Qdrant
- Check `OPENAI_API_KEY` for embedding generation
- Verify user_id is consistent (not "guest" on every session)

---

## 📊 Performance Tips

1. **Lazy Loading**: Services load only on first tool call (instant startup)
2. **Batch Embeddings**: Use `batch_embed_catalog` for bulk indexing
3. **Filter Early**: Apply Qdrant filters to reduce reranking load
4. **Cache Context**: Call `get_current_context` once per session

---

## 🔮 Future Enhancements

- [ ] Image search (CLIP embeddings)
- [ ] Wishlist/cart management tools
- [ ] Size recommendation ML model
- [ ] Multi-language support (Hindi, Tamil, etc.)
- [ ] Virtual try-on integration

---

## 📝 License

MIT License - Feel free to adapt for your fashion platform!

---

## 🙏 Credits

- **Parlant SDK**: Agentic framework
- **Qwen Models**: Alibaba's embedding & reranking models
- **Mem0**: Long-term memory system
- **Qdrant**: Vector database