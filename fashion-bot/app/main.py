import os, asyncio, httpx
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from .services.parlant_agent import boot_parlant_server

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY","")

PARLANT_BASE = os.getenv("PARLANT_BASE", "http://127.0.0.1:8800")
API_HOST = os.getenv("API_HOST","127.0.0.1")
API_PORT = int(os.getenv("API_PORT","8010"))

class ChatIn(BaseModel):
    customer_id: str = "vedant"
    message: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Boot Parlant
    server, agent_id = await boot_parlant_server()
    app.state.agent_id = agent_id

    # Wait for Parlant HTTP to be ready
    async with httpx.AsyncClient(timeout=3, trust_env=False) as c:
        for _ in range(60):
            try:
                r = await c.get(PARLANT_BASE)
                if r.status_code < 500:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.25)

    yield
    # Shutdown Parlant cleanly
    try:
        await server.__aexit__(None, None, None)
    except Exception:
        pass

app = FastAPI(lifespan=lifespan)

@app.get("/healthz")
async def healthz():
    return {"ok": True, "agent_id": getattr(app.state, "agent_id", None)}

@app.post("/chat")
async def chat(inb: ChatIn):
    agent_id = app.state.agent_id
    async with httpx.AsyncClient(timeout=60, trust_env=False) as c:
        # Reuse session per customer
        sid_key = f"sess:{inb.customer_id}"
        sid = getattr(app.state, sid_key, None)
        if not sid:
            sess = (await c.post(f"{PARLANT_BASE}/sessions",
                                 json={"agent_id": agent_id, "customer_id": inb.customer_id,
                                       "allow_greeting": False})).json()
            sid = sess["id"]
            setattr(app.state, sid_key, sid)

        await c.post(f"{PARLANT_BASE}/sessions/{sid}/events",
                     json={"kind":"message","source":"customer","message":{"content": inb.message}})
        ev = (await c.get(f"{PARLANT_BASE}/sessions/{sid}/events",
                          params={"waitForData": True})).json()
        replies = [e["message"]["content"] for e in ev.get("items", [])
                   if e.get("kind")=="message" and e.get("source")=="ai_agent"]
        return {"agent_id": agent_id, "replies": replies}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=False)
