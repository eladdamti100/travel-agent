import json
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from src.graph.workflow import graph as travel_graph

app = FastAPI(title="Marco Travel Agent API")

# ── CORS Middleware (CRITICAL FOR REACT INTEGRATION) ──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (e.g., localhost:5173)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# ── Request Models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ResumeRequest(BaseModel):
    action: str       
    feedback: str = ""

# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_db_connection():
    db_path = Path("data/checkpoints.db")
    return sqlite3.connect(str(db_path))

def _last_ai_content(state) -> str:
    messages = state.values.get("messages", []) if state else []
    if not messages:
        return ""
    last = messages[-1]
    return last.content if hasattr(last, "content") else str(last)

# ── POST /chat ────────────────────────────────────────────────────────────────
@app.post("/chat")
def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.session_id}}
    input_state = {"messages": [("user", request.message)]}

    try:
        travel_graph.invoke(input_state, config=config)
        state = travel_graph.get_state(config)
        return {
            "status": "success",
            "reply": _last_ai_content(state),
            "hitl": bool(state and state.next),
        }
    except GraphInterrupt:
        state = travel_graph.get_state(config)
        return {
            "status": "success",
            "reply": _last_ai_content(state),
            "hitl": True,
        }
    except Exception as e:
        return {"status": "error", "reply": f"Graph Execution Error: {str(e)}"}

# ── GET /chat/stream/{session_id} ─────────────────────────────────────────────
@app.get("/chat/stream/{session_id}")
def chat_stream(session_id: str, message: str):
    config = {"configurable": {"thread_id": session_id}}
    input_state = {"messages": [("user", message)]}

    def event_generator():
        try:
            for chunk in travel_graph.stream(
                input_state,
                config=config,
                stream_mode="updates",
            ):
                for node_name in chunk.keys():
                    payload = json.dumps({"type": "node", "node": node_name})
                    yield f"data: {payload}\n\n"

            state = travel_graph.get_state(config)
            hitl = bool(state and state.next)
            reply = _last_ai_content(state)
            yield f"data: {json.dumps({'type': 'done', 'reply': reply, 'hitl': hitl})}\n\n"

        except GraphInterrupt:
            state = travel_graph.get_state(config)
            reply = _last_ai_content(state)
            yield f"data: {json.dumps({'type': 'done', 'reply': reply, 'hitl': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'reply': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── POST /session/{session_id}/resume ─────────────────────────────────────────
@app.post("/session/{session_id}/resume")
def resume_session(session_id: str, request: ResumeRequest):
    config = {"configurable": {"thread_id": session_id}}
    resume_value = {"decision": request.action, "feedback": request.feedback}

    try:
        travel_graph.invoke(Command(resume=resume_value), config=config)
        state = travel_graph.get_state(config)
        reply = _last_ai_content(state) or "Plan updated."
        return {"status": "success", "reply": reply}

    except GraphInterrupt:
        state = travel_graph.get_state(config)
        return {"status": "success", "reply": _last_ai_content(state), "hitl": True}

    except Exception as e:
        return {"status": "error", "reply": str(e)}

# ── GET /session/{session_id}/state ──────────────────────────────────────────
@app.get("/session/{session_id}/state")
def get_state(session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    state = travel_graph.get_state(config)

    if not state:
        return {"values": {}, "next": []}

    values = dict(state.values)
    critique = values.get("critique_result", {})
    if critique:
        values["critic_results"] = {
            "score":       critique.get("score"),
            "issues":      critique.get("issues", []),
            "suggestions": critique.get("suggestions", []),
        }

    values.setdefault("cache_status", "Cache Miss")
    values.setdefault("cache_ttl",    "Live Feed")

    return {"values": values, "next": list(state.next) if state.next else []}

# ── GET /sessions ─────────────────────────────────────────────────────────────
@app.get("/sessions")
def get_sessions():
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        sessions = [row[0] for row in cursor.fetchall()]
        conn.close()
        return {"sessions": sessions}
    except Exception:
        return {"sessions": []}

# ── DELETE /session/{session_id} ──────────────────────────────────────────────
@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
        conn.commit()
        conn.close()
        return {"status": "success", "message": f"Session {session_id} successfully purged from database."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Purge Failed: {str(e)}")

# ── DELETE /sessions ──────────────────────────────────────────────────────────
@app.delete("/sessions")
def clear_all_sessions():
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints")
        conn.commit()
        conn.close()
        return {"status": "success", "message": "All operational thread checkpoints successfully wiped clean."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Global Purge Failed: {str(e)}")