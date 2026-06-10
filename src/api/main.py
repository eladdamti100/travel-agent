import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

try:
    from src.graph.workflow import graph as travel_graph
except Exception as _graph_err:
    import logging as _logging
    _logging.getLogger("startup").error("Graph failed to load: %s", _graph_err)
    travel_graph = None

app = FastAPI(title="Marco Travel Agent API")

@app.on_event("startup")
async def _warm_models():
    import threading
    def _load():
        try:
            from src.services.semantic_cache import warm_embedding_model
            warm_embedding_model()
        except Exception:
            pass
    threading.Thread(target=_load, daemon=True).start()

# ── Health check (required by Railway) ───────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "marco-travel-agent"}

# ── CORS Middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    """ מוצא את הודעת ה-AI האמיתית האחרונה כדי שהשאילתה של המשתמש לא תהדהד בחזרה """
    if not state or not hasattr(state, "values"):
        return ""
    vals = state.values
    
    # במידה וזה קאש נקי, נחזיר את תשובת הקאש
    if str(vals.get("cache_status", "")).lower() == "hit" and vals.get("cache_answer"):
        return str(vals.get("cache_answer"))
        
    messages = vals.get("messages", [])
    if not messages: return ""
    
    # חיפוש הפוך על ההודעות כדי למצוא את הודעת ה-AI האמיתית האחרונה
    for m in reversed(messages):
        if "AI" in m.__class__.__name__ or getattr(m, "type", "") == "ai":
            return m.content if hasattr(m, "content") else str(m)
            
    # גיבוי: מחזיר את התוכן של ההודעה האחרונה אם שום דבר אחר לא נמצא
    last = messages[-1]
    return last.content if hasattr(last, "content") else str(last)


def _get_planning_query(state) -> str:
    if not state:
        return ""
    vals = state.values if hasattr(state, "values") else state
    return str(vals.get("planning_query") or "")


def _get_cache_matched_query(state) -> str:
    if not state:
        return ""
    vals = state.values if hasattr(state, "values") else state
    return str(vals.get("cache_matched_query") or "")

# ── POST /chat ────────────────────────────────────────────────────────────────
@app.post("/chat")
def chat(request: ChatRequest):
    import logging
    logger = logging.getLogger(__name__)
    if travel_graph is None:
        raise HTTPException(status_code=503, detail="Graph not loaded — check server logs for startup errors.")

    config = {"configurable": {"thread_id": request.session_id}}
    input_state = {
        "messages": [("user", request.message)],
        "trip_context": None,
        "total_budget": None,
        "critic_attempts": 0,
        "force_replan": False,
        "hitl_feedback": "",
        "hitl_decision": "",
        "planner_task_results": {},
        "over_budget": False,
    }

    logger.info(f"CHAT_REQUEST: session={request.session_id} msg={request.message[:50]}")

    try:
        travel_graph.invoke(input_state, config=config)
        state = travel_graph.get_state(config)
        
        cache_status = state.values.get('cache_status', 'N/A')
        next_nodes = list(state.next) if state.next else []
        
        logger.info(f"CHAT_RESPONSE: cache_status={cache_status} next_nodes={next_nodes}")
        
        return {
            "status": "success",
            "reply": _last_ai_content(state),
            "hitl": bool(state and state.next),
            "query": _get_planning_query(state),
            "cache_matched_query": _get_cache_matched_query(state),
        }
    except GraphInterrupt:
        state = travel_graph.get_state(config)
        logger.info(f"CHAT_INTERRUPT: GraphInterrupt caught")
        return {
            "status": "success",
            "reply": _last_ai_content(state),
            "hitl": True,
            "query": _get_planning_query(state),
            "cache_matched_query": _get_cache_matched_query(state),
        }
    except Exception as e:
        logger.error(f"CHAT_ERROR: {e}", exc_info=True)
        return {"status": "error", "reply": f"Graph Execution Error: {str(e)}"}

# ── GET /chat/stream/{session_id} ─────────────────────────────────────────────
@app.get("/chat/stream/{session_id}")
def chat_stream(session_id: str, message: str):
    config = {"configurable": {"thread_id": session_id}}
    input_state = {
        "messages": [("user", message)],
        "trip_context": None,
        "total_budget": None,
        "critic_attempts": 0,
        "force_replan": False,
        "hitl_feedback": "",
        "hitl_decision": "",
        "planner_task_results": {},
        "over_budget": False,
    }

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
            yield f"data: {json.dumps({'type': 'done', 'reply': reply, 'hitl': hitl, 'query': _get_planning_query(state), 'cache_matched_query': _get_cache_matched_query(state)})}\n\n"

        except GraphInterrupt:
            state = travel_graph.get_state(config)
            reply = _last_ai_content(state)
            yield f"data: {json.dumps({'type': 'done', 'reply': reply, 'hitl': True, 'query': _get_planning_query(state), 'cache_matched_query': _get_cache_matched_query(state)})}\n\n"

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
    import logging
    logger = logging.getLogger(__name__)
    
    config = {"configurable": {"thread_id": session_id}}
    resume_value = {"decision": request.action, "feedback": request.feedback}
    
    logger.info(f"RESUME_SESSION: session={session_id} action={request.action}")

    try:
        logger.info(f"RESUME_SESSION: Invoking graph with resume command")
        travel_graph.invoke(Command(resume=resume_value), config=config)
        state = travel_graph.get_state(config)
        reply = _last_ai_content(state) or "Plan updated."
        
        logger.info(f"RESUME_SESSION: Graph returned, cache_status={state.values.get('cache_status')}, next={list(state.next) if state.next else []}")
        
        return {"status": "success", "reply": reply}

    except GraphInterrupt:
        state = travel_graph.get_state(config)
        logger.info(f"RESUME_SESSION: GraphInterrupt after resume")
        return {"status": "success", "reply": _last_ai_content(state), "hitl": True}

    except Exception as e:
        logger.error(f"RESUME_SESSION: Error {e}", exc_info=True)
        return {"status": "error", "reply": str(e)}

# ── GET /session/{session_id}/state ──────────────────────────────────────────
@app.get("/session/{session_id}/state")
def get_state(session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    state = travel_graph.get_state(config)

    if not state:
        return {"values": {}, "next": []}

    raw_values = dict(state.values)
    is_hit = str(raw_values.get("cache_status", "")).lower() == "hit"

    # 1. ניקוי הודעות: מסננים שאילתות כפולות ומערכים פנימיים
    serialized_messages = []
    for m in raw_values.get("messages", []):
        class_name = m.__class__.__name__
        role = "human" if "Human" in class_name else "ai"
        content = m.content if hasattr(m, "content") else str(m)
        
        # אם התוכן לא מתחיל בסוגריים מסולסלים (כדי לסנן הודעות פנימיות של State)
        if content.strip() and not (content.startswith("{") and content.endswith("}")):
            serialized_messages.append({"type": role, "content": content})
            
    # אם זה קאש היט, כדאי שנוסיף את תשובת הקאש כהודעת עוזר (AI) אם היא לא קיימת שם עדיין
    if is_hit and raw_values.get("cache_answer"):
        cache_ans = str(raw_values["cache_answer"])
        if not any(msg["content"] == cache_ans for msg in serialized_messages):
            serialized_messages.append({"type": "ai", "content": cache_ans})

    raw_values["messages"] = serialized_messages

    # 2. פונקציית עזר לחילוץ נתונים רלוונטיים בלבד
    def _extract_component_data(key: str) -> Any:
        res = raw_values.get("planner_task_results", {}).get(key) or \
              raw_values.get("final_plan", {}).get(key) or \
              raw_values.get("planner_structured_results", {}).get(key)
        
        if res:
            if isinstance(res, str):
                try: return json.loads(res)
                except: return res
            return res
            
        # אם זה קאש וחסר נתונים, ננסה לחלץ את הרכיב הנכון מתוך JSON של cache_answer
        if is_hit:
            ans = raw_values.get("cache_answer", "")
            try:
                js = json.loads(ans)
                if isinstance(js, dict) and key in js:
                    return js[key]
            except: pass
        return None

    # נזריק רק את הנתונים הנקיים!
    raw_values["planner_task_results"] = {
        "fetch_flights": _extract_component_data("fetch_flights"),
        "fetch_hotels": _extract_component_data("fetch_hotels"),
        "fetch_weather": _extract_component_data("fetch_weather")
    }

    # 3. קידוד בטוח
    safe_values = jsonable_encoder(raw_values)

    # 4. ביקורת (Critic)
    critique = safe_values.get("critique_result", {})
    if critique and isinstance(critique, dict):
        safe_values["critic_results"] = {
            "score":       critique.get("score"),
            "issues":      critique.get("issues", []),
            "suggestions": critique.get("suggestions", []),
        }

    safe_values.setdefault("cache_status", "Cache Miss")
    safe_values.setdefault("cache_ttl",    "Live Feed")

    return {"values": safe_values, "next": list(state.next) if state.next else []}

# ── DEBUG: POST /session/{session_id}/debug-chat ────────────────────────────
@app.post("/session/{session_id}/debug-chat")
def debug_chat(session_id: str, request: ChatRequest):
    """Debug endpoint: send message, get detailed state info about cache/hitl flow"""
    config = {"configurable": {"thread_id": session_id}}
    input_state = {"messages": [("user", request.message)]}
    
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"DEBUG_CHAT: Starting with message: {request.message[:100]}")
        
        # Invoke graph
        state = travel_graph.invoke(input_state, config=config)
        
        logger.info(f"DEBUG_CHAT: Graph returned:")
        logger.info(f"  cache_status={state.get('cache_status')}")
        logger.info(f"  planner_status={state.get('planner_status')}")
        logger.info(f"  awaiting_hitl_decision={state.get('awaiting_hitl_decision')}")
        
        # Get state again
        state2 = travel_graph.get_state(config)
        logger.info(f"DEBUG_CHAT: get_state returned:")
        logger.info(f"  next={list(state2.next) if state2.next else []}")
        
        return {
            "status": "success",
            "reply": _last_ai_content(state2),
            "debug": {
                "cache_status": str(state.get("cache_status", "")),
                "planner_status": str(state.get("planner_status", "")),
                "next_nodes": list(state2.next) if state2.next else [],
                "msg_count": len(state.get("messages", [])),
            }
        }
    except Exception as e:
        logger.error(f"DEBUG_CHAT: Error {e}", exc_info=True)
        return {"status": "error", "reply": str(e)}

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
        return {"status": "success", "message": f"Session {session_id} successfully purged."}
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
        return {"status": "success", "message": "All thread checkpoints wiped clean."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Global Purge Failed: {str(e)}")
    
# ── GET /logs/stream — SSE live log tail ─────────────────────────────────────
_LOG_FILE = Path(__file__).parent.parent.parent / "logs" / "travel_agent.log"

@app.get("/logs/stream")
def stream_logs():
    """
    Server-Sent Events endpoint that tails travel_agent.log in real time.
    Each event is a JSON object: { level, name, message, time }
    """
    def _tail():
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FILE.touch(exist_ok=True)

        with open(_LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            # Send last 50 lines as history first
            lines = f.readlines()
            for line in lines[-50:]:
                parsed = _parse_log_line(line)
                if parsed:
                    yield f"data: {json.dumps(parsed)}\n\n"

            # Then tail for new lines
            while True:
                line = f.readline()
                if line:
                    parsed = _parse_log_line(line)
                    if parsed:
                        yield f"data: {json.dumps(parsed)}\n\n"
                else:
                    yield ": keep-alive\n\n"
                    time.sleep(0.3)

    return StreamingResponse(
        _tail(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _parse_log_line(line: str) -> Optional[dict]:
    """Parse a log line into structured JSON. Format: HH:MM:SS | LEVEL | name | message"""
    line = line.rstrip()
    if not line:
        return None
    parts = line.split(" | ", 3)
    if len(parts) == 4:
        return {"time": parts[0], "level": parts[1].strip(), "name": parts[2].strip(), "message": parts[3]}
    return {"time": "", "level": "INFO", "name": "log", "message": line}


# ── PATCH /session/{session_id} ──────────────────────────────────────────────
@app.patch("/session/{session_id}")
async def update_session_name(session_id: str, data: dict):
    try:
        new_name = data.get("name")
        # כאן אתה צריך להוסיף לוגיקה ששומרת את השם החדש ב-DB אם אתה רוצה שהוא יישמר לצמיתות.
        # אם אין לך טבלה לשמות, אפשר פשוט להחזיר הצלחה:
        return {"status": "success", "message": f"Session {session_id} renamed to {new_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update Failed: {str(e)}")