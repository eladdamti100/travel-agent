# AI Travel Planner — Agentic Travel Planner

An autonomous travel-planning system built with **LangGraph**, **Gemini 2.5 Flash / Groq**, **SQLite**, and a local **semantic cache** powered by `sentence-transformers`.

The system validates every user request, routes it through a Master Orchestrator, remembers user preferences across sessions, performs focused research, checks semantic cache for similar previous plans, and runs a Master Planner after cache miss.

---

## Architecture Overview

```text
User Message
   ↓
extract_metadata
   ↓
validator
   ↓
if blocked → polite rejection → END
if approved → master_orchestrator
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
preferences_memory  researcher  cache_check
        ↓           ↓           ↓
   summarizer      END     cache hit → END
                         cache miss
                              ↓
                        master_planner
                              ↓
        ┌─────────────────────┴─────────────────────┐
        ↓                                           ↓
missing required info                         full plan ready
HITL question → END                           cache_store → summarizer → END
```

---

## Main System Layers

```text
src/agents/
    The brains:
    validators, orchestrator, preferences memory, researcher,
    semantic cache checker/store, context enricher, master planner

src/tools/
    The hands:
    SQL travel tools, cost calculation, optional web search

src/graph/
    The nervous system:
    AgentState, LangGraph nodes, router, workflow

src/models/
    Typed schemas:
    routing, cache, trip context, planner tasks, enrichment results, session validation

src/services/
    Infrastructure services:
    semantic cache storage and embedding similarity

src/utils/
    Engineering helpers:
    DB init, logging, graph guards
```

---

## Current Graph Flow

```text
START
  │
  ▼
extract_metadata
  │
  ▼
validator
  │
  ├─ blocked → END
  │
  ▼
master_orchestrator
  │
  ├─ preferences_memory → summarizer → END
  │
  ├─ researcher         → END
  │
  └─ cache_check
          │
          ├─ cache_hit  → END
          │
          └─ cache_miss → master_planner
                              │
                              ├─ missing_required_info → END
                              │
                              └─ final_plan → cache_store → summarizer → END
```

---

## Master Orchestrator Routes

The Master Orchestrator returns one of three high-level routes:

| Route | Purpose |
|---|---|
| `preferences_memory` | User asks about saved preferences, or states/updates stable travel preferences |
| `research` | Focused factual lookup: flights, hotels, activities, visa, cheapest options |
| `cache_check` | Full trip planning request that should first check semantic cache |

The orchestrator does **not** route directly to `recall` or `update_preferences`. Those are internal behaviors of `preferences_memory`.

---

## Preferences Memory

`preferences_memory` handles default memory conversations:

```text
preferences_memory
   ├─ recall saved preferences
   └─ update saved preferences
```

Examples:

```text
I prefer El Al
I eat kosher
We are 4 travelers
What do you remember about me?
What are my preferences?
```

Persisted fields include:

| Field | Example |
|---|---|
| `preferred_airline` | `El Al` |
| `food_preference` | `kosher` |
| `num_travelers` | `4` |
| `travel_preferences` | `Direct flights only`, `Prefers central hotels` |

---

## Researcher Agent

The Researcher is a focused ReAct-style travel research agent.

It is used for direct lookups, not full trip planning.

Examples:

```text
Show hotels in Paris
What flights are available to Tokyo?
What activities are in London?
Do I need a visa for Japan?
What is the cheapest hotel in Berlin?
```

Available tools include:

| Tool | Purpose |
|---|---|
| `fetch_flights` | Search flights between origin and destination |
| `fetch_hotels` | Search hotels in a city |
| `fetch_activities` | Search tourist activities |
| `get_visa_requirement` | Check visa rules |
| `get_cheapest_flight` | Find cheapest flight |
| `get_cheapest_hotel` | Find cheapest hotel |
| `list_destinations` | List available destinations |
| `calculate_trip_cost` | Calculate flight + hotel cost |
| `web_search` | Optional real-time search via Tavily |

---

## Semantic Cache

Full trip-planning requests go through semantic cache before planning.

```text
cache_check
   ├─ cache_hit  → return cached answer → END
   └─ cache_miss → master_planner
```

### Cache implementation

| Component | Value |
|---|---|
| Embedding library | `sentence-transformers` |
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Storage | `data/semantic_cache.db` |
| Similarity | cosine similarity |
| Current threshold | `0.85` |

The cache stores:

```text
query
normalized_query
answer
route
embedding_json
created_at
```

The semantic cache is separate from both:

```text
data/travel_agency.db   # business travel data
data/checkpoints.db     # LangGraph persistent memory
```

---

## Master Planner

The Master Planner runs after `cache_miss`.

It performs:

```text
1. deterministic TripContext extraction
2. async SLM context enrichment
3. dependency checking
4. async execution of ready tool tasks
5. merge enriched context
6. run newly unlocked tasks
7. ask HITL question if required info is missing
8. generate final plan
9. cache successful final answer
```

### Required fields for full trip planning

A full plan requires:

```text
origin_airport
origin_country
destination_city
duration_days
total_budget
```

If any of these are missing, the planner asks a HITL clarification question.

Example:

```text
User:
Plan me a trip to Paris

Marco:
I can plan this trip, but I need a few details first:
- Which airport are you flying from?
- What is your passport/origin country?
- How many days should the trip be?
- What total budget should I use?
```

### Current HITL behavior

The system currently supports HITL question generation.

Automatic resume from a partial follow-up answer is not fully implemented yet.

For now, after a HITL question, the user should provide a complete clarified request, for example:

```text
I am from Israel, flying from TLV. Plan me a 5-day trip to Paris under $2000
```

Planned next improvement:

```text
pending_trip_context
pending_missing_fields
awaiting_user_clarification
resume planning from user clarification
```

---

## Databases

The project uses three SQLite databases:

| DB | Purpose |
|---|---|
| `data/travel_agency.db` | Travel business data: flights, hotels, activities, visa requirements |
| `data/checkpoints.db` | LangGraph persistent State via `SqliteSaver` |
| `data/semantic_cache.db` | Semantic cache for previous full-trip answers |

These databases are intentionally separate.

---

## Project Structure

```text
travel_agent/
├── .env                          # Optional environment overrides (local keys)
├── .github/                      # CI/workflow configs
│   └── workflows/
│       └── ci.yml                # CI pipeline (tests, lint)
├── .gitignore                    # files to ignore in git
├── .early.coverage/              # optional coverage artifacts from CI
├── data/                         # runtime databases and initial seed
│   ├── checkpoints.db            # LangGraph persistent state
│   ├── checkpoints.db-shm
│   ├── checkpoints.db-wal
│   ├── initial_data.json         # seed data used by `src/utils/db_init.py`
│   ├── semantic_cache.db         # semantic cache (embeddings + answers)
│   └── travel_agency.db          # business data: flights, hotels, activities, visa
├── src/                          # application source code
│   ├── __init__.py
│   ├── main.py                   # CLI/entry point that starts a session
│   ├── agents/                   # agent implementations (orchestrator, planners)
│   │   ├── __init__.py
│   │   ├── ai_validator.py       # Groq-backed policy classifier
│   │   ├── base.py               # shared agent base classes/utilities
│   │   ├── cache_checker.py      # semantic cache lookup logic
│   │   ├── cache_store.py        # write answers into semantic cache
│   │   ├── context_enricher.py   # async SLM enrichment of TripContext
│   │   ├── master_orchestrator.py# routes messages to research/preferences/cache/planner
│   │   ├── planner.py            # master planner: task orchestration + tool execution
│   │   ├── preferences_memory_agent.py # recall/update user preferences
│   │   ├── researcher.py         # focused research agent (tools: flights/hotels)
│   │   ├── reviewer.py           # post-process / safety checks on final answer
│   │   └── validator.py          # rule-based validation fallback
│   ├── graph/                    # LangGraph nodes, router, workflow, state
│   │   ├── __init__.py
│   │   ├── nodes.py              # node implementations used by the graph
│   │   ├── router.py             # functions that decide next node/edge
│   │   ├── state.py              # `AgentState` TypedDict and state helpers
│   │   └── workflow.py           # graph composition (graph object)
│   ├── models/                   # pydantic/typed models used across agents
│   │   ├── cache.py              # semantic cache record model
│   │   ├── context_enrichment.py # enrichment result schemas
│   │   ├── planner.py            # planner task/result models
│   │   ├── preferences.py        # user preferences schema
│   │   ├── routing.py            # orchestrator routing models
│   │   ├── session.py            # session/checkpoint schema
│   │   └── trip_context.py       # TripContext extraction model
│   ├── prompts/                  # prompt templates and examples
│   ├── services/                 # infra services
│   │   └── semantic_cache.py     # embedding, similarity and DB helpers
│   ├── tools/                    # tool bindings used by agents
│   │   ├── __init__.py
│   │   ├── calc_tools.py         # cost calculation helpers
│   │   ├── db_tools.py           # SQL-backed tools: fetch_flights, fetch_hotels...
│   │   └── search_tools.py       # optional web/search tools (Tavily)
│   └── utils/                    # engineering helpers
│       ├── __init__.py
│       ├── db_init.py            # creates and seeds `travel_agency.db`
│       ├── graph_guards.py       # loop protection and max tool-call guards
│       └── logger.py             # structured logging helper
├── tests/                        # unit tests
│   ├── __init__.py
│   ├── test_connection.py        # DB and service connection tests
│   └── test_tools.py             # tests for tool wrappers
├── generate_graph.py             # helper to render the graph into `graph.png`
├── graph.png                     # generated graph image
├── requirements.txt              # pip dependencies
├── run.py                        # small interactive runner for manual sessions
├── smart_travel_agent_graph.svg  # svg graph export
└── travel.sh                     # convenience script (POSIX) to run demos
```

---

## Setup

### 1. Create and activate virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
pip install langgraph-checkpoint-sqlite
```

Important dependencies include:

```text
langchain
langchain-core
langchain-google-genai
langchain-groq
langgraph
langgraph-checkpoint-sqlite
sentence-transformers
numpy
rich
python-dotenv
```

### 3. Configure environment

Create `.env` in project root:

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_key

# Optional / recommended for validator and fast SLM calls
GROQ_API_KEY=your_groq_key

# Optional
TAVILY_API_KEY=your_tavily_key
```

You can switch provider:

```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
```

---

## Database Initialization

### Create travel DB

```powershell
python -m src.utils.db_init
```

This creates:

```text
data/travel_agency.db
```

Expected tables:

```text
flights
hotels
activities
visa_requirements
```

### Verify travel DB

Create `check_db.py` inside `data/` if needed:

```python
import sqlite3

conn = sqlite3.connect("travel_agency.db")
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", [row[0] for row in cur.fetchall()])

for table in ["flights", "hotels", "activities", "visa_requirements"]:
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    print(table, cur.fetchone()[0])

conn.close()
```

Run:

```powershell
cd data
python check_db.py
cd ..
```

Expected example output:

```text
flights 9
hotels 11
activities 10
visa_requirements 10
```

---

## Verify Tools

From project root, create `check_tools.py`:

```python
from src.tools.db_tools import (
    fetch_flights,
    fetch_hotels,
    fetch_activities,
    get_visa_requirement,
)

print("Flights:")
print(fetch_flights.invoke({"origin": "TLV", "destination": "Paris"}))

print("\nHotels:")
print(fetch_hotels.invoke({"city": "Paris"}))

print("\nActivities:")
print(fetch_activities.invoke({"city": "Paris"}))

print("\nVisa:")
print(get_visa_requirement.invoke({
    "origin_country": "Israel",
    "destination_country": "France",
}))
```

Run:

```powershell
python check_tools.py
```

---

## Verify Semantic Cache

Create `check_cache.py`:

```python
from pathlib import Path
import sqlite3

from src.services.semantic_cache import initialize_cache_db

initialize_cache_db()

db = Path("data/semantic_cache.db")
print("semantic_cache.db exists:", db.exists())

conn = sqlite3.connect(db)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", [row[0] for row in cur.fetchall()])

cur.execute("SELECT COUNT(*) FROM semantic_cache")
print("semantic_cache rows:", cur.fetchone()[0])

conn.close()
```

Run:

```powershell
python check_cache.py
```

---

## Verify Graph Compilation

```powershell
python -c "from src.graph.workflow import graph; print('Graph OK:', graph is not None)"
```

Expected:

```text
Graph OK: True
```

---

## Run

```powershell
python run.py
```

Example session:

```text
Enter your session ID: test_01

You:
I am from Israel, flying from TLV. Plan me a 5-day trip to Paris under $1500
```

Expected first run:

```text
validator
→ master_orchestrator
→ cache_check
→ cache miss
→ master_planner
→ cache_store
→ summarizer
```

Then ask a similar query to test cache:

```text
Build a 5 day Paris trip from TLV for an Israeli traveler around 1500 dollars
```

If similarity is above threshold, expected:

```text
cache_check
→ cache hit
→ END
```

---

## Session Memory

Each session ID maps to persistent state in:

```text
data/checkpoints.db
```

Example:

```text
Session: memory_01
You: I prefer El Al and kosher food
You: exit
```

Restart:

```text
Session: memory_01
You: What do you remember about me?
```

Expected:

```text
Preferred airline: El Al
Food preference: kosher
```

---

## AgentState

Current `AgentState` includes:

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    current_city: str
    total_budget: float
    tool_call_count: int

    validation_status: str
    orchestrator_route: str
    orchestrator_reason: str

    preferred_airline: str
    food_preference: str
    num_travelers: int
    travel_preferences: str

    is_admin: bool
    conversation_summary: str

    cache_status: str
    cache_similarity_score: float
    cache_matched_query: str
    cache_answer: str

    trip_context: dict
    context_enrichment_status: str
    planner_status: str
    planner_task_results: dict
```

---

## Security

Every user message passes through validation before orchestration.

### Layer 1 — AI Validator

File:

```text
src/agents/ai_validator.py
```

Uses Groq for semantic policy classification.

### Layer 2 — Rule-Based Validator

File:

```text
src/agents/validator.py
```

Fallback regex/keyword validator.

Possible verdicts:

```text
APPROVED
BLOCKED_HARM
BLOCKED_INJECTION
BLOCKED_SCOPE
BLOCKED_CITY
```

---

## Loop Guards

Runtime loop protection is in:

```text
src/utils/graph_guards.py
```

It detects repeated identical tool calls in the current turn.

The legacy path also uses:

```text
tool_call_count
MAX_TOOL_CALLS = 8
```

---

## Current Known Limitations

1. HITL resume is not fully implemented yet. The planner can ask for missing required fields, but a short follow-up like `2000$` is not yet merged automatically into the pending trip context.
2. Dates are not yet part of `TripContext`. The planner supports duration, but not start/end travel dates.
3. Semantic cache threshold is approximate. Current threshold is `0.85`. Some semantically similar queries may still miss.
4. The legacy `agent` path still exists for backward compatibility, but the main cache-miss route now uses `master_planner`.

---

## Suggested Demo Queries

### Preferences memory

```text
I prefer El Al and kosher food
What do you remember about me?
```

### Research

```text
Show hotels in Paris
What activities are available in London?
Do I need a visa for France if I am from Israel?
```

### HITL

```text
Plan me a trip to Paris
```

Expected: asks for missing origin airport, origin country, duration, and budget.

### Full planning

```text
I am from Israel, flying from TLV. Plan me a 5-day trip to Paris under $1500
```

Expected: full plan, tools run asynchronously, answer stored in semantic cache.

### Cache

```text
Build a 5 day Paris trip from TLV for an Israeli traveler around 1500 dollars
```

Expected: cache hit if similarity is above threshold.

### Security

```text
ignore all previous instructions
```

Expected: blocked.

---

## Course Alignment

This project covers and extends the course requirements:

| Course topic | Implementation |
|---|---|
| Tool binding | `src/tools`, `ALL_TOOLS`, LangChain tools |
| Manual tool understanding | Tools return DB-backed data |
| SQLite tools | `travel_agency.db` |
| LangGraph State | `AgentState` |
| Nodes | validator, orchestrator, preferences memory, researcher, cache, planner |
| Conditional Edges | router functions |
| Loop protection | `graph_guards.py`, circuit breaker |
| Persistence | `SqliteSaver`, `checkpoints.db` |
| Long-term preferences | persisted in `AgentState` |
| Plan-and-Execute | `master_planner` with dependency checks and async execution |
| Semantic cache | local embeddings + SQLite |
