# Marco — AI Travel Planner

> Enterprise-grade AI travel planner powered by LangGraph, Groq/Gemini, and a local SQLite knowledge base.

Marco accepts natural-language trip requests, validates them through a multi-agent security pipeline, and returns a structured travel plan covering flights, hotels, activities, visa requirements, currency conversion, and cost estimates.

---

## Features

- **Natural-language input** — "Plan a 7-day trip from TLV to Paris for $3000"
- **Hierarchical multi-agent pipeline** — 7 sub-agents in two tiers run concurrently inside a `WebSupervisor` boundary, combining fast deterministic DB results with live web intelligence in a single parallel wave
- **Zero-Trust security layer** — `WebSupervisor` + `CyberAgent` enforce a 6-step pipeline on every dispatch: Lakera Guard v2 outbound injection check, regex sanitization, Google Safe Browsing inbound URL scan, **Microsoft Presidio PII redaction** (fully local, no API key), and malicious-content regex scan
- **Tier 1 DB agents** — `TransportAgent`, `StayAgent`, `ExperienceAgent` serve SQLite-backed flight, hotel, activity, weather, and event data deterministically
- **Tier 2 Hierarchical Web Agents** — `TransportWebAgent`, `StayWebAgent`, `ExperienceWebAgent`, `ManagerWebAgent` run autonomous LLM ReAct loops for live pricing, events, geocoding, and currency data
- **Smart Replanner** — detects parameter changes and re-runs only the affected agents (e.g. changing `origin_airport` re-runs flights and `transport_live_research`, while reusing hotels and activities from cache)
- **Semantic caching** — 384-dim cosine similarity (MiniLM-L6-v2, threshold ≥ 0.85) serves repeated requests instantly, with background `ThreadPoolExecutor` writes that never block the user
- **HITL (Human-in-the-Loop)** — the agent asks for missing trip details before planning, then resumes; users can approve, edit, or cancel the final plan
- **Deterministic plan critic** — budget + completeness gate with configurable auto-replan cap
- **PDF export** — completed plans exported to `data/exports/` via `export_plan_to_pdf`
- **Dual LLM providers** — Groq (fast, cheap) or Gemini (capable), switchable via `.env`
- **Session memory** — airline, dietary, and travel preferences persist across conversations
- **Token tracking** — input/output tokens and estimated cost logged per LLM call
- **LangSmith tracing** — optional observability via `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true`

---

## Supported Destinations

**Paris · London · Tokyo · New York · Berlin**

Adding a new destination requires entries in `src/config/city_registry.py` and SQLite rows in `src/utils/db_init.py`. No other files need changing.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Presidio PII redaction (local, no API key)

```bash
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_sm
```

> For best accuracy you can use `en_core_web_lg` instead of `en_core_web_sm`.
> Marco runs without Presidio installed — PII redaction is silently skipped until the spaCy model is present.

### 3. Configure

Create `.env` in the project root:

```env
# Required — pick one LLM provider
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key

# or:
# LLM_PROVIDER=gemini
# GOOGLE_API_KEY=your_google_key

# Recommended — web enrichment tools (all have offline fallbacks)
TAVILY_API_KEY=your_key          # Tier 2 web research (TransportWebAgent, StayWebAgent, etc.)
OPENCAGE_API_KEY=your_key        # Geocoding (static city coordinates fallback)
TICKETMASTER_API_KEY=your_key    # Live events (static mock list fallback)
EXCHANGERATE_API_KEY=your_key    # Live currency rates (static baseline fallback)

# Optional — security hardening (regex / fail-open fallbacks are active without these)
# LAKERA_API_KEY=your_key              # Lakera Guard v2 outbound prompt-injection detection
# GOOGLE_SAFE_BROWSING_KEY=your_key   # Inbound URL safety scan

# Optional — observability
# LANGSMITH_API_KEY=your_key
# LANGSMITH_TRACING=true
```

### 4. Run

```bash
python run.py
```

The app validates your `.env` on startup and prints a clear error if any required key is missing. Use `python run.py --check` to run the health check without starting the REPL.

---

## Example Session

```
Marco: Welcome! I'm Marco, your AI travel planner.
You: Plan a 7-day trip to Paris with a budget of $3000

Marco: I'll need a few more details to plan your trip:
  → What's your departure airport? (e.g. TLV, JFK, LHR)
  → What country is your passport from?

You: From TLV, Israeli passport

Marco: [Runs planner — dispatches all 7 agents in parallel ...]

─── Section 1 — Database Data ────────────────────────────────
Trip Summary
- Origin: TLV (Tel Aviv) → Paris, France
- Duration: 7 days | Budget: $3,000.00 USD

Flights
- El Al: LY315, $450  •  Air France: AF1502, $520

Hotels
- Hotel Louvre Rivoli: $120.00/night (4 stars)  •  ...

Activities, Restaurants, Weather, Visa ...

─── Section 2 — Live Web Data ────────────────────────────────
Currency (live): 3000.00 USD = 2760.00 EUR
Upcoming events: Jazz Festival at Parc Floral (June 14–15) ...
Top hotels by reviews: ...

─── Section 3 — Notes and Assumptions ───────────────────────
Your $3,000 budget comfortably covers this 7-day Paris trip ...

Critic score: 8/10 ✓  Plan is within budget.

[A] Approve  [E] Edit  [C] Cancel  →  a

Marco: Plan approved! Saving to cache in the background.
```

---

## Admin Mode

Append `ADMIN00` to the session ID prompt to enable the async reviewer, which critiques the approved plan and appends its analysis to the terminal:

```
Session ID: mysessionADMIN00
```

---

## Testing

The system has passed a **77/77 Regression & Integration Audit**, confirming that all legacy mechanisms (HITL resume, Replanner, Semantic Cache) integrate correctly with the Epic 3 Hierarchical Web Agents and Zero-Trust security pipeline.

```bash
# Run the full test suite
pytest

# Audit suite — the four core legacy + Epic 3 integration checks
pytest tests/test_validator.py \
       tests/test_hitl_resume.py \
       tests/test_cache_store.py \
       tests/test_cache_store_check.py \
       tests/test_web_supervisor.py \
       tests/test_planner_dependency_graph.py -v

# Fully offline web API tests (zero HTTP calls, zero API cost)
pytest tests/test_web_api_mocked.py -v

# Semantic cache integration tests
pytest tests/test_semantic_cache.py tests/test_semantic_cache_mission2.py -v

# P0 regression tests
pytest tests/test_p0_fixes.py -v
```

### Audit Coverage Summary

| Test File | What It Confirms |
|---|---|
| `test_validator.py` | 3-stage validation (regex → vector → LLM) catches Blocked Scope and Prompt Injection without crashing |
| `test_hitl_resume.py` | HITL state recovery: `resume_hitl_context` merges clarification replies into `pending_trip_context` and clears the pause flag |
| `test_cache_store_check.py` | `DEFAULT_HIT_THRESHOLD = 0.85` (inclusive); `cache_store._executor` is a `ThreadPoolExecutor` singleton; `run_cache_store` returns immediately (fire-and-forget) |
| `test_web_supervisor.py` | All 7 sub-agents dispatched via `asyncio.gather`; outbound context sanitized; inbound results pass CyberAgent inspection |
| `test_planner_dependency_graph.py` | `diff_changed_tasks`: `origin_airport` change invalidates `fetch_flights` **and** `transport_live_research` (Tier 2), while `fetch_hotels` is preserved; dual-format task registry resolves both string and enum keys |

---

## Architecture Overview

```
User message
  │
  ▼
3-stage Validator (regex → vector → Groq LLM)
  │
  ▼
Master Orchestrator → cache_check / preferences_memory / researcher
  │ cache miss
  ▼
Master Planner (wave scheduler)
  │
  ▼
WebSupervisor ─── CyberAgent (6-step Zero-Trust pipeline)
  │
  ├─ asyncio.gather ────────────────────────────────────────┐
  │                                                          │
  │  Tier 1 DB Agents              Tier 2 Web Agents        │
  │  ┌─────────────────┐           ┌───────────────────┐    │
  │  │ TransportAgent  │           │ TransportWebAgent │    │
  │  │ StayAgent       │           │ StayWebAgent      │    │
  │  │ ExperienceAgent │           │ ExperienceWebAgent│    │
  │  └─────────────────┘           │ ManagerWebAgent   │    │
  │                                └───────────────────┘    │
  └──────────────────────────────────────────────────────────┘
  │ merged results
  ▼
Critic (deterministic budget gate)
  │
  ▼
HITL Approval → cache_store (background ThreadPoolExecutor) → summarizer
```

For full Mermaid diagrams, dependency DAG, and wave scheduling tables, see [GRAPHS.md](GRAPHS.md).
For developer conventions, exact invariants, and security pipeline details, see [CLAUDE.md](CLAUDE.md).

---

## Project Status

| Phase | Focus | Status |
|---|---|---|
| P0 — Blocking fixes | `asyncio` safety, stale state reset, HITL loop cap, initial test suite | Done |
| P1 — High priority | City registry, settings module, planner decomposition, token tracking | Done |
| P2 — Important | Dead code removal, typed state, logging standards, mock tests, semantic cache | Done |
| **Epic 3 — Hierarchical Agents & Security** | `WebSupervisor` + `CyberAgent` Zero-Trust pipeline, 4 Hierarchical Web Agents, **Microsoft Presidio** PII redaction (local), Lakera Guard v2, `diff_changed_tasks` cascade fix, **77/77 Regression Audit passed** | **Done** |
| P4 — Future | FastAPI/HTTP server (`src/api/` stub exists), `AsyncSqliteSaver` migration | Planned |

---

## License

MIT
