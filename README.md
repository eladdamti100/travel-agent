# Marco — AI Travel Planner

> Enterprise-grade AI travel planner powered by LangGraph, Groq/Gemini, and a local SQLite knowledge base.

Marco accepts natural-language trip requests, validates them through a multi-agent security pipeline, and returns a structured travel plan covering flights, hotels, activities, visa requirements, currency conversion, and cost estimates.

---

## Features

- **Natural-language input** — "Plan a 7-day trip from TLV to Paris for $3000"
- **Multi-agent pipeline** — orchestrator → validator → planner → critic → human approval
- **Real-time web enrichment** — live currency, events, geocoding, Tavily research
- **HITL (Human-in-the-Loop)** — the agent asks for missing trip details, then resumes; users can approve, edit, or cancel the final plan
- **Semantic caching** — similar requests return cached plans (cosine similarity, MiniLM-L6-v2)
- **Dual LLM providers** — Groq (fast, cheap) or Gemini (capable), switchable via `.env`
- **Session memory** — preferences (airline, diet, style) persist across conversations
- **Token tracking** — input/output tokens and estimated cost logged per LLM call

---

## Supported Destinations

**Paris · London · Tokyo · New York · Berlin**

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure

Create `.env` in the project root:

```env
# Required — pick one provider
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key

# or:
# LLM_PROVIDER=gemini
# GOOGLE_API_KEY=your_google_key

# Optional — web enrichment tools (all have offline fallbacks)
OPENCAGE_API_KEY=your_key
TICKETMASTER_API_KEY=your_key
EXCHANGERATE_API_KEY=your_key
TAVILY_API_KEY=your_key
```

### 3. Run

```bash
python run.py
```

The app validates your `.env` on startup and prints a clear error if any required key is missing.

---

## Example Session

```
Marco: Welcome! I'm Marco, your AI travel planner.
You: Plan a 7-day trip to Paris with a budget of $3000

Marco: I'll need a few more details to plan your trip:
  → What's your departure airport? (e.g. TLV, JFK, LHR)
  → What country is your passport from?

You: From TLV, Israeli passport

Marco: [Runs planner — fetches flights, hotels, activities, visa, currency...]

─── Section 1 — Database Data ────────────────────────────────
Trip Summary
- Origin: TLV (Tel Aviv)
- Destination: Paris, France
- Duration: 7 days | Budget: $3,000.00 USD

Flights
- El Al: LY315, $450
- Air France: AF1502, $520

Hotels
- Hotel Louvre Rivoli: $120.00/night (4 stars)
...

─── Section 2 — Live Web Data ────────────────────────────────
Currency Conversion (live)
- Budget: 3000.00 USD = 2760.00 EUR
...

─── Section 3 — Notes and Assumptions ───────────────────────
Your budget of $3,000 comfortably covers this 7-day Paris trip...

Critic score: 8/10 ✓  Plan is within budget.

[A] Approve  [E] Edit  [C] Cancel  →  a

Marco: Plan approved! Saving to cache.
```

---

## Architecture

```
run.py
  └─ src/main.py              REPL + HITL approval prompt
       └─ src/graph/
            ├─ workflow.py    StateGraph + SqliteSaver (12 nodes)
            ├─ nodes.py       node functions
            ├─ router.py      conditional edge functions
            └─ state.py       AgentState TypedDict
```

### Graph Flow

```
START → extract_metadata → validator
          │ blocked        → END
          │ HITL resume    → resume_hitl_context → master_planner
          └─ approved      → master_orchestrator
                               ├─ preferences_memory → summarizer → END
                               ├─ researcher → END
                               └─ cache_check
                                    ├─ hit  → END
                                    └─ miss → master_planner
                                                  └─ critic
                                                       ├─ fail (<2x) → master_planner (auto-replan)
                                                       └─ pass       → hitl_approval
                                                                           ├─ approved  → cache_store → summarizer → END
                                                                           ├─ edit      → master_planner (with feedback)
                                                                           └─ cancelled → END
```

### Key Modules

| Layer | Module | Purpose |
|---|---|---|
| **Config** | `src/config/settings.py` | All env vars & thresholds (Pydantic BaseSettings) |
| **Config** | `src/config/city_registry.py` | City/airport/currency maps — single source of truth |
| **Agents** | `src/agents/planner.py` | Master planner orchestration (466 lines) |
| **Agents** | `src/agents/critic.py` | Deterministic budget + completeness gate |
| **Agents** | `src/agents/hitl_feedback_parser.py` | Free-text edit → TripContext overrides |
| **Services** | `src/services/plan_formatter.py` | Deterministic Section 1 & 2 builders (no LLM) |
| **Services** | `src/services/plan_generator.py` | Section 3 LLM call (notes & assumptions) |
| **Services** | `src/services/plan_enricher.py` | Web fallback + cost calculation |
| **Utils** | `src/utils/token_tracker.py` | Token + cost logging per LLM call |

---

## Configuration

All settings live in `src/config/settings.py` (Pydantic `BaseSettings`):

| Setting | Env Var | Default |
|---|---|---|
| LLM provider | `LLM_PROVIDER` | `gemini` |
| LLM model override | `LLM_MODEL` | provider default |
| Cache similarity threshold | `CACHE_HIT_THRESHOLD` | `0.85` |
| SLM enrichment timeout (s) | `ENRICHMENT_TIMEOUT_SECONDS` | `10.0` |
| Max auto-replan attempts | `MAX_CRITIC_ATTEMPTS` | `2` |
| Max user edit cycles | `MAX_HITL_EDIT_ATTEMPTS` | `3` |

---

## Testing

```bash
# Full suite (354 passing)
pytest

# Fully offline web API tests — zero HTTP calls, zero API cost
pytest tests/test_web_api_mocked.py -v

# P0 regression tests
pytest tests/test_p0_fixes.py -v
```

---

## Admin Mode

Append `ADMIN00` to the session ID prompt to enable the async reviewer:

```
Session ID: mysessionADMIN00
```

The reviewer agent critiques the approved plan and appends its analysis to the terminal output.

---

## Adding a New Destination

1. Add entries to `src/config/city_registry.py` (`CITY_KEYWORDS`, `AIRPORT_BY_CITY`, `COUNTRY_BY_CITY`, `CURRENCY_BY_CITY`).
2. Add SQLite rows in `src/utils/db_init.py` for flights, hotels, activities, visa, weather.
3. No other files need changing.

---

## Project Status

| Phase | Focus | Status |
|---|---|---|
| P0 — Blocking fixes | `asyncio` safety, stale state, HITL loop cap, test suite | ✅ Done |
| P1 — High priority | City registry, settings module, planner decomposition, token tracking | ✅ Done |
| P2 — Important | Dead code removal, typed state, logging standards, mock tests | ✅ Done |
| P3 — Nice to have | Health check, env configs, fuzzing tests, LLM guards, FastAPI | 🔄 In progress |

---

## License

MIT
