# Marco — AI Travel Planner

> Enterprise-grade AI travel planner powered by LangGraph, Groq/Gemini, and a local SQLite knowledge base.

Marco accepts natural-language trip requests, validates them through a multi-agent security pipeline, and returns a structured travel plan covering flights, hotels, activities, visa requirements, currency conversion, and cost estimates.

---

## Features

- **Natural-language input** — "Plan a 7-day trip from TLV to Paris for $3000"
- **Multi-agent pipeline** — orchestrator → validator → planner → critic → human approval
- **Security layer** — Zero-Trust 6-step pipeline: `WebSupervisor` + `CyberAgent` check outbound fields via Lakera Guard v2 (regex fallback), sanitize them, then scan inbound results via Google Safe Browsing, Microsoft Presidio PII redaction, and malicious-content regex
- **Hierarchical web agent team** — four specialised web agents (`TransportWebAgent`, `StayWebAgent`, `ExperienceWebAgent`, `ManagerWebAgent`) run concurrently inside `WebSupervisor.dispatch()`, replacing the monolithic `WebAgent`
- **Real-time web enrichment** — live currency, events, geocoding, Tavily research
- **HITL (Human-in-the-Loop)** — the agent asks for missing trip details, then resumes; users can approve, edit, or cancel the final plan
- **Semantic caching** — similar requests return cached plans (cosine similarity, MiniLM-L6-v2)
- **PDF export** — completed plans can be exported to PDF via `export_plan_to_pdf`
- **Dual LLM providers** — Groq (fast, cheap) or Gemini (capable), switchable via `.env`
- **Session memory** — preferences (airline, diet, style) persist across conversations
- **Token tracking** — input/output tokens and estimated cost logged per LLM call
- **LangSmith tracing** — optional observability via `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true`

---

## Supported Destinations

**Paris · London · Tokyo · New York · Berlin**

Adding a new destination requires entries in `src/config/city_registry.py` and SQLite rows in `src/utils/db_init.py`. No other files need changing.

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

# Optional — security hardening (regex/fail-open fallbacks active without these)
# LAKERA_API_KEY=your_key          # Lakera Guard v2 prompt-injection detection
# GOOGLE_SAFE_BROWSING_KEY=your_key  # URL safety scan on inbound web results

# Optional — LangSmith observability
# LANGSMITH_API_KEY=your_key
# LANGSMITH_TRACING=true
```

> Microsoft Presidio PII redaction runs fully locally — no API key needed. After installing requirements, run `python -m spacy download en_core_web_sm` (or `en_core_web_lg` for best accuracy).

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

## Admin Mode

Append `ADMIN00` to the session ID prompt to enable the async reviewer, which critiques the approved plan and appends its analysis to the terminal output:

```
Session ID: mysessionADMIN00
```

---

## Testing

```bash
# Full suite
pytest

# Fully offline web API tests — zero HTTP calls, zero API cost
pytest tests/test_web_api_mocked.py -v

# P0 regression tests
pytest tests/test_p0_fixes.py -v

# Epic 3 pre-flight smoke test (59 assertions — runs from project root)
python -m tests.preflight_check
```

---

## Project Status

| Phase | Focus | Status |
|---|---|---|
| P0 — Blocking fixes | `asyncio` safety, stale state, HITL loop cap, test suite | Done |
| P1 — High priority | City registry, settings module, planner decomposition, token tracking | Done |
| P2 — Important | Dead code removal, typed state, logging standards, mock tests | Done |
| P3 — Epic 3 | `WebSupervisor`/`CyberAgent` Zero-Trust pipeline, hierarchical web agent team (4 agents), Presidio PII redaction, Lakera Guard v2, `diff_changed_tasks` cascade fix | Done |
| P4 — Future | FastAPI/HTTP server (`src/api/` stub exists), `AsyncSqliteSaver` migration | Planned |

---

## License

MIT
