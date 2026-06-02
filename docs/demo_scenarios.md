# Demo Scenarios — Session 7 HITL Graph

## Demo 1 — Paris 7 Days $500: Critic Fails → Agent Self-Corrects

**Goal:** Show the `master_planner → critic → master_planner` replan loop.

### Setup
- Session ID: `demo-paris-ADMIN00` (admin optional, not required)
- Budget: $500 (intentionally too low for 7 days in Paris)

### Script

1. Start the app: `python run.py`
2. Enter session ID: `demo-paris`
3. Prompt:
   ```
   Plan a 7-day trip to Paris from New York for 1 person with a total budget of $500.
   ```

### Expected Flow

```
master_planner  → generates plan (~$2,000 total estimated)
critic          → FAIL: "Estimated cost $2,000 exceeds budget $500 by $1,500"
                  attempt 1/2 → route_after_critic returns "master_planner"
master_planner  → replans with critic suggestions injected
                  (e.g. "Find a hotel under $32/night", "Look for flights under $175 total")
critic          → may still FAIL at attempt 2 (cap reached) → passes through to hitl_approval
hitl_approval   → graph suspends, shows Critic Review panel
```

Terminal output at HITL pause:
```
╭─ Critic Review ───────────────────────────────────╮
│ Score: 3/10                                        │
│ Summary: Budget exceeded by $1,200 ...             │
│ Issues:                                            │
│   • Estimated cost $1,700 exceeds budget $500      │
│ Suggestions:                                       │
│   → Find a hotel under $32/night                   │
╰────────────────────────────────────────────────────╯

What would you like to do? [A]pprove  [E]dit  [C]ancel
```

4. Choose **E** → type: `"Extend budget to $1500 or find budget accommodation"`
5. Graph resumes, planner regenerates with the feedback, critic passes, HITL shows again with score ≥ 7.
6. Choose **A** → plan is stored in cache.

### What to Observe
- Two `critic` node events in the stream before `hitl_approval`
- `critic_attempts` increments: 1 → 2
- Suggestions from critic appear in the planner's next replan context (Section 3 Notes)
- Final plan shows cheaper options (hostel, budget flights)

---

## Demo 2 — Normal Trip → HITL Stop (All 3 Options)

**Goal:** Show the plan-approval HITL with all three user paths (Approve, Edit, Cancel).

### Setup
- Session ID: `demo-tokyo`
- Budget: $3,000 (well within range for 7 days Tokyo)

### Script

1. Prompt:
   ```
   Plan a 7-day trip to Tokyo from London for 2 people, budget $3000.
   ```

### Path A — Approve

```
master_planner  → complete plan
critic          → PASS (score 8/10, within budget)
hitl_approval   → graph suspends
```

Terminal:
```
╭─ Critic Review ────────────────────────────────────╮
│ Score: 8/10                                        │
│ Summary: Plan is within budget ($2,800 / $3,000)   │
╰────────────────────────────────────────────────────╯

What would you like to do? [A]pprove  [E]dit  [C]ancel
```

Choose **A** → plan cached, `summarizer` runs, session ends cleanly.

### Path B — Edit

Same flow up to HITL pause.

Choose **E** → type: `"I prefer a ryokan hotel instead of a business hotel"`

Graph resumes:
- `hitl_decision = "edit"`, `hitl_feedback = "I prefer a ryokan hotel..."`
- `route_after_hitl` returns `"master_planner"`
- Planner regenerates; Section 3 Notes includes the feedback line
- Critic runs again, HITL pause again with updated plan

Choose **A** → plan cached.

### Path C — Cancel

Same flow up to HITL pause.

Choose **C** → terminal prints:
```
Trip planning cancelled. Safe travels!
```
`route_after_hitl` returns `END`. No cache write. Session loop continues for next query.

---

## Verification Checklist

| Check | Pass? |
|-------|-------|
| `critic_node` fires after every `master_planner` completion | |
| `critic_attempts` resets to 0 on each new user turn | |
| Rejected plan (attempt < 2) routes back to `master_planner` | |
| Attempt cap (2) routes to `hitl_approval` regardless of pass/fail | |
| Graph fully suspends at `hitl_approval` (no further node events) | |
| Approve → `cache_store` → `summarizer` → END | |
| Edit → `master_planner` re-runs with `hitl_feedback` visible in Section 3 | |
| Cancel → END, no cache write | |
| `hitl_feedback` and `critic_attempts` cleared after plan completion | |
