"""
Pre-flight checklist — Marco AI Travel Planner (Epic 3).
Run from the project root: python tests/preflight_check.py
"""
import asyncio
import inspect
import os
import sys

os.environ.setdefault("LLM_PROVIDER", "gemini")
os.environ.setdefault("GOOGLE_API_KEY", "test-key-preflight")

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
results: list[tuple[str, str]] = []


def check(label: str, condition: bool, *, critical: bool = True) -> None:
    tag = PASS if condition else (FAIL if critical else WARN)
    results.append((tag, label))
    print(f"  {tag}  {label}")


# ── PILLAR 1: Environment & Dependencies ──────────────────────────────────────
print()
print("=== PILLAR 1: Environment & Dependencies ===")

req_txt = open("requirements.txt", encoding="utf-8").read()
check("presidio-analyzer>=2.2.0 in requirements.txt", "presidio-analyzer>=2.2.0" in req_txt)
check("presidio-anonymizer>=2.2.0 in requirements.txt", "presidio-anonymizer>=2.2.0" in req_txt)
check("httpx in requirements.txt", "httpx" in req_txt)

from src.config.settings import Settings, settings  # noqa: E402

fields = Settings.model_fields
check("lakera_api_key in Settings",                    "lakera_api_key" in fields)
check("google_safe_browsing_key in Settings",          "google_safe_browsing_key" in fields)
check("pangea_redact_token NOT in Settings",           "pangea_redact_token" not in fields)
check("pangea_domain NOT in Settings",                 "pangea_domain" not in fields)

settings_src = open("src/config/settings.py", encoding="utf-8").read()
check("model_validator NOT imported in settings.py",   "model_validator" not in settings_src)

try:
    settings.validate_startup()
    check("validate_startup() passes with GOOGLE_API_KEY set", True)
except Exception as exc:
    check(f"validate_startup() raised: {exc}", False)

# ── PILLAR 2: CyberAgent & Presidio ──────────────────────────────────────────
print()
print("=== PILLAR 2: CyberAgent & Presidio ===")

from src.agents.cyber_agent import CyberAgent, _get_presidio_engines  # noqa: E402

cyber = CyberAgent()
check("check_prompt_injection is a coroutine function",
      inspect.iscoroutinefunction(cyber.check_prompt_injection))
check("check_urls is a coroutine function",
      inspect.iscoroutinefunction(cyber.check_urls))
check("redact_sensitive_data is a coroutine function",
      inspect.iscoroutinefunction(cyber.redact_sensitive_data))

cyber_src = open("src/agents/cyber_agent.py", encoding="utf-8").read()
check("Lakera uses /v2/guard endpoint",           "/v2/guard" in cyber_src)
check("Lakera uses messages[] payload format",    '"messages"' in cyber_src and '"role"' in cyber_src)
check("Lakera v1 endpoint NOT present",           "/v1/prompt_injection" not in cyber_src)
check("Safe Browsing uses threatMatches:find",    "threatMatches:find" in cyber_src)
check("Safe Browsing extracts threat url",        'threat"]["url"]' in cyber_src)

analyzer, anonymizer = _get_presidio_engines()
check("Presidio AnalyzerEngine initialized",      analyzer is not None)
check("Presidio AnonymizerEngine initialized",    anonymizer is not None)

check("asyncio.to_thread wraps analyzer.analyze",
      "asyncio.to_thread" in cyber_src and "analyzer.analyze" in cyber_src)
check("asyncio.to_thread wraps anonymizer.anonymize",
      "asyncio.to_thread" in cyber_src and "anonymizer.anonymize" in cyber_src)

sample = "Contact support@example.com or call (555) 867-5309, card 4111111111111111."
redacted = asyncio.run(cyber.redact_sensitive_data(sample))
check("Presidio redacts EMAIL_ADDRESS", "<EMAIL_ADDRESS>" in redacted)
check("Presidio redacts PHONE_NUMBER",  "<PHONE_NUMBER>"  in redacted)
check("Presidio redacts CREDIT_CARD",   "<CREDIT_CARD>"   in redacted)

# ── PILLAR 3: Multi-Agent Routing & Dependencies ──────────────────────────────
print()
print("=== PILLAR 3: Multi-Agent Routing & Dependencies ===")

from src.agents.task_registry import get_planner_agents  # noqa: E402

agents = get_planner_agents()
check(f"Exactly 7 agents registered (got {len(agents)})", len(agents) == 7)

agent_names = [getattr(a, "agent_name", a.__class__.__name__) for a in agents]
for expected in [
    "transport_agent", "stay_agent", "experience_agent",
    "transport_web_agent", "stay_web_agent", "experience_web_agent", "manager_web_agent",
]:
    check(f"  Agent registered: {expected}", expected in agent_names)

from src.agents.sub_agents.web_agents.manager_web_agent import ManagerWebAgent  # noqa: E402

mgr = ManagerWebAgent()
for key in ("live_currency_conversion", "fetch_country_metadata", "web_research_tavily"):
    check(f"  ManagerWebAgent.result_keys has {key}", key in mgr.result_keys)

from src.agents.planner_dependencies import diff_changed_tasks  # noqa: E402
from src.models.trip_context import TripContext  # noqa: E402

base = TripContext(
    destination_city="paris", origin_airport="TLV", origin_country="Israel",
    destination_country="France", duration_days=5, total_budget=3000, travel_month="July",
)

r_dest   = diff_changed_tasks(base, base.model_copy(update={"destination_city": "london",
                                                             "destination_country": "UK"}))
r_origin = diff_changed_tasks(base, base.model_copy(update={"origin_airport": "JFK"}))
r_budget = diff_changed_tasks(base, base.model_copy(update={"total_budget": 9999}))
r_none   = diff_changed_tasks(base, base)

check("destination change -> full replan (>10 tasks)",          len(r_dest) > 10)
check("destination change -> transport_live_research included",  "transport_live_research" in r_dest)
check("destination change -> stay_live_research included",       "stay_live_research" in r_dest)
check("origin_airport change -> fetch_flights invalidated",      "fetch_flights" in r_origin)
check("origin_airport change -> check_visa invalidated",         "check_visa" in r_origin)
check("origin_airport change -> transport_live_research",        "transport_live_research" in r_origin)
check("total_budget change -> calculate_trip_cost invalidated",  "calculate_trip_cost" in r_budget)
check("total_budget change -> live_currency_conversion",         "live_currency_conversion" in r_budget)
check("no change -> empty list",                                 r_none == [])

ws_src = open("src/agents/web_supervisor.py", encoding="utf-8").read()
check("WebSupervisor Step 1 check_prompt_injection present",    "check_prompt_injection" in ws_src)
check("WebSupervisor Step 2 _sanitize_context present",         "_sanitize_context" in ws_src)
check("WebSupervisor Step 3 asyncio.gather present",            "asyncio.gather" in ws_src)
check("WebSupervisor Step 4 check_urls present",                "check_urls" in ws_src)
check("WebSupervisor Step 5 redact_sensitive_data present",     "redact_sensitive_data" in ws_src)
check("WebSupervisor Step 6 inspect_inbound present",           "inspect_inbound" in ws_src)

# ── PILLAR 4: Documentation ───────────────────────────────────────────────────
print()
print("=== PILLAR 4: Documentation ===")

claude_md = open("CLAUDE.md", encoding="utf-8").read()
readme_md = open("README.md", encoding="utf-8").read()
check("CLAUDE.md has no Pangea references",          "pangea" not in claude_md.lower())
check("README.md has no Pangea references",          "pangea" not in readme_md.lower())
check("CLAUDE.md documents LAKERA_API_KEY",          "LAKERA_API_KEY" in claude_md)
check("CLAUDE.md documents GOOGLE_SAFE_BROWSING_KEY","GOOGLE_SAFE_BROWSING_KEY" in claude_md)
check("CLAUDE.md references Microsoft Presidio",     "Presidio" in claude_md)

# ── PILLAR 5: Import Integrity ───────────────────────────────────────────────
print()
print("=== PILLAR 5: Import Integrity ===")

try:
    from src.agents.web_supervisor import WebSupervisor  # noqa: F401
    check("WebSupervisor imports cleanly", True)
except Exception as exc:
    check(f"WebSupervisor import failed: {exc}", False)

try:
    from src.agents.sub_agents.replanning_agent import ReplanningResult  # noqa: F401
    import typing
    hint = typing.get_type_hints(ReplanningResult)["changed_tasks"]
    check(f"ReplanningResult.changed_tasks is List[str] (got {hint})",
          str(hint) == "typing.List[str]")
except Exception as exc:
    check(f"ReplanningResult import failed: {exc}", False)

try:
    from src.agents.researcher import run_researcher, _cyber as res_cyber  # noqa: F401
    check("researcher.py imports cleanly with _cyber instance", True)
    check("researcher CyberAgent is same class",
          isinstance(res_cyber, CyberAgent))
except Exception as exc:
    check(f"researcher import failed: {exc}", False)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
failures = [r for r in results if r[0] == FAIL]
warnings = [r for r in results if r[0] == WARN]
passes   = [r for r in results if r[0] == PASS]
print(f"TOTAL:  {len(passes)} PASS  |  {len(warnings)} WARN  |  {len(failures)} FAIL")
print()

if failures:
    print("STATUS: NO-GO")
    print()
    print("Critical failures:")
    for _, label in failures:
        print(f"  x  {label}")
    sys.exit(1)

print("STATUS: GO  --  System ready for Live Demo")
print()
print("  .env keys checklist:")
print()
print("  [REQUIRED — demo will not start without one of these]")
print("    GOOGLE_API_KEY              if LLM_PROVIDER=gemini (default)")
print("    GROQ_API_KEY                if LLM_PROVIDER=groq")
print()
print("  [STRONGLY RECOMMENDED — live web features degrade to static fallbacks]")
print("    TAVILY_API_KEY              web_research_tavily + all Tavily sub-tools")
print("    OPENCAGE_API_KEY            geocode_location (fallback: static coords)")
print("    TICKETMASTER_API_KEY        fetch_live_events (fallback: mock event list)")
print("    EXCHANGERATE_API_KEY        live_currency_conversion (fallback: static rates)")
print()
print("  [OPTIONAL — security hardening; regex/fail-open fallbacks active without them]")
print("    LAKERA_API_KEY              Lakera Guard v2 prompt-injection detection")
print("    GOOGLE_SAFE_BROWSING_KEY    URL safety scan on inbound web results")
print()
print("  [NOT NEEDED — runs fully local, no key]")
print("    PII redaction               Microsoft Presidio  (spaCy en_core_web_lg installed)")
