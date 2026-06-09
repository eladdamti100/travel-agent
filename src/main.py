"""
AI Travel Planner interactive terminal entrypoint.
"""

import concurrent.futures
import os
import re
import time
from pathlib import Path
from typing import Optional

# Suppress tqdm progress bars globally (embedding model loading)
os.environ["TQDM_DISABLE"] = "1"

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from rich.panel import Panel
from rich.prompt import Prompt

from src.cli.status import update_status_for_node
from src.cli.ui import (
    ask_for_session_id,
    console,
    extract_text,
    print_agent,
    print_banner,
    print_status,
)
from langgraph.types import Command
from src.graph.workflow import graph
from src.utils.logger import get_logger

load_dotenv(Path(__file__).parent.parent / ".env")

logger = get_logger("main")


def validate_provider_env() -> None:
    """
    Exits early when the selected LLM provider does not have an API key.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key or api_key.startswith("your_"):
            raise SystemExit(
                "\n[ERROR] GROQ_API_KEY is not set.\n"
                "Edit the .env file and add your Groq key, or set LLM_PROVIDER=gemini.\n"
            )
        return

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        raise SystemExit(
            "\n[ERROR] GOOGLE_API_KEY is not set.\n"
            "Edit the .env file in the project root and add your real key.\n"
        )


def print_rate_limit_error(error: str) -> None:
    """
    Prints a provider-specific rate limit message.
    """
    wait = re.search(r"retry in (\d+)", error)
    wait_msg = f"Retry in {wait.group(1)}s." if wait else "Try again in a moment."

    provider = os.getenv("LLM_PROVIDER", "gemini").upper()
    model = os.getenv(
        "LLM_MODEL",
        "llama-3.3-70b-versatile" if provider == "GROQ" else "gemini-2.5-flash",
    )

    if provider == "GROQ":
        limits = (
            "llama-3.3-70b-versatile: 500 req/day · "
            "llama-3.1-8b-instant: 14,400 req/day"
        )
        tip = "Switch to a faster model: set LLM_MODEL=llama-3.1-8b-instant in .env"
    else:
        limits = "20 requests/day · 10 per minute"
        tip = "Switch to Groq for higher limits: set LLM_PROVIDER=groq in .env"

    console.print(Panel(
        f"[yellow]Rate limit reached on {provider} ({model}).\n{wait_msg}[/yellow]\n\n"
        f"[dim]{limits}\n{tip}[/dim]",
        title="[red]Rate Limited[/red]",
        border_style="red",
    ))


def is_rate_limit_error(error: str) -> bool:
    """
    Returns True when an exception message looks like provider rate limiting.
    """
    return (
        "429" in error
        or "RESOURCE_EXHAUSTED" in error
        or "rate_limit" in error.lower()
        or "quota" in error.lower()
        or "rate limit" in error.lower()
    )


def _prompt_plan_approval(critique: dict) -> dict:
    """
    Shows the critic summary and prompts the user to Approve, Edit, or Cancel.

    Returns {"decision": "approved"|"edit"|"cancelled", "feedback": str}.
    """
    score = critique.get("score", "?")
    reason = critique.get("reason", "")
    issues = critique.get("issues", [])
    suggestions = critique.get("suggestions", [])

    console.print()
    console.print(Panel(
        "[bold white]The graph is paused — waiting for your decision.[/bold white]\n\n"
        f"[bold]Score:[/bold] {score}/10\n"
        f"[bold]Summary:[/bold] {reason}\n"
        + (
            "\n[bold]Issues:[/bold]\n" + "\n".join(f"  • {i}" for i in issues)
            if issues else ""
        )
        + (
            "\n[bold]Suggestions:[/bold]\n" + "\n".join(f"  → {s}" for s in suggestions)
            if suggestions else ""
        ),
        title="[cyan bold]Human Approval Required — Critic Review[/cyan bold]",
        border_style="cyan",
    ))
    console.print()
    console.print(
        "  [green bold][A] Approve[/green bold] — accept this plan and save it\n"
        "  [yellow bold][E] Edit[/yellow bold]   — describe what to change, the agent will replan\n"
        "  [red bold][C] Cancel[/red bold] — discard this plan and start over\n"
    )

    while True:
        choice = Prompt.ask(
            "[bold cyan]Your choice (A / E / C)[/bold cyan]",
        ).strip().lower()

        if choice in ("a", "approve", "approved"):
            return {"decision": "approved", "feedback": ""}

        if choice in ("c", "cancel", "cancelled"):
            return {"decision": "cancelled", "feedback": ""}

        if choice in ("e", "edit"):
            feedback = Prompt.ask(
                "[bold cyan]Describe what you'd like changed[/bold cyan]"
            ).strip()
            return {"decision": "edit", "feedback": feedback}

        console.print("[dim]Please enter A, E, or C.[/dim]")


def _run_reviewer_async(plan_text: str, *, is_admin: bool) -> None:
    """
    Runs the plan reviewer in a background thread so it never blocks regular users.

    Admin sessions display the review. Regular sessions only log the review internally.
    """
    from src.agents.reviewer import review_plan

    def _do_review() -> None:
        try:
            review = review_plan(plan_text)
            if is_admin:
                console.print()
                console.print(Panel(
                    review,
                    title="[yellow]Plan Review (Admin)[/yellow]",
                    border_style="yellow",
                ))
            else:
                logger.info("Plan review (non-admin, internal only):\n%s", review)
        except Exception as err:
            logger.warning("Reviewer failed: %s", err)

    if is_admin:
        with console.status("[dim]Reviewing plan...[/dim]", spinner="dots"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_review)
                try:
                    future.result(timeout=60)
                except concurrent.futures.TimeoutError:
                    console.print("[dim]Plan review timed out.[/dim]")
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(_do_review)
        executor.shutdown(wait=False)


def _prewarm_cache_model() -> None:
    """
    Loads the sentence-transformers embedding model before the first user turn.

    On first run this downloads ~91 MB from HuggingFace. Subsequent runs load
    from the local disk cache in under a second.
    """
    from src.services.semantic_cache import warm_embedding_model

    with console.status("[dim]Initializing semantic cache...[/dim]", spinner="dots"):
        downloaded = warm_embedding_model()

    if downloaded:
        console.print(
            "[dim]  Embedding model downloaded and ready.[/dim]\n"
        )


def run() -> None:
    """
    Starts the interactive terminal loop.
    """
    validate_provider_env()
    print_banner()
    _prewarm_cache_model()

    session_id = ask_for_session_id()
    config = {"configurable": {"thread_id": session_id}}

    is_admin = session_id.upper().endswith("ADMIN00")

    if is_admin:
        console.print(
            f"[dim]  Session: {session_id}[/dim]  "
            "[bold yellow]⚙  ADMIN MODE — plan reviews enabled[/bold yellow]\n"
        )
    else:
        console.print(
            f"[dim]  Session: {session_id} — memory will be saved and restored automatically.[/dim]\n"
        )

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! Safe travels! ✈[/dim]\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            console.print("\n[dim]Goodbye! Safe travels! ✈[/dim]\n")
            break

        console.print()

        turn_start_time = time.perf_counter()
        node_timings: dict[str, float] = {}

        seen_contents: set[str] = set()
        accumulated: dict = {
            "current_city": None,
            "total_budget": None,
            "tool_call_count": 0,
            "cache_status": None,
            "planning_mode": None,
        }

        final_plan_text: str | None = None
        plan_nodes = {"master_planner", "cache_check"}
        pending_interrupt: Optional[dict] = None
        # Track whether we're inside a critic-triggered replan (suppress plan reprint)
        _critic_replan_in_progress: bool = False

        def _stream_graph(input_payload):
            """Stream one graph pass, return (final_plan_text, interrupt_payload)."""
            nonlocal final_plan_text, pending_interrupt, _critic_replan_in_progress

            with console.status("[tool.call]Starting...[/tool.call]", spinner="dots") as status:
                for event in graph.stream(
                    input_payload,
                    config,
                    stream_mode="updates",
                ):
                    node_start_time = time.perf_counter()
                    node_name = next(iter(event))
                    node_data = event[node_name]

                    # LangGraph surfaces interrupts as {"__interrupt__": [...]}
                    if node_name == "__interrupt__":
                        interrupts = node_data if isinstance(node_data, (list, tuple)) else [node_data]
                        for intr in interrupts:
                            value = getattr(intr, "value", intr) if not isinstance(intr, dict) else intr
                            if isinstance(value, dict) and value.get("type") == "plan_approval":
                                pending_interrupt = value
                        node_timings[node_name] = node_timings.get(node_name, 0.0) + (
                            time.perf_counter() - node_start_time
                        )
                        continue

                    if node_data is None:
                        node_timings[node_name] = node_timings.get(node_name, 0.0) + (
                            time.perf_counter() - node_start_time
                        )
                        continue

                    for key in ("current_city", "total_budget", "tool_call_count", "cache_status", "planning_mode"):
                        if key in node_data:
                            accumulated[key] = node_data[key]

                    update_status_for_node(node_name, node_data, status)

                    # Print a visible line when the critic rejects the plan so the
                    # demo audience can clearly see the self-correction loop.
                    if node_name == "critic":
                        critique = node_data.get("critique_result") or {}
                        attempts = node_data.get("critic_attempts", 1)
                        score = critique.get("score", "?")
                        from src.graph.nodes import MAX_CRITIC_ATTEMPTS
                        if not critique.get("passed", True) and attempts < MAX_CRITIC_ATTEMPTS:
                            reason = critique.get("reason", "")
                            status.stop()
                            console.print(
                                f"\n[yellow bold]Critic failed[/yellow bold] "
                                f"(score {score}/10, attempt {attempts}/{MAX_CRITIC_ATTEMPTS})"
                                + (f" — {reason}" if reason else "")
                            )
                            console.print(
                                "[yellow]  → Replanning silently with critic feedback...[/yellow]\n"
                            )
                            status.start()
                            _critic_replan_in_progress = True
                        else:
                            _critic_replan_in_progress = False

                    messages = node_data.get("messages", [])
                    if messages:
                        last_msg = messages[-1]

                        if (
                            isinstance(last_msg, AIMessage)
                            and last_msg.content
                            and not getattr(last_msg, "tool_calls", None)
                        ):
                            text = extract_text(last_msg.content)

                            # Suppress reprinting the plan during a critic-triggered replan.
                            # Only the final approved plan (after critic passes or max attempts)
                            # gets displayed.
                            is_critic_replan_output = (
                                _critic_replan_in_progress
                                and node_name == "master_planner"
                            )

                            if text and text not in seen_contents and not is_critic_replan_output:
                                seen_contents.add(text)
                                status.stop()
                                print_agent(text)
                                status.start()
                            elif text:
                                seen_contents.add(text)

                            if node_name in plan_nodes and text:
                                is_hitl_stop = (
                                    node_data.get("planner_status") == "missing_required_info"
                                )
                                if not is_hitl_stop:
                                    final_plan_text = text

                    node_timings[node_name] = node_timings.get(node_name, 0.0) + (
                        time.perf_counter() - node_start_time
                    )

        try:
            _stream_graph({"messages": [("user", user_input)], "is_admin": is_admin})

            # Handle plan-approval interrupt produced by hitl_approval_node.
            while pending_interrupt is not None:
                critique = pending_interrupt.get("critique", {})
                pending_interrupt = None

                user_response = _prompt_plan_approval(critique)

                if user_response["decision"] == "cancelled":
                    console.print("\n[dim]Trip planning cancelled. Safe travels![/dim]\n")
                    break

                if user_response["decision"] == "approved":
                    console.print("\n[green]Plan approved — saving to cache...[/green]\n")

                _stream_graph(Command(resume=user_response))

            # Only show the status line for full-planning and replanning turns.
            # Researcher queries, preference updates, HITL clarification requests,
            # and other non-planning paths produce no meaningful city/budget/cache
            # summary, so suppressing avoids a noisy blank or partial status bar.
            _planning_mode = accumulated.get("planning_mode")
            _cache_status = accumulated.get("cache_status")
            if _planning_mode in ("full_planning", "replanning") or _cache_status == "hit":
                print_status(
                    city=accumulated.get("current_city"),
                    budget=accumulated.get("total_budget"),
                    tool_count=accumulated.get("tool_call_count", 0),
                    cache_status=_cache_status,
                    planning_mode=_planning_mode,
                )

            turn_elapsed = time.perf_counter() - turn_start_time
            logger.info(
                "turn performance | total=%.2fs | nodes=%s",
                turn_elapsed,
                {
                    node: round(seconds, 2)
                    for node, seconds in node_timings.items()
                },
            )

            if final_plan_text:
                _run_reviewer_async(final_plan_text, is_admin=is_admin)

        except Exception as error:
            err = str(error)

            if is_rate_limit_error(err):
                print_rate_limit_error(err)
            else:
                console.print(Panel(
                    f"[red]Unexpected error:[/red] {err[:300]}",
                    title="[red]Error[/red]",
                    border_style="red",
                ))

        logger.info(
            "turn complete | city=%s | budget=%s | tools=%d | cache=%s",
            accumulated.get("current_city", "—"),
            accumulated.get("total_budget", "—"),
            accumulated.get("tool_call_count", 0),
            accumulated.get("cache_status", "—"),
        )


if __name__ == "__main__":
    db_path = Path(__file__).parent.parent / "data" / "travel_agency.db"

    if not db_path.exists():
        console.print("[dim]First run — initializing database...[/dim]")
        from src.utils.db_init import create_travel_db

        create_travel_db()

    run()