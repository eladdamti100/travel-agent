"""
AI Travel Planner — interactive terminal UI.

Run:  python run.py
"""

import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

from src.graph.workflow import graph
from src.models.session import validate_session_id
from src.utils.logger import get_logger

load_dotenv(Path(__file__).parent.parent / ".env")

_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
if _provider == "groq":
    _api_key = os.getenv("GROQ_API_KEY", "")
    if not _api_key or _api_key.startswith("your_"):
        raise SystemExit(
            "\n[ERROR] GROQ_API_KEY is not set.\n"
            "Edit the .env file and add your Groq key, or set LLM_PROVIDER=gemini.\n"
        )
else:
    _api_key = os.getenv("GOOGLE_API_KEY", "")
    if not _api_key or _api_key.startswith("your_"):
        raise SystemExit(
            "\n[ERROR] GOOGLE_API_KEY is not set.\n"
            "Edit the .env file in the project root and add your real key.\n"
        )

logger = get_logger("main")

_THEME = Theme({
    "user.label":        "bold cyan",
    "user.text":         "cyan",
    "agent.border":      "green",
    "reviewer.border":   "yellow",
    "researcher.border": "blue",
    "tool.call":         "dim white",
    "status.city":       "bold magenta",
    "status.budget":     "bold green",
    "exit.hint":         "dim",
})

console = Console(theme=_THEME, highlight=False)

_SUPPORTED = "Paris  ·  London  ·  Tokyo  ·  New York  ·  Berlin"

_TOOL_LABELS = {
    "fetch_flights":        "✈  Searching flights",
    "fetch_hotels":         "🏨  Searching hotels",
    "fetch_activities":     "🎭  Finding activities",
    "get_visa_requirement": "🛂  Checking visa requirements",
    "calculate_trip_cost":  "💰  Calculating costs",
    "get_cheapest_flight":  "✈  Finding cheapest flight",
    "get_cheapest_hotel":   "🏨  Finding cheapest hotel",
    "list_destinations":    "🗺  Listing destinations",
    "web_search":           "🌐  Searching the web",
}


def _extract_text(content) -> str:
    """
    Converts LangChain message content into printable text.
    """
    if isinstance(content, list):
        return "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _is_review(text: str) -> bool:
    """
    Returns True for reviewer messages that need review styling.
    """
    return text.startswith("\n---\n**Plan Review")


def _is_researcher(text: str) -> bool:
    """
    Returns True for quick lookup answers that need researcher styling.
    """
    return (
        text.startswith("**Hotels in")
        or text.startswith("**Flights")
        or text.startswith("**Activities")
    )


def _print_banner() -> None:
    """
    Prints the startup banner for the terminal UI.
    """
    console.print()

    provider_label = os.getenv("LLM_PROVIDER", "gemini").upper()

    console.print(Panel(
        Text.assemble(
            ("  AI Travel Planner\n", "bold white"),
            (f"  Powered by {provider_label} + LangGraph\n\n", "dim white"),
            ("  Destinations: ", "dim white"),
            (_SUPPORTED, "bold cyan"),
        ),
        title="[bold white]✈  Marco[/bold white]",
        border_style="green",
        padding=(0, 2),
    ))

    console.print(Text("  Type 'exit' to quit.", style="exit.hint"))
    console.print()


def _print_agent(text: str) -> None:
    """
    Prints an AI message using the appropriate terminal panel style.
    """
    if _is_review(text):
        clean = text.replace("\n---\n**Plan Review (auto):**\n", "").strip()
        console.print(Panel(
            Markdown(clean),
            title="[yellow]✦ Auto Review[/yellow]",
            border_style="yellow",
            padding=(1, 2),
        ))
        return

    if _is_researcher(text):
        console.print(Panel(
            Markdown(text),
            title="[blue]⚡ Quick Lookup[/blue]",
            border_style="blue",
            padding=(1, 2),
        ))
        return

    console.print(Panel(
        Markdown(text),
        title="[green]✈  Marco[/green]",
        border_style="green",
        padding=(1, 2),
    ))


def _print_status(city: Optional[str], budget: Optional[float], tool_count: int) -> None:
    """
    Prints a compact turn summary after graph execution.
    """
    if not city and not budget:
        return

    parts = []

    if city:
        parts.append(Text.assemble(("Destination: ", "dim"), (city, "status.city")))

    if budget:
        parts.append(Text.assemble(("Budget: $", "dim"), (f"{budget:,.0f}", "status.budget")))

    if tool_count:
        parts.append(Text(f"Tools used: {tool_count}", style="dim"))

    console.print(Rule(style="dim"))
    console.print(Columns(parts, padding=(0, 4)))
    console.print()


def _ask_for_session_id() -> str:
    """
    Prompts the user for a safe session ID.

    The session ID becomes LangGraph's thread_id, so it is validated before use.
    """
    max_attempts = 3

    for _ in range(max_attempts):
        raw = Prompt.ask(
            "[bold cyan]Enter your session ID[/bold cyan] [dim](press Enter for default)[/dim]",
            default="session_01",
        )

        result = validate_session_id(raw)

        if result.is_valid:
            return raw

        console.print(Panel(
            f"[red]Invalid session ID:[/red] {result.error_message}\nPlease try again.",
            border_style="red",
            padding=(0, 2),
        ))

    console.print("[red]Too many invalid attempts. Using default session.[/red]")
    return "session_01"


def _update_status_for_node(node_name: str, node_data: dict, status) -> None:
    """
    Updates the terminal spinner text according to the current graph node.
    """
    if node_name == "extract_metadata":
        status.update("[tool.call]Reading your message...[/tool.call]")

    elif node_name == "validator":
        status.update("[tool.call]Validating your request...[/tool.call]")

    elif node_name == "master_orchestrator":
        status.update("[tool.call]Routing your request...[/tool.call]")

    elif node_name == "preferences_memory":
        status.update("[tool.call]Checking and updating travel memory...[/tool.call]")

    elif node_name == "researcher":
        status.update("[tool.call]Searching travel data...[/tool.call]")

    elif node_name == "cache_check":
        status.update("[tool.call]Checking cache...[/tool.call]")

    elif node_name == "master_planner":
        status.update("[tool.call]Planning your trip...[/tool.call]")

    elif node_name == "cache_store":
        status.update("[tool.call]Saving answer to cache...[/tool.call]")

    elif node_name == "agent":
        msgs = node_data.get("messages", [])

        if msgs and hasattr(msgs[-1], "tool_calls") and msgs[-1].tool_calls:
            labels = [
                _TOOL_LABELS.get(tc["name"], f"⚙  {tc['name']}")
                for tc in msgs[-1].tool_calls
            ]
            status.update(f"[tool.call]{' · '.join(labels)}...[/tool.call]")
        else:
            status.update("[tool.call]Marco is thinking...[/tool.call]")

    elif node_name == "tools":
        status.update("[tool.call]Running tools...[/tool.call]")

    elif node_name == "reviewer":
        status.update("[tool.call]Reviewing your plan...[/tool.call]")

    elif node_name == "summarizer":
        status.update("[tool.call]Compressing conversation...[/tool.call]")


def run() -> None:
    """
    Starts the interactive terminal loop.
    """
    _print_banner()

    session_id = _ask_for_session_id()
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

        seen_contents: set[str] = set()
        accumulated: dict = {
            "current_city": None,
            "total_budget": None,
            "tool_call_count": 0,
        }

        try:
            with console.status("[tool.call]Starting...[/tool.call]", spinner="dots") as status:
                for event in graph.stream(
                    {"messages": [("user", user_input)], "is_admin": is_admin},
                    config,
                    stream_mode="updates",
                ):
                    node_name = next(iter(event))
                    node_data = event[node_name]

                    if node_data is None:
                        continue

                    for key in ("current_city", "total_budget", "tool_call_count"):
                        if key in node_data:
                            accumulated[key] = node_data[key]

                    _update_status_for_node(node_name, node_data, status)

                    msgs = node_data.get("messages", [])
                    if msgs:
                        last_msg = msgs[-1]

                        if (
                            isinstance(last_msg, AIMessage)
                            and last_msg.content
                            and not getattr(last_msg, "tool_calls", None)
                        ):
                            text = _extract_text(last_msg.content)

                            if text and text not in seen_contents:
                                seen_contents.add(text)
                                status.stop()
                                _print_agent(text)
                                status.start()

            _print_status(
                city=accumulated.get("current_city"),
                budget=accumulated.get("total_budget"),
                tool_count=accumulated.get("tool_call_count", 0),
            )

        except Exception as e:
            err = str(e)
            is_rate_limit = (
                "429" in err
                or "RESOURCE_EXHAUSTED" in err
                or "rate_limit" in err.lower()
                or "quota" in err.lower()
                or "rate limit" in err.lower()
            )

            if is_rate_limit:
                wait = re.search(r"retry in (\d+)", err)
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

            else:
                console.print(Panel(
                    f"[red]Unexpected error:[/red] {err[:300]}",
                    title="[red]Error[/red]",
                    border_style="red",
                ))

        logger.info(
            "turn complete | city=%s | budget=%s | tools=%d",
            accumulated.get("current_city", "—"),
            accumulated.get("total_budget", "—"),
            accumulated.get("tool_call_count", 0),
        )


if __name__ == "__main__":
    db_path = Path(__file__).parent.parent / "data" / "travel_agency.db"

    if not db_path.exists():
        console.print("[dim]First run — initializing database...[/dim]")
        from src.utils.db_init import create_travel_db

        create_travel_db()

    run()
