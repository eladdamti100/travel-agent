"""
AI Travel Planner interactive terminal entrypoint.
"""

import os
import re
from pathlib import Path

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


def run() -> None:
    """
    Starts the interactive terminal loop.
    """
    validate_provider_env()
    print_banner()

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

                    update_status_for_node(node_name, node_data, status)

                    messages = node_data.get("messages", [])
                    if messages:
                        last_msg = messages[-1]

                        if (
                            isinstance(last_msg, AIMessage)
                            and last_msg.content
                            and not getattr(last_msg, "tool_calls", None)
                        ):
                            text = extract_text(last_msg.content)

                            if text and text not in seen_contents:
                                seen_contents.add(text)
                                status.stop()
                                print_agent(text)
                                status.start()

            print_status(
                city=accumulated.get("current_city"),
                budget=accumulated.get("total_budget"),
                tool_count=accumulated.get("tool_call_count", 0),
            )

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
