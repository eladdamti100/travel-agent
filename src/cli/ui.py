"""
Rich terminal rendering helpers for the travel planner CLI.
"""

import os
from typing import Optional

from rich.columns import Columns
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

from src.models.session import validate_session_id

_THEME = Theme({
    "user.label": "bold cyan",
    "user.text": "cyan",
    "agent.border": "green",
    "reviewer.border": "yellow",
    "researcher.border": "blue",
    "tool.call": "dim white",
    "status.city": "bold magenta",
    "status.budget": "bold green",
    "exit.hint": "dim",
})

console = Console(theme=_THEME, highlight=False)

_SUPPORTED = "Paris  ·  London  ·  Tokyo  ·  New York  ·  Berlin"


def extract_text(content) -> str:
    """
    Converts LangChain message content into printable text.
    """
    if isinstance(content, list):
        return "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def is_review(text: str) -> bool:
    """
    Returns True for reviewer messages that need review styling.
    """
    return text.startswith("\n---\n**Plan Review")


def is_researcher(text: str) -> bool:
    """
    Returns True for quick lookup answers that need researcher styling.
    """
    return (
        text.startswith("**Hotels in")
        or text.startswith("**Flights")
        or text.startswith("**Activities")
    )


def print_banner() -> None:
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


def print_agent(text: str) -> None:
    """
    Prints an AI message using the appropriate terminal panel style.

    When the text contains section separators (written by _generate_final_plan),
    each section is rendered as Markdown and the gaps between sections are filled
    with a green dashed Rule that matches Marco's border colour.
    """
    if is_review(text):
        clean = text.replace("\n---\n**Plan Review (auto):**\n", "").strip()
        console.print(Panel(
            Markdown(clean),
            title="[yellow]✦ Auto Review[/yellow]",
            border_style="yellow",
            padding=(1, 2),
        ))
        return

    if is_researcher(text):
        console.print(Panel(
            Markdown(text),
            title="[blue]⚡ Quick Lookup[/blue]",
            border_style="blue",
            padding=(1, 2),
        ))
        return

    sections = text.split("\n\n---\n\n")
    if len(sections) > 1:
        renderables = []
        for i, section in enumerate(sections):
            if section.strip():
                renderables.append(Markdown(section))
            if i < len(sections) - 1:
                renderables.append(Rule(characters="─ ", style="green"))
        console.print(Panel(
            Group(*renderables),
            title="[green]✈  Marco[/green]",
            border_style="green",
            padding=(1, 2),
        ))
        return

    console.print(Panel(
        Markdown(text),
        title="[green]✈  Marco[/green]",
        border_style="green",
        padding=(1, 2),
    ))


def print_status(
    city: Optional[str],
    budget: Optional[float],
    tool_count: int,
    cache_status: Optional[str] = None,
    planning_mode: Optional[str] = None,
) -> None:
    """
    Prints a compact turn summary after graph execution.
    """
    if not city and not budget:
        return

    parts = []

    if city:
        parts.append(Text.assemble(("Destination: ", "dim"), (city, "status.city")))

    if budget:
        parts.append(
            Text.assemble(("Budget: $", "dim"), (f"{budget:,.0f}", "status.budget"))
        )
    
    if planning_mode:
        parts.append(Text.assemble(("Planning Mode: ", "dim"), (planning_mode, "status.city")))

    if cache_status == "hit":
        parts.append(Text("Tools used: 0", style="dim"))
        parts.append(Text("Cache: HIT ⚡", style="bold green"))
    else:
        parts.append(Text(f"Tools used: {tool_count}", style="dim"))
        parts.append(Text("Cache: MISS", style="dim yellow"))

    console.print(Rule(style="dim"))
    console.print(Columns(parts, padding=(0, 4)))
    console.print()


def ask_for_session_id() -> str:
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
