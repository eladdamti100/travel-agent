"""
Cache viewer — shows all semantic cache entries with live expiry status.

Usage:
    python scripts/view_cache.py              # latest 50 entries
    python scripts/view_cache.py --all        # every row
    python scripts/view_cache.py --dest paris # filter by destination
    python scripts/view_cache.py --watch      # refresh every 5 seconds
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

_DB = Path.home() / ".cache" / "travel-agent" / "semantic_cache.db"

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.text import Text
    from rich.live import Live
    from rich.panel import Panel
    from rich.columns import Columns
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _expiry_status(row: dict) -> tuple[str, str]:
    """Returns (label, color) describing the row's expiry state."""
    today = date.today().isoformat()

    # Trip-date expiry
    vu = row.get("valid_until")
    if vu:
        if vu < today:
            return f"expired {vu}", "red"
        days_left = (date.fromisoformat(vu) - date.today()).days
        if days_left <= 2:
            return f"expires {vu} ({days_left}d)", "yellow"
        return f"valid until {vu}", "green"

    # TTL expiry
    ttl = row.get("ttl_days", 30)
    created = row.get("created_at", "")[:10]
    if created:
        try:
            from datetime import timedelta
            exp = (date.fromisoformat(created) + __import__("datetime").timedelta(days=ttl)).isoformat()
            if exp < today:
                return f"TTL expired {exp}", "red"
            days_left = (date.fromisoformat(exp) - date.today()).days
            if days_left <= 3:
                return f"TTL expires {exp} ({days_left}d)", "yellow"
            return f"TTL ok (expires {exp})", "green"
        except ValueError:
            pass

    return "unknown", "dim"


def _load_rows(dest_filter: str | None, limit: int | None) -> list[dict]:
    if not _DB.exists():
        return []
    with sqlite3.connect(_DB) as conn:
        conn.row_factory = sqlite3.Row
        sql = """
            SELECT id, route, source, ttl_days, valid_until, created_at,
                   json_extract(trip_context_json, '$.destination_city') AS destination,
                   json_extract(trip_context_json, '$.origin_airport')   AS origin,
                   json_extract(trip_context_json, '$.total_budget')     AS budget,
                   json_extract(trip_context_json, '$.travel_start_date') AS start_date,
                   substr(query, 1, 72)                                  AS query_short,
                   length(answer)                                        AS answer_len
            FROM semantic_cache
        """
        params = []
        if dest_filter:
            sql += " WHERE LOWER(json_extract(trip_context_json, '$.destination_city')) = ?"
            params.append(dest_filter.lower())
        sql += " ORDER BY id DESC"
        if limit:
            sql += f" LIMIT {limit}"
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def _load_stats() -> dict:
    if not _DB.exists():
        return {}
    with sqlite3.connect(_DB) as conn:
        total = conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
        web   = conn.execute("SELECT COUNT(*) FROM semantic_cache WHERE source='web'").fetchone()[0]
        db_   = conn.execute("SELECT COUNT(*) FROM semantic_cache WHERE source='db'").fetchone()[0]
        today = date.today().isoformat()
        expired = conn.execute(
            """SELECT COUNT(*) FROM semantic_cache
               WHERE (valid_until IS NOT NULL AND valid_until < ?)
                  OR created_at < datetime('now', '-' || ttl_days || ' days')""",
            (today,),
        ).fetchone()[0]
        wal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    return {"total": total, "web": web, "db": db_, "expired": expired, "wal": wal_mode}


# ── rendering ─────────────────────────────────────────────────────────────────

def _build_table(rows: list[dict], stats: dict) -> "Table":
    table = Table(
        box=box.ROUNDED,
        title=f"[bold cyan]Semantic Cache[/]  [dim]{_DB}[/]",
        caption=(
            f"[green]{stats.get('total',0)} total[/]  "
            f"[blue]{stats.get('db',0)} db[/]  "
            f"[magenta]{stats.get('web',0)} web[/]  "
            f"[red]{stats.get('expired',0)} expired[/]  "
            f"WAL=[bold]{'✓' if stats.get('wal')=='wal' else '✗'}[/]"
        ),
        show_lines=False,
        expand=True,
    )
    table.add_column("ID",      style="dim",        width=5,  justify="right")
    table.add_column("Source",  width=5)
    table.add_column("Origin",  width=6)
    table.add_column("Dest",    width=10)
    table.add_column("Budget",  width=8,  justify="right")
    table.add_column("Start",   width=12)
    table.add_column("TTL",     width=4,  justify="right")
    table.add_column("Status",  width=26)
    table.add_column("Ans KB",  width=6,  justify="right")
    table.add_column("Query",   min_width=20)

    today = date.today().isoformat()
    for r in rows:
        label, color = _expiry_status(r)
        dest = (r.get("destination") or "—").title()
        budget = f"${int(r['budget'])}" if r.get("budget") else "—"
        src_color = "magenta" if r.get("source") == "web" else "blue"
        ans_kb = f"{r['answer_len']/1024:.1f}" if r.get("answer_len") else "—"
        table.add_row(
            str(r["id"]),
            Text(r.get("source", "?"), style=src_color),
            (r.get("origin") or "—").upper(),
            dest,
            budget,
            r.get("start_date") or "—",
            str(r.get("ttl_days", "?")),
            Text(label, style=color),
            ans_kb,
            r.get("query_short") or "—",
        )

    return table


def _print_plain(rows: list[dict], stats: dict) -> None:
    print(f"Cache: {_DB}")
    print(f"Total: {stats.get('total',0)}  web: {stats.get('web',0)}  db: {stats.get('db',0)}  expired: {stats.get('expired',0)}")
    print("-" * 100)
    fmt = "{:<5} {:<5} {:<6} {:<12} {:<8} {:<12} {:<4} {:<24} {}"
    print(fmt.format("ID", "Src", "Origin", "Dest", "Budget", "Start", "TTL", "Status", "Query"))
    print("-" * 110)
    for r in rows:
        label, _ = _expiry_status(r)
        dest = (r.get("destination") or "—").title()
        budget = f"${int(r['budget'])}" if r.get("budget") else "—"
        origin = (r.get("origin") or "—").upper()
        print(fmt.format(
            r["id"],
            r.get("source", "?"),
            origin,
            dest[:12],
            budget,
            r.get("start_date") or "—",
            str(r.get("ttl_days", "?")),
            label[:24],
            (r.get("query_short") or "—")[:60],
        ))


# ── main ──────────────────────────────────────────────────────────────────────

def _render(dest: str | None, limit: int | None) -> None:
    rows = _load_rows(dest, limit)
    stats = _load_stats()
    if HAS_RICH:
        console = Console()
        if not rows:
            console.print(f"[yellow]No entries in cache.[/]  DB: {_DB}")
        else:
            console.print(_build_table(rows, stats))
    else:
        _print_plain(rows, stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="View semantic cache entries.")
    parser.add_argument("--all",   action="store_true", help="Show all rows (default: latest 50)")
    parser.add_argument("--dest",  metavar="CITY",       help="Filter by destination city")
    parser.add_argument("--watch", action="store_true", help="Refresh every 5 seconds")
    args = parser.parse_args()

    if not _DB.exists():
        print(f"Cache DB not found: {_DB}")
        print("Run the travel agent at least once to create it.")
        sys.exit(1)

    limit = None if args.all else 50

    if args.watch and HAS_RICH:
        console = Console()
        with Live(console=console, refresh_per_second=0.2, screen=False) as live:
            while True:
                rows = _load_rows(args.dest, limit)
                stats = _load_stats()
                live.update(_build_table(rows, stats))
                time.sleep(5)
    else:
        _render(args.dest, limit)


if __name__ == "__main__":
    main()
