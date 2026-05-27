"""
Terminal status text helpers for graph execution.
"""

TOOL_LABELS = {
    "fetch_flights": "✈  Searching flights",
    "fetch_hotels": "🏨  Searching hotels",
    "fetch_activities": "🎭  Finding activities",
    "get_visa_requirement": "🛂  Checking visa requirements",
    "calculate_trip_cost": "💰  Calculating costs",
    "get_cheapest_flight": "✈  Finding cheapest flight",
    "get_cheapest_hotel": "🏨  Finding cheapest hotel",
    "list_destinations": "🗺  Listing destinations",
    "web_search": "🌐  Searching the web",
}

PLANNER_TASK_LABELS = {
    "fetch_flights": "✈ Flights",
    "fetch_hotels": "🏨 Hotels",
    "fetch_activities": "🎭 Activities",
    "check_visa": "🛂 Visa",
    "calculate_trip_cost": "💰 Cost",
    "fetch_restaurants": "🍽 Restaurants",
    "local_transport_guide": "🚇 Local Transport",
    "fetch_weather": "🌦 Weather",
    "events_finder": "🎉 Events",
    "airport_transfer_info": "🚖 Airport Transfer",
}


def update_status_for_node(node_name: str, node_data: dict, status) -> None:
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
        planner_status = node_data.get("planner_status")
        scheduler = node_data.get("planner_scheduler_result") or {}

        waves = scheduler.get("waves", [])

        if waves:
            current_wave = waves[0]

            labels = [
                PLANNER_TASK_LABELS.get(task, task)
                for task in current_wave.get("tasks", [])
            ]

            status.update(
                "[tool.call]Planner Wave "
                f"{current_wave.get('wave_number', 1)}: "
                f"{' · '.join(labels)}[/tool.call]"
            )

        elif planner_status == "missing_required_info":
            status.update("[tool.call]Waiting for missing trip information...[/tool.call]")

        else:
            status.update("[tool.call]Planning your trip...[/tool.call]")

    elif node_name == "cache_store":
        status.update("[tool.call]Saving answer to cache...[/tool.call]")

    elif node_name == "agent":
        messages = node_data.get("messages", [])

        if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
            labels = [
                TOOL_LABELS.get(tool_call["name"], f"⚙  {tool_call['name']}")
                for tool_call in messages[-1].tool_calls
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