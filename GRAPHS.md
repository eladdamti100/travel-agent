# Travel Agent Ecosystem: Technical Architecture & Execution Matrix

## 1. System Topology (LangGraph Architecture)
*This graph illustrates the high-level routing, safety guardrails, and intent dispatch logic.*

```mermaid
graph TD
    %% Styling Configuration
    classDef startEnd fill:#A2C2E8,stroke:#333,stroke-width:2px,rx:10px,ry:10px;
    classDef coreProcess fill:#F0F4F8,stroke:#4A6B82,stroke-width:2px;
    classDef routerStyle fill:#FFEAA7,stroke:#D6A2E8,stroke-width:2px,shape:diamond;
    classDef cacheStyle fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
    classDef hitlStyle fill:#FFF3CD,stroke:#FFC107,stroke-width:2px;

    %% Subgraphs for visual clarity
    subgraph Guardrails [Layer 1: Security & Metadata]
        START((START)):::startEnd
        EXTRACT_META[Extract Metadata]:::coreProcess
        VALIDATOR[Validator]:::coreProcess
        ROUTE_VAL{Valid?}:::routerStyle
    end

    subgraph Orchestration [Layer 2: Intent Routing]
        MASTER_ORCH[Master Orchestrator]:::coreProcess
        ROUTE_ORCH{Intent Routing}:::routerStyle
    end

    subgraph Execution [Layer 3: Cache & Planning]
        PREFS_MEM[Preferences Memory]:::coreProcess
        RESEARCHER[Researcher Agent]:::coreProcess
        CACHE_CHECK[Semantic Cache]:::cacheStyle
        RESUME_HITL[Resume HITL Context]:::hitlStyle
        MASTER_PLANNER[Master Planner]:::coreProcess
        CACHE_STORE[Cache Store]:::cacheStyle
    end

    SUMMARIZER[Summarizer]:::coreProcess
    END_NODE((END)):::startEnd

    %% Flow Paths
    START --> EXTRACT_META
    EXTRACT_META --> VALIDATOR
    VALIDATOR --> ROUTE_VAL
    
    ROUTE_VAL -- Blocked --> END_NODE
    ROUTE_VAL -- HITL --> RESUME_HITL
    ROUTE_VAL -- Approved --> MASTER_ORCH
    
    RESUME_HITL --> MASTER_PLANNER
    MASTER_ORCH --> ROUTE_ORCH
    
    ROUTE_ORCH -- "Profile" --> PREFS_MEM
    ROUTE_ORCH -- "Research" --> RESEARCHER
    ROUTE_ORCH -- "Trip Plan" --> CACHE_CHECK
    
    PREFS_MEM --> SUMMARIZER
    RESEARCHER --> END_NODE
    
    CACHE_CHECK --> ROUTE_CACHE{Cache Result}:::routerStyle
    ROUTE_CACHE -- "Hit" --> END_NODE
    ROUTE_CACHE -- "Miss" --> MASTER_PLANNER
    
    MASTER_PLANNER --> ROUTE_PLAN{Planner Result}:::routerStyle
    ROUTE_PLAN -- "Missing Info" --> END_NODE
    ROUTE_PLAN -- "Final Plan" --> CACHE_STORE
    
    CACHE_STORE --> SUMMARIZER
    SUMMARIZER --> END_NODE
```

---

## 2. Async Task Dependency DAG (Master Planner Scheduling)

```mermaid
graph TD
    %% Styling Configuration
    classDef agentStyle fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px,rx:8px,ry:8px;
    classDef toolStyle fill:#FFF3E0,stroke:#FFB74D,stroke-width:1px,shape:rect;
    classDef readyStyle fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px;
    classDef blockedStyle fill:#FFF3E0,stroke:#FFB74D,stroke-width:1px;

    %% Data Inputs
    subgraph TripContext [TripContext State Model]
        ORIGIN_AIRPORT[origin_airport]:::contextStyle
        ORIGIN_COUNTRY[origin_country]:::contextStyle
        DEST_CITY[destination_city]:::contextStyle
        DURATION[duration_days]:::contextStyle
        BUDGET[total_budget]:::contextStyle
    end

    %% Execution Waves
    subgraph Wave1 [Wave 1: Async Concurrent Sub-Agents]
        CHECK_VISA[check_visa]:::readyStyle
        FETCH_FLIGHTS[fetch_flights]:::readyStyle
        FETCH_HOTELS[fetch_hotels]:::readyStyle
        FETCH_ACTIVITIES[fetch_activities]:::readyStyle
    end

    subgraph Wave2 [Wave 2: Downstream Blocked Calculations]
        CALCULATE_TRIP_COST[calculate_trip_cost]:::blockedStyle
    end

    %% Pathing
    ORIGIN_COUNTRY & DEST_CITY --> CHECK_VISA
    ORIGIN_AIRPORT & DEST_CITY --> FETCH_FLIGHTS
    DEST_CITY & DURATION --> FETCH_HOTELS
    DEST_CITY --> FETCH_ACTIVITIES

    FETCH_FLIGHTS & FETCH_HOTELS & FETCH_ACTIVITIES & BUDGET --> CALCULATE_TRIP_COST

    FINAL_PLAN[Synthesize Structured Trip Plan]:::finalStyle
    CALCULATE_TRIP_COST & CHECK_VISA --> FINAL_PLAN
    ```