"""
agent.py
LangGraph 4-node agentic workflow for EV infrastructure planning.
Nodes: analyze_demand → retrieve_context → generate_report → finalize
"""

import os
from typing import TypedDict

from langgraph.graph import StateGraph, END
from rag import retrieve_context as rag_retrieve


# ── API key: Streamlit secrets → os.environ ───────────────────────────────────
def _get_groq_key() -> str:
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


# ── State schema ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    zone_id: str
    demand_series: list
    mean: float
    std: float
    peak_threshold: float
    is_high_load: bool
    context: str
    report: str
    status: str


# ── Node 1: Analyze Demand ────────────────────────────────────────────────────
def analyze_demand(state: AgentState) -> AgentState:
    import numpy as np
    arr = np.array(state["demand_series"], dtype=float)
    mu = float(arr.mean())
    sigma = float(arr.std())
    threshold = mu + sigma
    is_high = bool(float(arr.max()) > threshold * 1.5)
    return {
        **state,
        "mean": round(mu, 3),
        "std": round(sigma, 3),
        "peak_threshold": round(threshold, 3),
        "is_high_load": is_high,
    }


# ── Node 2: Retrieve Context ──────────────────────────────────────────────────
def retrieve_context(state: AgentState) -> AgentState:
    load_label = "high load" if state["is_high_load"] else "moderate load"
    query = (
        f"EV charging zone {load_label} demand {state['mean']:.1f} kWh average "
        f"peak threshold {state['peak_threshold']:.1f} kWh. "
        "Infrastructure planning charger deployment grid capacity."
    )
    try:
        context = rag_retrieve(query, k=4)
    except Exception as e:
        context = f"[RAG unavailable: {e}] Apply default EV planning standards."
    return {**state, "context": context}


# ── Node 3: Generate Report ───────────────────────────────────────────────────
def generate_report(state: AgentState) -> AgentState:
    api_key = _get_groq_key()
    if not api_key:
        return {**state, "report": "", "status": "fallback"}

    prompt = f"""You are an expert EV infrastructure planner. Write a 5-section planning report.

## Zone Data
- Zone ID: {state['zone_id']}
- Average Demand: {state['mean']:.2f} kWh/hour
- Std Deviation: {state['std']:.2f} kWh
- Peak Threshold (μ+σ): {state['peak_threshold']:.2f} kWh
- High Load Zone: {'YES ⚠️' if state['is_high_load'] else 'No'}

## Relevant Planning Guidelines
{state['context']}

## Report Structure (use exactly these headings)
### 1. 📊 Demand Profile Summary
### 2. ⚡ Charger Deployment Recommendation
### 3. 🔋 Grid Capacity Assessment
### 4. 📅 Implementation Roadmap
### 5. 🚨 Risk Factors & Mitigation

Be specific, data-driven, and reference the actual numbers above."""

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior EV infrastructure planning specialist with expertise "
                        "in grid engineering, urban mobility, and smart charging systems."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.3,
        )
        report = response.choices[0].message.content
        return {**state, "report": report, "status": "success"}
    except Exception as e:
        return {**state, "report": "", "status": f"error:{e}"}


# ── Node 4: Finalize (validate + rule-based fallback) ────────────────────────
def finalize(state: AgentState) -> AgentState:
    if state.get("report") and len(state["report"]) > 100:
        return {**state, "status": "success"}

    # Rule-based fallback
    load_label = "HIGH-LOAD" if state["is_high_load"] else "MODERATE"
    charger_count = max(3, int(state["mean"] / 10))
    fast_count = max(1, charger_count // 3)
    grid_kw = int(state["peak_threshold"] * 1.3)

    report = f"""## AI Planning Report — Zone {state['zone_id']} *(Rule-Based Fallback)*

> **Note:** LLM service unavailable. This report uses the rule-based fallback engine.

### 1. 📊 Demand Profile Summary
Zone **{state['zone_id']}** is classified as a **{load_label}** demand zone.
- Average hourly demand: **{state['mean']:.2f} kWh**
- Standard deviation: **{state['std']:.2f} kWh**
- Peak threshold (μ+σ): **{state['peak_threshold']:.2f} kWh**
- High-load designation: **{'Yes ⚠️' if state['is_high_load'] else 'No'}**

### 2. ⚡ Charger Deployment Recommendation
Based on average demand of **{state['mean']:.2f} kWh/hour**:
- **{charger_count} Level 2 AC chargers** (7.2 kW) as the primary fleet
- **{fast_count} DC Fast Chargers** (50 kW) for peak demand periods
- Deploy in clusters of 4–6 units to maximise grid efficiency
- Stagger charging windows to reduce coincident peak draw

### 3. 🔋 Grid Capacity Assessment
- Estimated peak draw: **{state['peak_threshold']:.2f} kWh** sustained
- Recommended dedicated grid connection: **{grid_kw} kW**
- Install on-site battery buffer (≥ 100 kWh) for peak shaving
- Smart load management system mandatory for high-utilisation periods

### 4. 📅 Implementation Roadmap
- **Phase 1 (0–6 months):** Grid assessment, permits, install {max(2, charger_count // 2)} Level 2 units
- **Phase 2 (6–18 months):** DC Fast Chargers, BESS, smart management system
- **Phase 3 (18+ months):** Solar integration, V2G evaluation, capacity review

### 5. 🚨 Risk Factors & Mitigation
1. **Grid overload** → Demand response contracts + dynamic load balancing
2. **Low Phase 1 utilisation** → Incentivise early adopters with reduced rates
3. **Equipment reliability** → 97% uptime SLA + remote monitoring platform
"""
    return {**state, "report": report, "status": "fallback"}


# ── Graph compilation ─────────────────────────────────────────────────────────
def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("analyze_demand", analyze_demand)
    g.add_node("retrieve_context", retrieve_context)
    g.add_node("generate_report", generate_report)
    g.add_node("finalize", finalize)
    g.set_entry_point("analyze_demand")
    g.add_edge("analyze_demand", "retrieve_context")
    g.add_edge("retrieve_context", "generate_report")
    g.add_edge("generate_report", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


_graph = None


def run_agent(zone_id: str, demand_series: list) -> dict:
    """
    Run the full 4-node pipeline.
    Returns final state dict with keys: report, status, mean, std,
    peak_threshold, is_high_load, context.
    """
    global _graph
    if _graph is None:
        _graph = _build_graph()

    initial: AgentState = {
        "zone_id": zone_id,
        "demand_series": demand_series,
        "mean": 0.0,
        "std": 0.0,
        "peak_threshold": 0.0,
        "is_high_load": False,
        "context": "",
        "report": "",
        "status": "pending",
    }
    return _graph.invoke(initial)
