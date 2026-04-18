"""
pages/2_AI_Planner.py
Agentic AI EV Infrastructure Planning Assistant — Streamlit Page 2.

IMPORTANT: sys.path fix must come before ANY project imports.
"""

# ── Path fix: insert project root so imports resolve correctly ────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Infrastructure Planner",
    page_icon="🤖",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.page-title {
    font-size: 2.1rem; font-weight: 700;
    background: linear-gradient(90deg, #1DB954, #4ECDC4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.page-sub { font-size: 1rem; color: #888; margin-bottom: 1.5rem; }

.node-box {
    background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
    border: 1px solid #333;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}
.node-box:hover { transform: translateY(-3px); }
.node-icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
.node-name { font-size: 0.85rem; font-weight: 700; color: #1DB954; }
.node-desc { font-size: 0.72rem; color: #aaa; margin-top: 0.25rem; }

.metric-card {
    background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
    border-radius: 10px; padding: 1rem; text-align: center; color: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.metric-val { font-size: 1.8rem; font-weight: 700; color: #1DB954; }
.metric-lbl { font-size: 0.8rem; color: #aaa; margin-top: 0.2rem; }

.high-load-badge {
    background: linear-gradient(90deg, #dc3545, #ff6b6b);
    color: white; padding: 6px 18px; border-radius: 20px;
    font-weight: 700; font-size: 0.85rem; display: inline-block;
}
.report-box {
    background: #0e1117; border: 1px solid #2d2d44;
    border-radius: 12px; padding: 1.5rem; margin-top: 1rem;
}
.arrow { font-size: 1.4rem; color: #555; text-align: center; padding-top: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Project imports (after sys.path fix) ──────────────────────────────────────
try:
    from agent import run_agent
    from pdf_export import generate_pdf
    IMPORTS_OK = True
except ImportError as _import_err:
    IMPORTS_OK = False
    st.error(f"❌ Import error: {_import_err}. Check that agent.py and pdf_export.py exist in the project root.")
    st.stop()

# ── Data constants ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "20220901-20230228_zone-cleaned-aggregated",
    "charge_1hour",
)
VOLUME_CSV = os.path.join(DATA_DIR, "volume.csv")


@st.cache_data(show_spinner="Loading dataset…")
def _load_wide(path: str):
    df = pd.read_csv(path)
    time_col = df.columns[0]
    df.rename(columns={time_col: "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"])
    zone_cols = [c for c in df.columns if c != "time"]
    return df, zone_cols


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">🤖 AI Infrastructure Planner</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Agentic AI · LangGraph 4-node pipeline · Groq LLaMA 3.3-70B · FAISS RAG</div>',
    unsafe_allow_html=True,
)

# ── Agent workflow diagram ────────────────────────────────────────────────────
st.markdown("### 🔄 Agent Workflow")

nodes = [
    ("📊", "analyze_demand", "Computes μ, σ, peak threshold, high-load flag"),
    ("🔍", "retrieve_context", "FAISS semantic search · keyword fallback"),
    ("🧠", "generate_report", "Groq LLaMA 3.3-70B · 5-section prompt"),
    ("✅", "finalize", "Validates output · rule-based fallback"),
]

cols = st.columns([3, 1, 3, 1, 3, 1, 3])
for i, (icon, name, desc) in enumerate(nodes):
    with cols[i * 2]:
        st.markdown(
            f'<div class="node-box">'
            f'<div class="node-icon">{icon}</div>'
            f'<div class="node-name">{name}</div>'
            f'<div class="node-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if i < len(nodes) - 1:
        with cols[i * 2 + 1]:
            st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Load data ─────────────────────────────────────────────────────────────────
if not os.path.exists(VOLUME_CSV):
    st.error("Dataset not found. Please ensure the default dataset is present.")
    st.stop()

wide_df, zone_cols = _load_wide(VOLUME_CSV)

# ── Sidebar zone selector ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Zone Selection")
    avg_demand = wide_df[zone_cols].mean().sort_values(ascending=False)
    top5 = avg_demand.head(5).index.tolist()
    options = top5 + [z for z in zone_cols if z not in top5]
    selected_zone = st.selectbox("Select zone to analyse:", options=options, index=0, key="planner_zone")
    st.markdown("---")
    st.caption("**Model:** Groq LLaMA 3.3-70B")
    st.caption("**RAG:** FAISS + sentence-transformers")
    st.caption("**Fallback:** Keyword overlap + rule-based")

# ── Zone demand stats ─────────────────────────────────────────────────────────
zone_series = wide_df[selected_zone].dropna().tolist()
mu = float(np.mean(zone_series))
sigma = float(np.std(zone_series))
threshold = mu + sigma
is_high = float(np.max(zone_series)) > threshold * 1.5

st.subheader(f"📊 Demand Profile — Zone {selected_zone}")

# Metric cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{mu:.2f}</div><div class="metric-lbl">Avg Demand (kWh/hr)</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{sigma:.2f}</div><div class="metric-lbl">Std Deviation (kWh)</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{threshold:.2f}</div><div class="metric-lbl">Peak Threshold μ+σ</div></div>', unsafe_allow_html=True)
with c4:
    badge = '<span class="high-load-badge">⚠️ HIGH LOAD</span>' if is_high else "✅ Moderate"
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:1.1rem;padding-top:0.4rem;">{badge}</div><div class="metric-lbl">Load Classification</div></div>', unsafe_allow_html=True)

# Demand profile chart
st.markdown("")
time_col = wide_df["time"]
demand_col = wide_df[selected_zone]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=time_col, y=demand_col,
    mode="lines", name="Hourly Demand",
    line=dict(color="#1DB954", width=1.2), opacity=0.85,
))
fig.add_hline(
    y=threshold, line_dash="dash", line_color="#FFD93D", line_width=1.5,
    annotation_text=f"Peak Threshold: {threshold:.2f} kWh",
    annotation_position="top right",
)
fig.add_hline(
    y=mu, line_dash="dot", line_color="#4ECDC4", line_width=1,
    annotation_text=f"Mean: {mu:.2f} kWh",
    annotation_position="bottom right",
)
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Time", yaxis_title="Demand (kWh)",
    height=360, margin=dict(l=40, r=20, t=30, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Generate Plan ─────────────────────────────────────────────────────────────
st.subheader("🤖 Generate Infrastructure Plan")

api_key_present = bool(
    (lambda: __import__("streamlit").secrets.get("GROQ_API_KEY", ""))()
    if True else ""
) or bool(os.environ.get("GROQ_API_KEY", ""))

if not api_key_present:
    st.warning(
        "⚠️ **GROQ_API_KEY not found.** "
        "Add it to `.streamlit/secrets.toml` locally or to Streamlit Cloud Secrets. "
        "The rule-based fallback report will be used instead.",
        icon="🔑",
    )

generate_btn = st.button(
    "⚡ Generate AI Planning Report",
    type="primary",
    use_container_width=True,
    key="generate_btn",
)

if generate_btn:
    progress = st.progress(0, text="Initialising agent…")
    status_box = st.empty()

    try:
        progress.progress(10, text="Node 1/4 — Analysing demand statistics…")
        status_box.info("📊 Analysing zone demand: computing μ, σ, peak threshold…")

        # Run agent (nodes 1-2 are fast, nodes 3-4 call Groq)
        # We update progress between calls by wrapping; LangGraph is synchronous
        import time as _time

        progress.progress(30, text="Node 2/4 — Retrieving planning context (RAG)…")
        status_box.info("🔍 Retrieving relevant EV planning guidelines…")
        _time.sleep(0.4)  # visual beat

        progress.progress(55, text="Node 3/4 — Calling Groq LLaMA 3.3-70B…")
        status_box.info("🧠 LLM generating 5-section planning report…")

        result = run_agent(zone_id=selected_zone, demand_series=zone_series)

        progress.progress(85, text="Node 4/4 — Finalising & validating output…")
        status_box.info("✅ Validating report structure…")
        _time.sleep(0.3)

        progress.progress(100, text="Done!")
        status_box.empty()

        # ── Store in session state ─────────────────────────────────────────────
        st.session_state["agent_result"] = result
        st.session_state["agent_zone"] = selected_zone

    except Exception as e:
        progress.empty()
        status_box.empty()
        st.error(f"❌ Agent error: {e}")

# ── Render report from session state ─────────────────────────────────────────
if "agent_result" in st.session_state and st.session_state.get("agent_zone") == selected_zone:
    result = st.session_state["agent_result"]
    report_text = result.get("report", "")
    agent_status = result.get("status", "unknown")

    if agent_status == "success":
        st.success("✅ Report generated by Groq LLaMA 3.3-70B")
    elif agent_status == "fallback":
        st.warning("⚠️ Rule-based fallback report (LLM unavailable or API key missing)")
    else:
        st.info(f"ℹ️ Status: {agent_status}")

    st.markdown("---")
    st.subheader(f"📋 Planning Report — Zone {selected_zone}")
    st.markdown(f'<div class="report-box">', unsafe_allow_html=True)
    st.markdown(report_text)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── PDF download ──────────────────────────────────────────────────────────
    st.markdown("")
    metrics = {
        "mean": result.get("mean", mu),
        "std": result.get("std", sigma),
        "peak_threshold": result.get("peak_threshold", threshold),
        "is_high_load": result.get("is_high_load", is_high),
    }
    try:
        pdf_bytes = generate_pdf(report_text, selected_zone, metrics)
        st.download_button(
            label="📥 Download Planning Report (PDF)",
            data=pdf_bytes,
            file_name=f"ev_plan_{selected_zone.replace(' ', '_')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="pdf_download",
        )
    except Exception as pdf_err:
        st.warning(f"PDF generation failed: {pdf_err}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.82rem;'>"
    "EV Demand Dashboard · Milestone 2 · Agentic AI · LangGraph + Groq + FAISS"
    "</div>",
    unsafe_allow_html=True,
)