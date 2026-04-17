"""
pdf_export.py
Generate a styled PDF planning report using fpdf2.
Returns raw bytes for use with st.download_button.
"""

import re
from datetime import datetime
from fpdf import FPDF


def _safe(text: str) -> str:
    """Strip markdown syntax and ensure latin-1 safety for fpdf2."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    safe = ""
    for ch in text:
        try:
            ch.encode("latin-1")
            safe += ch
        except (UnicodeEncodeError, ValueError):
            safe += "?"
    return safe


def generate_pdf(report_text: str, zone_id: str, metrics: dict) -> bytes:
    """
    Build a styled PDF from the markdown report.

    Args:
        report_text: Markdown string from the agent report.
        zone_id:     Zone identifier string.
        metrics:     Dict with keys: mean, std, peak_threshold, is_high_load.

    Returns:
        PDF content as bytes.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── Dark header banner ────────────────────────────────────────────────────
    pdf.set_fill_color(30, 30, 47)
    pdf.rect(0, 0, 210, 38, "F")

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(29, 185, 84)
    pdf.set_xy(10, 7)
    pdf.cell(0, 10, "EV Infrastructure Planning Report", ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(170, 170, 170)
    pdf.set_xy(10, 20)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(0, 7, f"Zone: {zone_id}  |  Generated: {ts}  |  Milestone 2 - Agentic AI", ln=True)

    # ── High-load badge ───────────────────────────────────────────────────────
    if metrics.get("is_high_load"):
        pdf.set_xy(148, 7)
        pdf.set_fill_color(220, 53, 69)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(52, 9, "  HIGH-LOAD ZONE  ", border=0, fill=True, align="C")

    pdf.ln(14)

    # ── Metric cards ──────────────────────────────────────────────────────────
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Key Demand Metrics", ln=True)
    pdf.ln(2)

    cards = [
        ("Avg Demand", f"{metrics.get('mean', 0):.2f} kWh"),
        ("Std Deviation", f"{metrics.get('std', 0):.2f} kWh"),
        ("Peak Threshold", f"{metrics.get('peak_threshold', 0):.2f} kWh"),
    ]
    card_w, card_h, gap = 56, 20, 5
    y0 = pdf.get_y()
    for idx, (label, val) in enumerate(cards):
        x0 = 10 + idx * (card_w + gap)
        pdf.set_fill_color(237, 247, 237)
        pdf.rect(x0, y0, card_w, card_h, "F")
        pdf.set_xy(x0 + 3, y0 + 3)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(29, 185, 84)
        pdf.cell(card_w - 6, 8, val)
        pdf.set_xy(x0 + 3, y0 + 12)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(card_w - 6, 6, label)

    pdf.set_xy(10, y0 + card_h + 6)

    # ── Divider ───────────────────────────────────────────────────────────────
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # ── Report body ───────────────────────────────────────────────────────────
    for line in report_text.split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(2)
            continue

        if line.startswith("### ") or line.startswith("## "):
            heading = _safe(line.lstrip("#").strip())
            pdf.set_fill_color(237, 247, 237)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(29, 185, 84)
            pdf.cell(0, 9, heading, fill=True, ln=True)
            pdf.ln(1)
        elif line.startswith("# "):
            heading = _safe(line.lstrip("#").strip())
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(30, 30, 47)
            pdf.cell(0, 10, heading, ln=True)
            pdf.ln(2)
        elif line.startswith(("- ", "* ")):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.set_x(15)
            pdf.multi_cell(0, 5, _safe("• " + line[2:]))
        elif line.startswith(">"):
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(120, 120, 120)
            pdf.set_x(15)
            pdf.multi_cell(0, 5, _safe(line.lstrip("> ")))
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(55, 55, 55)
            pdf.multi_cell(0, 5, _safe(line))

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 8, "EV Charging Demand Forecasting Dashboard  |  Milestone 2  |  Agentic AI Planner", align="C")

    return bytes(pdf.output())
