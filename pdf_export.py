"""
pdf_export.py
Generate a styled PDF planning report using fpdf2.
Returns raw bytes for use with st.download_button.
"""

import re
from datetime import datetime
from fpdf import FPDF

# Page layout constants (A4: 210 x 297 mm)
LEFT_MARGIN = 15
RIGHT_MARGIN = 15
PAGE_W = 210
CONTENT_W = PAGE_W - LEFT_MARGIN - RIGHT_MARGIN   # 180 mm
INDENT_W = CONTENT_W - 8                           # 172 mm (for bullets/quotes)


def _safe(text: str) -> str:
    """Strip markdown syntax and coerce to latin-1 for fpdf2 Helvetica."""
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
    return safe.strip()


def _body_line(pdf: FPDF, line: str):
    """Render a single body line, always from the left margin."""
    line = line.strip()
    if not line:
        pdf.ln(2)
        return

    # Reset to left margin before every element — prevents x-drift
    pdf.set_x(LEFT_MARGIN)

    if line.startswith("### ") or line.startswith("## "):
        heading = _safe(line.lstrip("#").strip())
        if not heading:
            return
        pdf.set_fill_color(237, 247, 237)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(29, 185, 84)
        pdf.set_x(LEFT_MARGIN)
        pdf.cell(CONTENT_W, 9, heading, fill=True, ln=True)
        pdf.ln(1)

    elif line.startswith("# "):
        heading = _safe(line.lstrip("#").strip())
        if not heading:
            return
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(30, 30, 47)
        pdf.set_x(LEFT_MARGIN)
        pdf.cell(CONTENT_W, 10, heading, ln=True)
        pdf.ln(2)

    elif line.startswith(("- ", "* ")):
        text = _safe("- " + line[2:])
        if not text.strip():
            return
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.set_x(LEFT_MARGIN + 5)
        pdf.multi_cell(INDENT_W, 5, text)

    elif line.startswith(">"):
        text = _safe(line.lstrip("> ").strip())
        if not text:
            return
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(120, 120, 120)
        pdf.set_x(LEFT_MARGIN + 5)
        pdf.multi_cell(INDENT_W, 5, text)

    else:
        text = _safe(line)
        if not text:
            return
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(55, 55, 55)
        pdf.set_x(LEFT_MARGIN)
        pdf.multi_cell(CONTENT_W, 5, text)


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
    pdf.set_margins(LEFT_MARGIN, 10, RIGHT_MARGIN)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Dark header banner ────────────────────────────────────────────────────
    pdf.set_fill_color(30, 30, 47)
    pdf.rect(0, 0, PAGE_W, 38, "F")

    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(29, 185, 84)
    pdf.set_xy(LEFT_MARGIN, 7)
    pdf.cell(CONTENT_W, 10, "EV Infrastructure Planning Report", ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(170, 170, 170)
    pdf.set_xy(LEFT_MARGIN, 20)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(CONTENT_W, 7, f"Zone: {_safe(zone_id)}  |  Generated: {ts}  |  Milestone 2 - Agentic AI", ln=True)

    # ── High-load badge (right side of header) ───────────────────────────────
    if metrics.get("is_high_load"):
        pdf.set_fill_color(220, 53, 69)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_xy(152, 8)
        pdf.cell(43, 8, "HIGH-LOAD ZONE", border=0, fill=True, align="C")

    # Move cursor below the header band
    pdf.set_xy(LEFT_MARGIN, 44)

    # ── "Key Demand Metrics" label ────────────────────────────────────────────
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(CONTENT_W, 7, "Key Demand Metrics", ln=True)
    pdf.ln(2)

    # ── Metric cards (3 fixed-width boxes side-by-side) ───────────────────────
    cards = [
        ("Avg Demand",     f"{metrics.get('mean', 0):.2f} kWh"),
        ("Std Deviation",  f"{metrics.get('std', 0):.2f} kWh"),
        ("Peak Threshold", f"{metrics.get('peak_threshold', 0):.2f} kWh"),
    ]
    card_w, card_h, gap = 55, 20, 5
    y0 = pdf.get_y()

    for idx, (label, val) in enumerate(cards):
        x0 = LEFT_MARGIN + idx * (card_w + gap)
        # Card background
        pdf.set_fill_color(237, 247, 237)
        pdf.rect(x0, y0, card_w, card_h, "F")
        # Value
        pdf.set_xy(x0 + 3, y0 + 2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(29, 185, 84)
        pdf.cell(card_w - 6, 8, _safe(val), ln=False)
        # Label
        pdf.set_xy(x0 + 3, y0 + 11)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(card_w - 6, 6, _safe(label), ln=False)

    # Advance cursor below the cards
    pdf.set_xy(LEFT_MARGIN, y0 + card_h + 5)

    # ── Divider ───────────────────────────────────────────────────────────────
    pdf.set_draw_color(200, 200, 200)
    pdf.line(LEFT_MARGIN, pdf.get_y(), PAGE_W - RIGHT_MARGIN, pdf.get_y())
    pdf.ln(6)

    # ── Report body ───────────────────────────────────────────────────────────
    for line in report_text.split("\n"):
        try:
            _body_line(pdf, line)
        except Exception:
            # Skip any single line that causes a rendering error
            pdf.set_x(LEFT_MARGIN)
            continue

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.set_x(LEFT_MARGIN)
    pdf.cell(
        CONTENT_W, 8,
        "EV Charging Demand Forecasting Dashboard  |  Milestone 2  |  Agentic AI Planner",
        align="C",
    )

    return bytes(pdf.output())
