"""Downloadable PDF report of one prediction + explanation (FR-6.3)."""
from __future__ import annotations

import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import DISCLAIMER, FEATURE_INFO, FEATURES


def _p(text, style):
    return Paragraph(escape(str(text)), style)


def build_pdf(result: dict, contrib=None, chart_png: bytes | None = None,
              model_name: str = "", summary: str = "") -> bytes:
    """Return the PDF as bytes.

    result: dict with keys inputs (dict), probability (float), label (str).
    contrib: optional DataFrame from explain.build_contributions.
    """
    styles = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Diabetes risk report")
    story = [
        _p("Explainable Diabetes Risk Report", styles["Title"]),
        _p(f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')}"
           + (f"  |  Model: {model_name}" if model_name else ""), styles["Normal"]),
        Spacer(1, 0.4 * cm),
        _p(f"Result: {result['label']}  ({result['probability'] * 100:.1f}% estimated probability)",
           styles["Heading2"]),
    ]
    if summary:
        story += [_p(summary, styles["Normal"]), Spacer(1, 0.3 * cm)]

    # Inputs table
    story.append(_p("Values you entered", styles["Heading3"]))
    rows = [["Feature", "Value"]]
    for f in FEATURES:
        info = FEATURE_INFO[f]
        v = result["inputs"].get(f)
        unit = f" {info['unit']}" if info["unit"] else ""
        rows.append([info["label"], "not provided" if v is None else f"{v:g}{unit}"])
    t = Table(rows, colWidths=[7 * cm, 6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story += [t, Spacer(1, 0.4 * cm)]

    # Explanation
    if contrib is not None and len(contrib):
        story.append(_p("What influenced the result (SHAP)", styles["Heading3"]))
        if chart_png:
            story.append(Image(io.BytesIO(chart_png), width=16 * cm, height=9 * cm,
                               kind="proportional"))
        rows = [["Feature", "Your value", "Effect"]]
        for r in contrib.itertuples():
            rows.append([r.label, r.value_text, r.direction])
        t2 = Table(rows, colWidths=[5 * cm, 7 * cm, 4 * cm])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story += [Spacer(1, 0.2 * cm), t2]
    else:
        story.append(_p("Explanation unavailable for this report.", styles["Normal"]))

    story += [Spacer(1, 0.6 * cm), _p("Disclaimer: " + DISCLAIMER, styles["Italic"])]
    doc.build(story)
    return buf.getvalue()
