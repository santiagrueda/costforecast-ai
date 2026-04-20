"""Convert docs/informe.md to docs/informe.pdf using reportlab + markdown."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

DOCS_DIR = Path(__file__).parent
MD_FILE = DOCS_DIR / "informe.md"
PDF_FILE = DOCS_DIR / "informe.pdf"

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

BASE = getSampleStyleSheet()

STYLES: dict[str, ParagraphStyle] = {
    "h1": ParagraphStyle(
        "h1",
        parent=BASE["Heading1"],
        fontSize=20,
        leading=26,
        spaceAfter=12,
        textColor=colors.HexColor("#1a1a2e"),
        fontName="Helvetica-Bold",
    ),
    "h2": ParagraphStyle(
        "h2",
        parent=BASE["Heading2"],
        fontSize=15,
        leading=20,
        spaceBefore=18,
        spaceAfter=6,
        textColor=colors.HexColor("#16213e"),
        fontName="Helvetica-Bold",
        borderPad=4,
    ),
    "h3": ParagraphStyle(
        "h3",
        parent=BASE["Heading3"],
        fontSize=12,
        leading=16,
        spaceBefore=10,
        spaceAfter=4,
        textColor=colors.HexColor("#0f3460"),
        fontName="Helvetica-Bold",
    ),
    "body": ParagraphStyle(
        "body",
        parent=BASE["Normal"],
        fontSize=10,
        leading=15,
        spaceAfter=6,
        fontName="Helvetica",
    ),
    "bullet": ParagraphStyle(
        "bullet",
        parent=BASE["Normal"],
        fontSize=10,
        leading=14,
        leftIndent=16,
        spaceAfter=3,
        bulletIndent=6,
        fontName="Helvetica",
    ),
    "blockquote": ParagraphStyle(
        "blockquote",
        parent=BASE["Normal"],
        fontSize=9,
        leading=13,
        leftIndent=20,
        rightIndent=20,
        spaceAfter=6,
        textColor=colors.HexColor("#555555"),
        fontName="Helvetica-Oblique",
        backColor=colors.HexColor("#f5f5f5"),
        borderPad=4,
    ),
    "code_inline": ParagraphStyle(
        "code_inline",
        parent=BASE["Normal"],
        fontSize=9,
        fontName="Courier",
        backColor=colors.HexColor("#f0f0f0"),
        leading=13,
    ),
    "meta": ParagraphStyle(
        "meta",
        parent=BASE["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#666666"),
        fontName="Helvetica-Oblique",
        spaceAfter=4,
    ),
}

TABLE_HEADER_BG = colors.HexColor("#1a1a2e")
TABLE_ROW_ALT = colors.HexColor("#f0f4ff")
TABLE_GRID = colors.HexColor("#cccccc")

# ---------------------------------------------------------------------------
# Markdown → ReportLab flowables
# ---------------------------------------------------------------------------

def _inline(text: str) -> str:
    """Convert inline markdown (bold, italic, code, links) to ReportLab XML."""
    # Bold + italic
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", text)
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    # Code
    text = re.sub(r"`(.+?)`", r'<font face="Courier" size="9" color="#333333">\1</font>', text)
    # Markdown links [text](url) → just text (PDF doesn't need live links)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Arrows and special chars
    text = text.replace("↑", "↑").replace("↓", "↓").replace("→", "→").replace("←", "←")
    return text


def _parse_table(lines: list[str]) -> Table | None:
    """Parse a markdown table block into a ReportLab Table."""
    rows = []
    for line in lines:
        if re.match(r"\|[-:| ]+\|", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return None

    col_count = max(len(r) for r in rows)
    # Pad rows
    padded = [r + [""] * (col_count - len(r)) for r in rows]

    # Convert cells to Paragraphs
    table_data = []
    for i, row in enumerate(padded):
        style = STYLES["body"] if i > 0 else ParagraphStyle(
            "th", parent=STYLES["body"], fontName="Helvetica-Bold",
            textColor=colors.white, fontSize=9,
        )
        table_data.append([Paragraph(_inline(cell), style) for cell in row])

    available_w = PAGE_W - 2 * MARGIN
    col_w = available_w / col_count

    tbl = Table(table_data, colWidths=[col_w] * col_count, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TABLE_ROW_ALT]),
        ("GRID", (0, 0), (-1, -1), 0.4, TABLE_GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


def md_to_flowables(md_text: str) -> list:
    """Convert markdown text to a list of ReportLab flowables."""
    flowables: list = []
    lines = md_text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]

        # --- Heading ---
        if line.startswith("# ") and not line.startswith("## "):
            flowables.append(Paragraph(_inline(line[2:]), STYLES["h1"]))
            flowables.append(HRFlowable(width="100%", thickness=1.5,
                                        color=colors.HexColor("#1a1a2e"), spaceAfter=6))
            i += 1

        elif line.startswith("## "):
            flowables.append(Spacer(1, 4))
            flowables.append(Paragraph(_inline(line[3:]), STYLES["h2"]))
            i += 1

        elif line.startswith("### "):
            flowables.append(Paragraph(_inline(line[4:]), STYLES["h3"]))
            i += 1

        # --- Horizontal rule ---
        elif line.strip() == "---":
            flowables.append(HRFlowable(width="100%", thickness=0.5,
                                        color=colors.HexColor("#cccccc"),
                                        spaceBefore=6, spaceAfter=6))
            i += 1

        # --- Table ---
        elif line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            tbl = _parse_table(table_lines)
            if tbl:
                flowables.append(Spacer(1, 4))
                flowables.append(tbl)
                flowables.append(Spacer(1, 6))

        # --- Blockquote ---
        elif line.startswith("> "):
            flowables.append(Paragraph(_inline(line[2:]), STYLES["blockquote"]))
            i += 1

        # --- Bullet ---
        elif re.match(r"^[-*] ", line):
            text = re.sub(r"^[-*] ", "", line)
            flowables.append(Paragraph("• " + _inline(text), STYLES["bullet"]))
            i += 1

        elif re.match(r"^\d+\. ", line):
            text = re.sub(r"^\d+\. ", "", line)
            num = re.match(r"^(\d+)\.", line).group(1)
            flowables.append(Paragraph(f"{num}. " + _inline(text), STYLES["bullet"]))
            i += 1

        # --- Meta / italic-only lines (author, date) ---
        elif line.startswith("**") and line.endswith("**"):
            flowables.append(Paragraph(_inline(line), STYLES["meta"]))
            i += 1

        # --- Empty line ---
        elif line.strip() == "":
            flowables.append(Spacer(1, 4))
            i += 1

        # --- Normal paragraph ---
        else:
            # Collect continuation lines
            para_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", ">", "|", "-", "*")) and not re.match(r"^\d+\. ", lines[i]) and lines[i].strip() != "---":
                para_lines.append(lines[i])
                i += 1
            full = " ".join(para_lines)
            if full.strip():
                flowables.append(Paragraph(_inline(full), STYLES["body"]))

    return flowables


# ---------------------------------------------------------------------------
# Header / Footer
# ---------------------------------------------------------------------------

def _on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    # Footer
    canvas.drawString(MARGIN, 1.2 * cm, "CostForecast AI — Informe Ejecutivo")
    canvas.drawRightString(PAGE_W - MARGIN, 1.2 * cm, f"Página {doc.page}")
    # Top line
    canvas.setLineWidth(0.4)
    canvas.setStrokeColor(colors.HexColor("#cccccc"))
    canvas.line(MARGIN, PAGE_H - MARGIN + 0.4 * cm, PAGE_W - MARGIN, PAGE_H - MARGIN + 0.4 * cm)
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_pdf(md_path: Path = MD_FILE, pdf_path: Path = PDF_FILE) -> None:
    md_text = md_path.read_text(encoding="utf-8")

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN + 0.5 * cm,
        bottomMargin=MARGIN,
        title="CostForecast AI — Informe Ejecutivo",
        author="Santiago Rueda",
    )

    flowables = md_to_flowables(md_text)
    doc.build(flowables, onFirstPage=_on_page, onLaterPages=_on_page)
    print(f"PDF generado: {pdf_path}  ({pdf_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    generate_pdf()
