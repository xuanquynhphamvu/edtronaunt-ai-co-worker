from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "exports" / "gucci-module-2-1.pdf"


TITLE_PREFIX = "2.1"
TITLE_TEXT = 'Module 1 — Frame the leadership problem & define “Group DNA” (35–45 min)'
ASK_ITEMS = [
    (
        "Write a <b>problem statement</b> balancing brand autonomy with Group needs "
        "(mobility, pipeline)."
    ),
    (
        "Talk to Gucci Group's CEO to gain more insights about the business, Mission, "
        "company culture and brand and group's DNA."
    ),
    (
        "Talk to Gucci Group's CHRO for direction about Group HR's mission and the "
        "Competency Framework (Vision, Entrepreneurship, Passion, Trust) in order to "
        "craft a <b>first-pass competency model</b> (4 themes + behavior indicators by "
        "level) referencing <b>Vision, Entrepreneurship, Passion, Trust</b>."
    ),
    (
        "Map <b>use cases</b> for the model (for example: recruitment, appraisal, "
        "development, mobility)."
    ),
]
FOOTER_LINES = [
    "<b>Inputs:</b> Gucci context; brand list; examples of leadership behaviors.",
    "<b>Deliverables:</b> 1-page problem statement; competency matrix (CSV) with 4 themes × 3 levels; 10-slide CEO pack.",
    "<b>Grounding:</b> The case emphasizes the bespoke framework and the four headline themes.",
]


def build_pdf(output_path: Path) -> Path:
    try:
        from reportlab.lib.colors import HexColor
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.pdfbase.pdfmetrics import stringWidth
        from reportlab.platypus import Paragraph
        from reportlab.pdfgen.canvas import Canvas
    except ImportError as exc:
        raise RuntimeError("ReportLab is required to generate this PDF.") from exc

    page_width = 15 * inch
    page_height = 10 * inch
    margin_left = 0.55 * inch
    margin_right = 0.5 * inch
    margin_top = 0.38 * inch
    margin_bottom = 0.45 * inch

    blue = HexColor("#2D61F3")
    text = HexColor("#22252A")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(output_path), pagesize=(page_width, page_height), pageCompression=0)
    canvas.setTitle("Gucci Group Module 2.1")
    canvas.setAuthor("Codex")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="SlideTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=27,
        leading=33,
        textColor=text,
    )
    section_style = ParagraphStyle(
        name="Section",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        textColor=text,
    )
    item_style = ParagraphStyle(
        name="Item",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=16,
        leading=22,
        textColor=text,
    )
    footer_style = ParagraphStyle(
        name="Footer",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=14.5,
        leading=19,
        textColor=text,
    )

    def draw_paragraph(text_value: str, style: ParagraphStyle, x: float, y_top: float, width: float) -> float:
        paragraph = Paragraph(text_value, style)
        needed_width, needed_height = paragraph.wrap(width, page_height)
        paragraph.drawOn(canvas, x, y_top - needed_height)
        return y_top - needed_height

    canvas.setFillColor(blue)
    canvas.setFont("Helvetica-Bold", 28)
    prefix_width = stringWidth(TITLE_PREFIX, "Helvetica-Bold", 28)
    title_x = margin_left + prefix_width + 10
    title_width = page_width - title_x - margin_right
    title_top = page_height - margin_top
    canvas.drawString(margin_left, title_top - 28, TITLE_PREFIX)
    draw_paragraph(TITLE_TEXT, title_style, title_x, title_top, title_width)

    cursor_y = title_top - 96
    cursor_y = draw_paragraph("Ask", section_style, margin_left, cursor_y, page_width - margin_left - margin_right) - 20

    number_font = ("Helvetica", 16)
    item_x = margin_left + 34
    item_width = page_width - item_x - margin_right

    for index, item in enumerate(ASK_ITEMS, start=1):
        canvas.setFillColor(blue)
        canvas.setFont(number_font[0], number_font[1])
        canvas.drawString(margin_left + 2, cursor_y - 16, f"{index}.")
        canvas.setFillColor(text)
        item_bottom = draw_paragraph(item, item_style, item_x, cursor_y, item_width)
        cursor_y = item_bottom - 22

    cursor_y -= 18
    for footer in FOOTER_LINES:
        cursor_y = draw_paragraph(footer, footer_style, margin_left, cursor_y, page_width - margin_left - margin_right) - 8

    canvas.save()
    return output_path


def main() -> int:
    path = build_pdf(OUTPUT_PATH)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
