"""Render saved report content in an optional ReportLab-capable interpreter."""

import json
import sys
import reportlab
from html import escape
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)
from reportlab.lib.utils import ImageReader


def render(directory):
    directory = Path(directory)
    pages = json.loads((directory / "report_content.json").read_text(encoding="utf-8"))
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyReport",
            fontName="Helvetica",
            fontSize=9.4,
            leading=12.3,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TitleReport",
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            spaceAfter=14,
            textColor=colors.HexColor("#17324d"),
        )
    )
    doc = SimpleDocTemplate(
        str(directory / "report.pdf"),
        pagesize=(612, 792),
        rightMargin=43,
        leftMargin=43,
        topMargin=42,
        bottomMargin=40,
    )
    story = []
    for i, page in enumerate(pages):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(escape(page["title"]), styles["TitleReport"]))
        for text in page.get("paragraphs", []):
            story.append(Paragraph(escape(text), styles["BodyReport"]))
        if page.get("table"):
            data = [
                [Paragraph(escape(str(x)), styles["BodyReport"]) for x in row]
                for row in page["table"]
            ]
            n = len(data[0])
            widths = [(526 / n) for _ in range(n)]
            table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5eef5")),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#17324d")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.extend([table, Spacer(1, 8)])
        figures = page.get("figures", [])
        for f in figures:
            p = directory / "figures" / Path(f).name
            iw, ih = ImageReader(str(p)).getSize()
            maxh = 155 if len(figures) > 1 else (320 if i == 6 else 245)
            scale = min(510 / iw, maxh / ih)
            story.extend(
                [Image(str(p), width=iw * scale, height=ih * scale), Spacer(1, 6)]
            )

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#5b6770"))
        canvas.drawString(43, 22, "CMPE 249 | Track A | verified local experiments")
        canvas.drawRightString(569, 22, str(doc.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    from pypdf import PdfReader

    pdf = PdfReader(directory / "report.pdf")
    (directory / "render_backend.json").write_text(
        json.dumps(
            {
                "renderer": "ReportLab",
                "version": reportlab.Version,
                "python": sys.executable,
                "pages": len(pdf.pages),
                "content_sections": len(pages),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert 6 <= len(pdf.pages) <= 10, f"Report must be 6-10 pages, got {len(pdf.pages)}"


if __name__ == "__main__":
    render(sys.argv[1])
