"""Generate a fictional annual report: prose, a ruled table, and one image-only "scanned" page."""

from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf

INTRO = (
    "Item 1. Business\n\n"
    "Northwind Harbor Logistics, Inc. is a fictional company created to test this software. "
    "It operates cold-chain freight terminals in three regions. The fiscal year ended "
    "December 31, 2025.\n\n"
    "Risk Factors\n\n"
    "Our largest customer accounted for 18.5% of total revenue in fiscal 2025. Fuel costs are "
    "our largest variable expense, and a sustained increase of 10% would reduce operating "
    "income by approximately $6.2 million."
)
MDA = (
    "Item 7. Management's Discussion and Analysis\n\n"
    "Total revenue increased to $842.6 million in fiscal 2025 from $761.3 million in fiscal "
    "2024, driven by the opening of the Rotterdam terminal. Capital expenditures were "
    "$57.4 million. The Company repurchased no shares during the year."
)
INCOME_ROWS = [
    ["(in millions, except per share)", "FY2025", "FY2024"],
    ["Total revenue", "842.6", "761.3"],
    ["Operating income", "118.9", "97.2"],
    ["Net income", "84.1", "66.8"],
    ["Diluted earnings per share", "3.27", "2.61"],
]
SCANNED_LINES = [
    "Consolidated Balance Sheets",
    "(in millions)                FY2025      FY2024",
    "Total assets                 1,204.7     1,098.2",
    "Total liabilities              612.3       587.9",
    "Net cash from operating activities  131.5  109.6",
]


def _draw_table(page: pymupdf.Page, rows: list[list[str]], top: float) -> None:
    widths, left, row_height = [260, 90, 90], 60, 24
    x_positions = [left]
    for width in widths:
        x_positions.append(x_positions[-1] + width)
    bottom = top + row_height * len(rows)
    for x in x_positions:
        page.draw_line((x, top), (x, bottom))
    for index in range(len(rows) + 1):
        y = top + row_height * index
        page.draw_line((left, y), (x_positions[-1], y))
    for row_index, row in enumerate(rows):
        for column, cell in enumerate(row):
            page.insert_text((x_positions[column] + 5, top + row_height * row_index + 16), cell, fontsize=10)


def build(path: Path) -> None:
    pdf = pymupdf.open()

    page = pdf.new_page()
    page.insert_textbox(pymupdf.Rect(60, 60, 540, 780), INTRO, fontsize=11)

    page = pdf.new_page()
    page.insert_textbox(pymupdf.Rect(60, 60, 540, 220), MDA, fontsize=11)
    page.insert_text((60, 245), "Consolidated Statements of Income", fontsize=11)
    _draw_table(page, INCOME_ROWS, 260)

    # Render text to a bitmap, so the page has an image and no text layer, as a scan does.
    source = pymupdf.open()
    scratch = source.new_page()
    for index, line in enumerate(SCANNED_LINES):
        scratch.insert_text((60, 80 + index * 26), line, fontname="cour", fontsize=12)
    image = scratch.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY).tobytes("png")
    page = pdf.new_page()
    page.insert_image(page.rect, stream=image)

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(path)
    pdf.close()
    source.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("samples/northwind_annual_report.pdf"))
    args = parser.parse_args()
    build(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
