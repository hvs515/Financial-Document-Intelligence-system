"""Turn files into Documents: PyMuPDF for PDFs, Tesseract for scans, Unstructured for the rest."""

from __future__ import annotations

import importlib
import io
import logging
import shutil
from pathlib import Path

import pymupdf
from PIL import Image

from .config import Settings
from .models import Document, Element

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_PARTITIONERS = {
    ".docx": ("docx", "partition_docx"),
    ".html": ("html", "partition_html"),
    ".htm": ("html", "partition_html"),
    ".txt": ("text", "partition_text"),
    ".md": ("md", "partition_md"),
    ".eml": ("email", "partition_email"),
}
SUPPORTED_SUFFIXES = {".pdf"} | IMAGE_SUFFIXES | set(_PARTITIONERS)

# The UB-Mannheim Windows installer does not add itself to PATH.
WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


class OCRUnavailable(RuntimeError):
    pass


def find_tesseract(settings: Settings) -> str:
    candidate = settings.tesseract_cmd or shutil.which("tesseract")
    if candidate:
        return candidate
    if WINDOWS_TESSERACT.exists():
        return str(WINDOWS_TESSERACT)
    raise OCRUnavailable(
        "Tesseract was not found. Install it, or set FINDOC_TESSERACT_CMD to its path."
    )


def ocr_image(image: Image.Image, settings: Settings) -> str:
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = find_tesseract(settings)
    # psm 6 reads the page as one block, which keeps statement rows on one line.
    config = "--psm 6 -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(image, lang=settings.ocr_lang, config=config).strip()


def _overlaps(a: pymupdf.Rect, b: pymupdf.Rect) -> bool:
    """True when most of block `a` sits inside table `b`."""
    area = a.get_area()
    return area > 0 and (a & b).get_area() / area > 0.5


def _table_markdown(table) -> str:
    """Render a table, keeping its own first row as the header.

    Not table.extract(): it can corrupt cells, for example turning 84.1 into "841\\n.".
    to_markdown() reads cells correctly, but may promote text above the table to a header row.
    """
    lines = table.to_markdown().strip().splitlines()
    if table.header.external and len(lines) > 3:
        lines = [lines[2], lines[1], *lines[3:]]
    return "\n".join(lines)


def _load_pdf(path: Path, settings: Settings) -> Document:
    doc = Document(source=path.name)
    with pymupdf.open(path) as pdf:
        for number, page in enumerate(pdf, start=1):
            native_chars = len(page.get_text("text").strip())
            if native_chars < settings.ocr_min_chars and page.get_images():
                pixmap = page.get_pixmap(dpi=settings.ocr_dpi, colorspace=pymupdf.csGRAY)
                image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                try:
                    text = ocr_image(image, settings)
                except OCRUnavailable as exc:
                    log.warning("%s page %d looks scanned and was skipped: %s", path.name, number, exc)
                    doc.skipped_pages.append(number)
                    continue
                if text:
                    doc.elements.append(Element("text", text, number, ocr=True))
                continue

            tables = page.find_tables().tables
            boxes = [pymupdf.Rect(table.bbox) for table in tables]
            placed = [(box.y0, Element("table", _table_markdown(table), number))
                      for box, table in zip(boxes, tables)]
            for block in page.get_text("blocks", sort=True):
                rect, text, block_type = pymupdf.Rect(block[:4]), block[4].strip(), block[6]
                if block_type != 0 or not text or any(_overlaps(rect, box) for box in boxes):
                    continue
                placed.append((rect.y0, Element("text", text, number)))
            doc.elements.extend(element for _, element in sorted(placed, key=lambda p: p[0]))
    return doc


def _load_image(path: Path, settings: Settings) -> Document:
    doc = Document(source=path.name)
    try:
        with Image.open(path) as image:
            text = ocr_image(image.convert("L"), settings)
    except OCRUnavailable as exc:
        log.warning("%s was skipped: %s", path.name, exc)
        doc.skipped_pages.append(1)
        return doc
    if text:
        doc.elements.append(Element("text", text, 1, ocr=True))
    return doc


def _load_with_unstructured(path: Path) -> Document:
    # Not partition.auto: its file-type detection imports python-magic, which can hang at import
    # on Windows. The extension already says which partitioner applies.
    module_name, function_name = _PARTITIONERS[path.suffix.lower()]
    try:
        module = importlib.import_module(f"unstructured.partition.{module_name}")
    except ImportError as exc:
        raise RuntimeError(
            f"Reading {path.suffix} files needs Unstructured: pip install 'findoc[office]'"
        ) from exc

    doc = Document(source=path.name)
    for element in getattr(module, function_name)(filename=str(path)):
        text = (element.text or "").strip()
        if not text:
            continue
        page = element.metadata.page_number or 1
        if element.category == "Table":
            doc.elements.append(Element("table", element.metadata.text_as_html or text, page))
        else:
            doc.elements.append(Element("text", text, page))
    return doc


def load_document(path: Path, settings: Settings) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, settings)
    if suffix in IMAGE_SUFFIXES:
        return _load_image(path, settings)
    if suffix in _PARTITIONERS:
        return _load_with_unstructured(path)
    raise ValueError(f"Unsupported file type: {path.name}")


def iter_paths(paths: list[Path]) -> list[Path]:
    """Expand directories into the supported files they contain."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES))
        else:
            found.append(path)
    return found
