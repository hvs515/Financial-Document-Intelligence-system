"""Fast tests: no model downloads. Hashing embeddings and a scripted LLM stand in for the models."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import httpx
import numpy as np
import pytest

from findoc import ingest
from findoc.chunking import _split_prose, _split_table, chunk_document
from findoc.config import Settings
from findoc.finetune.evaluate import is_correct
from findoc.finetune.prepare import convert, parse_query
from findoc.llm import OpenAICompatibleLLM
from findoc.models import Chunk, Document, Element, Hit
from findoc.pipeline import Pipeline
from findoc.qa import NOT_FOUND, answer_question, check_answer, extract_field
from findoc.stores import FaissStore, QdrantStore

ROOT = Path(__file__).resolve().parents[1]


class HashEmbedder:
    dim = 256

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class ScriptedLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[dict] = []

    def generate(self, messages: list[dict], max_new_tokens: int = 400) -> str:
        self.messages = messages
        return self.reply


def _hit(text: str, page: int = 1, source: str = "a.pdf") -> Hit:
    return Hit(Chunk(id=f"{source}{page}{text[:8]}", text=text, source=source, page=page), 1.0)


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    spec = importlib.util.spec_from_file_location("make_sample", ROOT / "scripts" / "make_sample.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path_factory.mktemp("samples") / "northwind.pdf"
    module.build(path)
    return path


@pytest.fixture
def no_tesseract(monkeypatch):
    def missing(settings):
        raise ingest.OCRUnavailable("Tesseract was not found.")

    monkeypatch.setattr(ingest, "find_tesseract", missing)


# --- ingestion ---


def test_pdf_extracts_prose_and_table_and_reports_scanned_page(sample_pdf, no_tesseract):
    doc = ingest.load_document(sample_pdf, Settings())
    tables = [element for element in doc.elements if element.kind == "table"]
    assert len(tables) == 1
    assert "842.6" in tables[0].text and "Diluted earnings per share" in tables[0].text
    assert tables[0].page == 2
    prose = " ".join(element.text for element in doc.elements if element.kind == "text")
    assert "18.5%" in prose
    # The table's cells must not also be indexed as loose prose.
    assert "Diluted earnings per share" not in prose
    assert doc.skipped_pages == [3]


def test_scanned_page_is_routed_to_ocr(sample_pdf, monkeypatch):
    seen = []

    def fake_ocr(image, settings):
        seen.append(image.size)
        return "Total assets 1,204.7"

    monkeypatch.setattr(ingest, "ocr_image", fake_ocr)
    doc = ingest.load_document(sample_pdf, Settings())
    ocr_elements = [element for element in doc.elements if element.ocr]
    assert [element.page for element in ocr_elements] == [3]
    assert doc.skipped_pages == []
    assert len(seen) == 1  # the native-text pages never reach OCR


def test_real_tesseract_reads_scanned_page(sample_pdf):
    try:
        ingest.find_tesseract(Settings())
    except ingest.OCRUnavailable:
        pytest.skip("Tesseract is not installed")
    doc = ingest.load_document(sample_pdf, Settings())
    text = " ".join(element.text for element in doc.elements if element.ocr)
    assert "1,204.7" in text and "612.3" in text and "131.5" in text


def test_table_uses_its_own_first_row_as_header_and_heading_is_metadata(sample_pdf, no_tesseract):
    doc = ingest.load_document(sample_pdf, Settings())
    table = next(element for element in doc.elements if element.kind == "table")
    assert table.text.splitlines()[0] == "|(in millions, except per share)|FY2025|FY2024|"
    chunks = chunk_document(doc)
    table_chunk = next(chunk for chunk in chunks if chunk.kind == "table")
    assert table_chunk.section == "Consolidated Statements of Income"
    assert "Consolidated Statements of Income" not in [chunk.text.strip() for chunk in chunks]


def test_blank_page_is_not_reported_as_scanned(tmp_path, no_tesseract):
    import pymupdf

    path = tmp_path / "blank.pdf"
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.save(path)
    pdf.close()
    doc = ingest.load_document(path, Settings())
    assert doc.elements == [] and doc.skipped_pages == []


def test_unstructured_reads_docx_tables(tmp_path):
    docx = pytest.importorskip("docx")
    pytest.importorskip("unstructured")
    path = tmp_path / "memo.docx"
    document = docx.Document()
    document.add_paragraph("Quarterly revenue commentary for the board of directors.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Metric", "Value"
    table.cell(1, 0).text, table.cell(1, 1).text = "Revenue", "42.5"
    document.save(path)
    doc = ingest.load_document(path, Settings())
    assert any(element.kind == "table" and "42.5" in element.text for element in doc.elements)
    assert any(element.kind == "text" and "commentary" in element.text for element in doc.elements)


def test_unsupported_type_is_rejected():
    with pytest.raises(ValueError):
        ingest.load_document(Path("report.xyz"), Settings())


# --- chunking ---


def test_chunks_never_span_pages_and_carry_sections():
    doc = Document(
        "a.pdf",
        [
            Element("text", "Item 7. Management's Discussion and Analysis", 1),
            Element("text", "Revenue grew.", 1),
            Element("text", "Costs fell.", 2),
        ],
    )
    chunks = chunk_document(doc)
    assert {chunk.page for chunk in chunks} == {1, 2}
    assert not any("Revenue" in chunk.text and "Costs" in chunk.text for chunk in chunks)
    assert all(chunk.section.startswith("Item 7") for chunk in chunks)


def test_prose_split_never_cuts_a_figure_and_covers_all_text():
    text = " ".join(f"Segment {i} revenue was $1,234.{i:02d} million." for i in range(100))
    pieces = _split_prose(text, 200, 40)
    assert all(len(piece) <= 200 for piece in pieces)
    joined = " ".join(pieces)
    for i in range(100):
        assert f"$1,234.{i:02d} million." in joined
    for piece in pieces:
        for number in re.findall(r"\d[\d,]*\.\d+", piece):
            assert re.fullmatch(r"1,234\.\d\d", number)


def test_prose_split_terminates_on_unbroken_text():
    pieces = _split_prose("x" * 5000, 200, 40)
    assert "".join(pieces).count("x") >= 5000


def test_oversized_table_repeats_header_in_every_part():
    header = "| Metric | FY25 |\n| --- | --- |"
    rows = "\n".join(f"| Row {i} | {i} |" for i in range(200))
    parts = _split_table(f"{header}\n{rows}", 300)
    assert len(parts) > 1
    assert all(part.startswith(header) for part in parts)
    assert sum(part.count("| Row ") for part in parts) == 200


# --- stores share one contract ---


@pytest.fixture(params=["faiss", "qdrant"])
def store(request, tmp_path):
    if request.param == "faiss":
        instance = FaissStore(tmp_path / "faiss", 3)
    else:
        pytest.importorskip("qdrant_client")
        instance = QdrantStore(3, "test", path=tmp_path / "qdrant")
    yield instance
    instance.close()


def _chunks(source: str, count: int) -> list[Chunk]:
    return [Chunk(id=f"{source}-{i}", text=f"{source} {i}", source=source, page=i + 1) for i in range(count)]


def test_store_search_ranks_filters_and_replaces(store):
    a = np.array([[1, 0, 0], [0.9, 0.436, 0]], dtype=np.float32)
    b = np.array([[0, 1, 0]], dtype=np.float32)
    store.replace_source("a.pdf", _chunks("a.pdf", 2), a)
    store.replace_source("b.pdf", _chunks("b.pdf", 1), b)

    query = np.array([1, 0, 0], dtype=np.float32)
    assert [hit.chunk.id for hit in store.search(query, 2)] == ["a.pdf-0", "a.pdf-1"]
    assert [hit.chunk.source for hit in store.search(query, 5, source="b.pdf")] == ["b.pdf"]
    assert store.sources() == {"a.pdf": 2, "b.pdf": 1}

    store.replace_source("a.pdf", _chunks("a.pdf", 1), a[:1])
    assert store.sources() == {"a.pdf": 1, "b.pdf": 1}
    store.replace_source("a.pdf", [], np.empty((0, 3), dtype=np.float32))
    assert store.sources() == {"b.pdf": 1}


def test_faiss_persists_across_reopen_and_rejects_wrong_dimension(tmp_path):
    store = FaissStore(tmp_path, 3)
    store.replace_source("a.pdf", _chunks("a.pdf", 1), np.array([[1, 0, 0]], dtype=np.float32))
    reopened = FaissStore(tmp_path, 3)
    assert reopened.search(np.array([1, 0, 0], dtype=np.float32), 1)[0].chunk.text == "a.pdf 0"
    with pytest.raises(ValueError):
        FaissStore(tmp_path, 4)


# --- answers ---


def test_grounding_check_treats_formatting_differences_as_the_same_number():
    hits = [_hit("Total assets were 1,204.70 million in 2025.")]
    valid, invalid, unverified = check_answer("Assets were 1204.7 million in 2025 [1].", hits)
    assert (valid, invalid, unverified) == ([1], [], [])


def test_grounding_check_flags_invented_numbers_and_bad_citations():
    hits = [_hit("Revenue was 842.6 million.")]
    valid, invalid, unverified = check_answer("Revenue was 900.1 million [1][7].", hits)
    assert valid == [1] and invalid == [7] and unverified == ["900.1"]


def test_grounding_check_flags_a_real_number_cited_to_the_wrong_excerpt():
    hits = [_hit("Total liabilities 612.3 587.9", 3), _hit("Net income 84.1", 2)]
    assert check_answer("Liabilities were 587.9 [2].", hits)[2] == ["587.9"]
    assert check_answer("Liabilities were 587.9 [1].", hits)[2] == []
    assert check_answer("Liabilities were 587.9.", hits)[2] == []  # uncited: any excerpt counts


def test_answer_without_hits_does_not_call_the_model():
    llm = ScriptedLLM("should not be used")
    answer = answer_question(llm, "What was revenue?", [])
    assert answer.text == NOT_FOUND and llm.messages == []


def test_prompt_numbers_excerpts_and_labels_pages_and_ocr():
    hit = _hit("Revenue was 5.", page=9)
    hit.chunk.ocr = True
    llm = ScriptedLLM("Revenue was 5 [1].")
    answer_question(llm, "What was revenue?", [hit])
    assert "[1] (a.pdf, page 9, OCR, may contain errors)" in llm.messages[1]["content"]


@pytest.mark.parametrize(
    "reply, status, value, page",
    [
        ('{"value": "842.6", "unit": "millions", "period": "FY2025", "excerpt": 2}', "ok", "842.6", 2),
        ('Sure! {"value": "999.9", "unit": null, "period": null, "excerpt": 2}', "unverified", "999.9", 2),
        ('{"value": "842.6", "excerpt": 9}', "bad_citation", "842.6", None),
        ('{"value": null, "excerpt": null}', "not_found", None, None),
        ("no json here", "unparseable", None, None),
    ],
)
def test_extract_field_statuses(reply, status, value, page):
    hits = [_hit("Unrelated.", 1), _hit("Total revenue 842.6", 2)]
    result = extract_field(ScriptedLLM(reply), "total revenue", hits)
    assert (result["status"], result["value"], result["page"]) == (status, value, page)


def test_openai_backend_sends_a_deterministic_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": " hi "}}]})

    client = httpx.Client(base_url="http://test/v1", transport=httpx.MockTransport(handler))
    llm = OpenAICompatibleLLM("http://test/v1", "qwen", client=client)
    assert llm.generate([{"role": "user", "content": "x"}], max_new_tokens=5) == "hi"
    assert captured["temperature"] == 0 and captured["max_tokens"] == 5


def test_pipeline_end_to_end_with_stub_models(sample_pdf, tmp_path, no_tesseract):
    settings = Settings(data_dir=tmp_path)
    llm = ScriptedLLM("Net income was 84.1 million [1].")
    pipeline = Pipeline(settings, embedder=HashEmbedder(), llm=llm)
    reports = pipeline.ingest([sample_pdf])
    assert reports[0].tables == 1 and reports[0].skipped_pages == [3]

    answer = pipeline.ask("net income diluted earnings per share")
    assert answer.hits[0].chunk.kind == "table" and answer.hits[0].chunk.page == 2
    assert answer.citations == [1] and answer.unverified_numbers == []

    # Re-ingesting replaces the document instead of duplicating it.
    pipeline.ingest([sample_pdf])
    assert pipeline.store.sources() == {sample_pdf.name: reports[0].chunks}


# --- fine-tuning data and scoring ---


def _finqa_row(question: str, context: str, answer: str) -> dict:
    query = (
        "Please answer the given financial question based on the context.\n"
        f"Context: {context}\nQuestion: {question}\nAnswer:"
    )
    return {"id": "x", "query": query, "answer": answer, "text": question}


def test_parse_query_handles_multiline_context():
    context, question = parse_query(_finqa_row("what?", "line one\n| a | b |", "1")["query"])
    assert context == "line one\n| a | b |" and question == "what?"


def test_convert_uses_the_production_prompt_and_adds_refusals():
    rows = [_finqa_row(f"question {i}?", f"context {i}", str(i)) for i in range(20)]
    examples = convert(rows, negatives=0.25)
    assert len(examples) == 25
    refusals = [e for e in examples if e["messages"][-1]["content"] == NOT_FOUND]
    assert len(refusals) == 5
    for example in refusals:
        user = example["messages"][1]["content"]
        number = re.search(r"question (\d+)\?", user).group(1)
        assert f"context {number}\n" not in user  # the evidence really is absent
    answered = next(e for e in examples if e["messages"][-1]["content"] != NOT_FOUND)
    assert answered["messages"][0]["role"] == "system"
    assert answered["messages"][-1]["content"].endswith("[1]")


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        ("The interest expense was $3.8 million [1]", "3.8 [1]", True),
        ("12.0% [1]", "0.12 [1]", True),
        ("1,204.7", "1204.7 [1]", True),
        ("3.9", "3.8 [1]", False),
        ("It was 3.8, then rose to 9.0", "3.8 [1]", False),  # the final number decides
        ("no figure here", "3.8 [1]", False),
        ("yes, it did [1]", "yes [1]", True),
        ("no [1]", "yes [1]", False),
        (NOT_FOUND, NOT_FOUND, True),
        ("3.8", NOT_FOUND, False),
    ],
)
def test_is_correct(prediction, gold, expected):
    assert is_correct(prediction, gold) is expected


def test_training_masks_the_prompt_and_drops_oversized_examples():
    pytest.importorskip("peft")
    from findoc.finetune.train import PadCollator, encode_example

    class Tokenizer:
        eos_token = "<e>"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return " ".join(message["content"] for message in messages)

        def __call__(self, text, add_special_tokens):
            return {"input_ids": list(range(len(text.split())))}

    messages = [{"role": "user", "content": "a b c"}, {"role": "assistant", "content": "d e"}]
    encoded = encode_example(Tokenizer(), messages, 100)
    assert encoded["labels"][:3] == [-100, -100, -100]
    assert all(label != -100 for label in encoded["labels"][3:])
    assert encode_example(Tokenizer(), messages, 3) is None

    batch = PadCollator(0)([encoded, {key: value[:2] for key, value in encoded.items()}])
    assert batch["input_ids"].shape == (2, len(encoded["input_ids"]))
    assert batch["labels"][1, -1] == -100 and batch["attention_mask"][1, -1] == 0
