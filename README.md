# findoc: Financial Document Intelligence

Ask questions of financial documents and get answers that cite the page they came from, or pull key figures out as JSON. Runs locally with open models.

| Component | Technology | Where |
|---|---|---|
| Document ingestion | PyMuPDF, Unstructured | `src/findoc/ingest.py`, `chunking.py` |
| OCR | Tesseract | `src/findoc/ingest.py` |
| LLM | Qwen or Llama, local or over an OpenAI-compatible server | `src/findoc/llm.py`, `qa.py` |
| Fine-tuning | Hugging Face, PEFT, LoRA/QLoRA | `src/findoc/finetune/` |
| Retrieval | FAISS or Qdrant, BGE embeddings | `src/findoc/stores.py`, `embedding.py` |

## How it works

```
file -> parse (PDF text and tables, OCR for scans) -> one-page chunks -> embeddings -> vector store
question -> top excerpts -> LLM -> answer with [n] citations -> checks -> warnings
```

Design choices made for financial documents:

- **Chunks never cross a page**, so a citation names an exact page.
- **Tables stay whole.** A table too big for one chunk is split by row, with its header repeated.
- **Unreadable pages are reported.** A scanned page that can't be read is listed as not indexed, and `ingest` exits with code 2.
- **Answers are checked.** Numbers must appear in the excerpts the answer cites, otherwise the CLI warns. This catches some errors, not all: it can't catch the right number from the wrong row of a cited table.
- **Training uses the production prompt**, because both import it from `qa.py`.

## Setup

Requires Python 3.10+. On Windows, keep the virtual environment on a short path, because the install can exceed the Windows path limit.

```bash
python -m venv ~/.venvs/findoc
~/.venvs/findoc/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
~/.venvs/findoc/Scripts/python -m pip install -e ".[llm,finetune,qdrant,office,dev]"
```

The second line installs CPU-only PyTorch. Skip it on a GPU machine. On Linux or macOS, the path is `bin/python`.

For scanned documents, install Tesseract. On Windows:

```bash
winget install --id UB-Mannheim.TesseractOCR --exact
```

## Usage

```bash
python scripts/make_sample.py
findoc ingest samples
findoc sources
findoc search "net income"
findoc ask "What was net income in FY2025?"
findoc extract northwind_annual_report.pdf --out fields.json
```

Use `findoc --store qdrant ...` for Qdrant. Each store has its own index, so ingest into the one you plan to query.

Settings come from `FINDOC_*` environment variables (see `src/findoc/config.py`):

| Goal | Variables |
|---|---|
| Use a larger model | `FINDOC_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct` |
| Use a trained adapter | `FINDOC_LLM_ADAPTER=outputs/qwen-finqa` |
| Use Ollama, vLLM or a llama.cpp server | `FINDOC_LLM_BACKEND=openai`, `FINDOC_LLM_BASE_URL`, `FINDOC_LLM_MODEL` |
| Use a Qdrant server | `FINDOC_STORE=qdrant`, `FINDOC_QDRANT_URL` |

Llama models also work, but Meta gates them on Hugging Face, so accept the license there first.

## Fine-tuning

QLoRA needs a CUDA GPU. Without one, use [notebooks/finetune_colab.ipynb](notebooks/finetune_colab.ipynb).

```bash
python -m findoc.finetune.prepare --out data/finetune --negatives 0.1
python -m findoc.finetune.train --mode qlora --train data/finetune/train.jsonl --eval data/finetune/valid.jsonl --output outputs/qwen-finqa
python -m findoc.finetune.evaluate --load-in-4bit --model Qwen/Qwen2.5-7B-Instruct --adapter outputs/qwen-finqa --test data/finetune/test.jsonl
```

Keep an adapter only if `evaluate` shows it beating the base model on accuracy, refusal rate and citation rate.

Training data is FinQA, from the `ChanceFocus/flare-finqa` mirror. That mirror lists no license, so check the original FinQA terms before any commercial use.

At the default 2,048-token limit, 98.2% of a 1,500-example sample fits. Longer examples are dropped, not truncated.

## Tests

```bash
python -m pytest
```

## Current status

Tested on one 3-page fictional sample, on CPU, with Qwen 1.5B.

| Area | Status |
|---|---|
| PDF text and ruled tables | Works on the sample |
| OCR | Works on a clean rendered page; not tried on a noisy real scan |
| `.docx` through Unstructured | Works in a test; `.html`, `.txt`, `.md` and `.eml` are wired up but untested |
| FAISS and Qdrant | Both work |
| LoRA training | 2-step run on CPU completes and saves an adapter; that proves the code runs, not that it helps |
| Adapter loading and evaluation | Runs on CPU, but on too few examples to say anything about quality |
| QLoRA and the Colab notebook | **Not run**, as this machine has no GPU |
| Qwen 7B and Llama | **Not run** |
| OpenAI-compatible backend | Tested only against a mocked server |

Known problems:

- **Wrong figures are possible.** Qwen 1.5B answered one question with the prior-year column and a wrong citation. Don't rely on answers without checking the cited page.
- **Citations aren't reliable.** Qwen 1.5B sometimes leaves them out, and the CLI warns when it does.
- **Tables without ruled lines**, common in real filings, are probably read as plain text rather than tables.
- **Sections** are found with a regex written for 10-K-style headings.
