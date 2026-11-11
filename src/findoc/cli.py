"""Command line: findoc ingest | search | ask | extract | sources."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import Settings
from .models import Hit
from .pipeline import Pipeline


def _print_hits(hits: list[Hit], numbered_from: int = 1) -> None:
    for index, hit in enumerate(hits, start=numbered_from):
        chunk = hit.chunk
        where = f"{chunk.source} p.{chunk.page}"
        if chunk.section:
            where += f" | {chunk.section}"
        if chunk.ocr:
            where += " | OCR"
        print(f"[{index}] {where} (score {hit.score:.3f}, {chunk.kind})")


def _cmd_ingest(pipeline: Pipeline, args: argparse.Namespace) -> int:
    reports = pipeline.ingest(args.paths)
    if not reports:
        print("No supported files found.")
        return 1
    incomplete = False
    for report in reports:
        print(
            f"{report.source}: {report.chunks} chunks "
            f"({report.tables} tables, {report.ocr_chunks} from OCR)"
        )
        if report.skipped_pages:
            incomplete = True
            print(f"  NOT INDEXED, scanned pages without OCR: {report.skipped_pages}")
    return 2 if incomplete else 0


def _cmd_search(pipeline: Pipeline, args: argparse.Namespace) -> int:
    hits = pipeline.search(args.query, args.k, args.source)
    for index, hit in enumerate(hits, start=1):
        _print_hits([hit], index)
        print(hit.chunk.text[:500] + "\n")
    return 0


def _cmd_ask(pipeline: Pipeline, args: argparse.Namespace) -> int:
    answer = pipeline.ask(args.question, args.k, args.source)
    print(answer.text + "\n")
    cited = [answer.hits[index - 1] for index in answer.citations]
    if cited:
        print("Sources:")
        for index, hit in zip(answer.citations, cited):
            _print_hits([hit], index)
    elif answer.hits and "not found" not in answer.text.lower():
        print("WARNING: the answer cites no source.")
    if answer.invalid_citations:
        print(f"WARNING: citations to excerpts that do not exist: {answer.invalid_citations}")
    if answer.unverified_numbers:
        print(
            "WARNING: numbers not found verbatim in the cited sources (a calculation or an error): "
            + ", ".join(answer.unverified_numbers)
        )
    return 0


def _cmd_extract(pipeline: Pipeline, args: argparse.Namespace) -> int:
    results = pipeline.extract(args.source, args.field or None)
    output = json.dumps(results, indent=2)
    if args.out:
        args.out.write_text(output, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(output)
    return 0


def _cmd_sources(pipeline: Pipeline, args: argparse.Namespace) -> int:
    for source, count in sorted(pipeline.store.sources().items()):
        print(f"{source}: {count} chunks")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="findoc", description=__doc__)
    parser.add_argument("--store", choices=["faiss", "qdrant"], help="vector store override")
    parser.add_argument("--data-dir", type=Path, help="where indexes live")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="parse and index files or folders")
    ingest.add_argument("paths", nargs="+", type=Path)
    ingest.set_defaults(run=_cmd_ingest)

    search = commands.add_parser("search", help="show retrieved passages, without the LLM")
    search.add_argument("query")
    search.add_argument("-k", type=int)
    search.add_argument("--source")
    search.set_defaults(run=_cmd_search)

    ask = commands.add_parser("ask", help="answer a question with citations")
    ask.add_argument("question")
    ask.add_argument("-k", type=int)
    ask.add_argument("--source")
    ask.set_defaults(run=_cmd_ask)

    extract = commands.add_parser("extract", help="pull key fields from one document as JSON")
    extract.add_argument("source", help="file name as shown by `findoc sources`")
    extract.add_argument("--field", action="append", help="repeatable; defaults to core financials")
    extract.add_argument("--out", type=Path)
    extract.set_defaults(run=_cmd_extract)

    sources = commands.add_parser("sources", help="list indexed documents")
    sources.set_defaults(run=_cmd_sources)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    settings = Settings.from_env()
    if args.store:
        settings.store = args.store
    if args.data_dir:
        settings.data_dir = args.data_dir

    pipeline = Pipeline(settings)
    try:
        return args.run(pipeline, args)
    finally:
        pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())
