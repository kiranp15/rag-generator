#!/usr/bin/env python3
"""Command-line interface for the RAG Generator.

Examples:
    python cli.py ingest --collection handbook --path sample_docs
    python cli.py ask --collection handbook --question "How many vacation days?"
    python cli.py list
    python cli.py delete --collection handbook
"""
from __future__ import annotations

import argparse
import os
import sys

from app.document_loader import SUPPORTED_EXTENSIONS
from app.rag_pipeline import RagPipeline
from app.vector_store import CollectionNotFoundError


def _collect_files(path: str) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    if os.path.isfile(path):
        with open(path, "rb") as f:
            files[os.path.basename(path)] = f.read()
        return files

    for root, _, names in os.walk(path):
        for name in names:
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
                full = os.path.join(root, name)
                with open(full, "rb") as f:
                    files[name] = f.read()
    return files


def cmd_ingest(args: argparse.Namespace) -> None:
    pipeline = RagPipeline()
    files = _collect_files(args.path)
    if not files:
        print(f"No supported files found under {args.path} "
              f"(supported: {', '.join(SUPPORTED_EXTENSIONS)})")
        sys.exit(1)

    result = pipeline.ingest(args.collection, files)
    print(f"Ingested {len(result.files_ingested)} file(s) into '{result.collection_id}': "
          f"{', '.join(result.files_ingested)}")
    print(f"Added {result.chunks_added} chunks (collection now has {result.total_chunks} total).")


def cmd_ask(args: argparse.Namespace) -> None:
    pipeline = RagPipeline()
    try:
        result = pipeline.query(args.collection, args.question, top_k=args.top_k)
    except CollectionNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n=== Answer ===")
    print(result.answer)
    print("\n=== Sources ===")
    for c in result.raw_chunks:
        print(f"- {c.source} (chunk {c.chunk_index}, score {c.score:.3f})")


def cmd_list(args: argparse.Namespace) -> None:
    pipeline = RagPipeline()
    collections = pipeline.list_collections()
    if not collections:
        print("No collections yet.")
        return
    for c in collections:
        print(c)


def cmd_delete(args: argparse.Namespace) -> None:
    pipeline = RagPipeline()
    pipeline.delete_collection(args.collection)
    print(f"Deleted collection '{args.collection}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Generator CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a file or directory of files into a collection.")
    p_ingest.add_argument("--collection", required=True)
    p_ingest.add_argument("--path", required=True, help="File or directory to ingest.")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a grounded question against a collection.")
    p_ask.add_argument("--collection", required=True)
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_ask.set_defaults(func=cmd_ask)

    p_list = sub.add_parser("list", help="List existing collections.")
    p_list.set_defaults(func=cmd_list)

    p_delete = sub.add_parser("delete", help="Delete a collection.")
    p_delete.add_argument("--collection", required=True)
    p_delete.set_defaults(func=cmd_delete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
