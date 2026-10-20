"""Command-line entry point for the RAG project."""

from __future__ import annotations

import argparse
from pathlib import Path


def _ingest(args: argparse.Namespace) -> None:
    from documents.write_milvus import run_ingestion

    run_ingestion(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        queue_size=args.queue_size,
    )


def _adaptive(args: argparse.Namespace) -> None:
    from graph_adaptive.adaptive_graph import answer, run_chat

    if args.query:
        print(answer(args.query))
    else:
        run_chat()


def _corrective(args: argparse.Namespace) -> None:
    from graph_corrective.corrective_graph import answer, run_chat

    if args.query:
        print(answer(args.query))
    else:
        run_chat()


def _agent(args: argparse.Namespace) -> None:
    from agent.rag_agent import run_agent

    response = run_agent(args.query, session_id=args.session_id)
    print(response)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-demo",
        description="Local RAG knowledge-base demo (Milvus Lite + LangGraph).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Parse Markdown files and write them to Milvus.")
    ingest.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing Markdown files (default: data/md).",
    )
    ingest.add_argument("--batch-size", type=int, default=20, help="Documents per write batch.")
    ingest.add_argument("--queue-size", type=int, default=20, help="Bounded queue size.")
    ingest.set_defaults(handler=_ingest)

    adaptive = subparsers.add_parser("adaptive", help="Run the Adaptive RAG workflow.")
    adaptive.add_argument("--query", type=str, default=None, help="Run one query and exit.")
    adaptive.set_defaults(handler=_adaptive)

    corrective = subparsers.add_parser("corrective", help="Run the Corrective RAG workflow.")
    corrective.add_argument("--query", type=str, default=None, help="Run one query and exit.")
    corrective.set_defaults(handler=_corrective)

    agent = subparsers.add_parser("agent", help="Run the tool-calling RAG agent.")
    agent.add_argument("--query", type=str, default="什么是EUV光刻机？")
    agent.add_argument("--session-id", type=str, default="default")
    agent.set_defaults(handler=_agent)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
