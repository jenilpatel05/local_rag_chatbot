from __future__ import annotations

import argparse
import sys


def _cmd_ingest(args: argparse.Namespace) -> int:
    from rag_chatbot.ingest import ingest_source
    rc = 0
    for target in args.targets:
        print(f"[ingest] {target} ...", end=" ", flush=True)
        try:
            pages, chunks = ingest_source(target)
            print(f"{pages} pages -> {chunks} new chunks stored")
        except Exception as e:
            print(f"FAILED: {e}")
            rc = 1
    return rc


def _cmd_ask(args: argparse.Namespace) -> int:
    from rag_chatbot.rag_chain import query as rag_query
    res = rag_query(args.question, model=args.model, top_k=args.top_k)
    print(f"\nAnswer:\n{res.answer}\n")
    print("Sources:")
    for s in res.sources:
        print(f"  - {s['source']}  page {s['page']}")
    return 0


def _cmd_list(_: argparse.Namespace) -> int:
    from rag_chatbot.ingest import list_ingested_sources
    sources = list_ingested_sources()
    if not sources:
        print("(no ingested documents)")
        return 0
    for s in sources:
        print(s)
    return 0


def _cmd_delete(args: argparse.Namespace) -> int:
    from rag_chatbot.ingest import delete_source
    n = delete_source(args.source)
    print(f"Removed {n} chunks for source: {args.source}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from rag_chatbot.evaluate import create_sample_qa, run_evaluation
    if args.create_sample:
        create_sample_qa()
        return 0
    run_evaluation(args.qa_path, model=args.model, top_k=args.top_k)
    return 0


def main(argv: list[str] | None = None) -> int:
    from rag_chatbot.config import LLM_MODEL, TOP_K

    parser = argparse.ArgumentParser(prog="rag-chatbot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Ingest files or URLs")
    p_ing.add_argument("targets", nargs="+")
    p_ing.set_defaults(func=_cmd_ingest)

    p_ask = sub.add_parser("ask", help="Query the RAG pipeline")
    p_ask.add_argument("question")
    p_ask.add_argument("--model", default=LLM_MODEL)
    p_ask.add_argument("--top_k", type=int, default=TOP_K)
    p_ask.set_defaults(func=_cmd_ask)

    p_list = sub.add_parser("list", help="List ingested sources")
    p_list.set_defaults(func=_cmd_list)

    p_del = sub.add_parser("delete", help="Delete a source by filename")
    p_del.add_argument("source")
    p_del.set_defaults(func=_cmd_delete)

    p_eval = sub.add_parser("eval", help="Run RAGAS evaluation")
    p_eval.add_argument("--qa_path", default="test_questions.json")
    p_eval.add_argument("--model", default=LLM_MODEL)
    p_eval.add_argument("--top_k", type=int, default=TOP_K)
    p_eval.add_argument("--create_sample", action="store_true")
    p_eval.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
