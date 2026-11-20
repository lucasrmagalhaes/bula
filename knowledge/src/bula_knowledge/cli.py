"""Linha de comando do serviço: `bula-knowledge <comando>`."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bula_knowledge.config import REPO_ROOT, Settings, get_settings


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="bula-knowledge", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="sobe a API HTTP")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)

    ingest = commands.add_parser("ingest", help="baixa, divide e vetoriza a documentação")
    ingest.add_argument("--source", action="append", help="id do catálogo (padrão: todas)")
    ingest.add_argument("--path", type=Path, help="pasta local de Markdown em vez do GitHub")
    ingest.add_argument("--id", help="id da fonte local (com --path)")
    ingest.add_argument("--title", help="título da fonte local (com --path)")
    ingest.add_argument("--base-url", help="endereço onde a pasta local é publicada, se for")
    ingest.add_argument("--rebuild", action="store_true", help="apaga o índice antes")

    search = commands.add_parser("search", help="busca híbrida, sem LLM")
    search.add_argument("query")
    search.add_argument("-k", type=int)
    search.add_argument("--mode", choices=["hybrid", "vector", "lexical"], default="hybrid")

    context = commands.add_parser("context", help="roda o workflow de RAG para uma pergunta")
    context.add_argument("question")

    commands.add_parser("sources", help="lista o que está indexado")

    evaluate = commands.add_parser("eval", help="avalia a busca contra o conjunto ouro")
    evaluate.add_argument("--golden", type=Path, default=REPO_ROOT / "knowledge/eval/golden.jsonl")
    evaluate.add_argument("-k", type=int, default=5)
    evaluate.add_argument("--modes", default="hybrid,vector,lexical")
    evaluate.add_argument("--no-mlflow", action="store_true", help="só imprime, sem gravar o run")
    evaluate.add_argument(
        "--workflow",
        action="store_true",
        help="mede o workflow de RAG com LLM, etapa por etapa (gasta tokens do provedor)",
    )
    evaluate.add_argument("--limit", type=int, help="usa só as N primeiras perguntas")

    args = parser.parse_args(argv)
    settings = get_settings()
    try:
        return _COMMANDS[args.command](args, settings)
    except (RuntimeError, ValueError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1


def _serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from bula_knowledge.logs import setup_logging

    setup_logging()
    uvicorn.run(
        "bula_knowledge.api:create_app",
        factory=True,
        host=args.host or settings.knowledge_host,
        port=args.port or settings.knowledge_port,
        log_config=None,
        access_log=False,
    )
    return 0


def _ingest(args: argparse.Namespace, settings: Settings) -> int:
    from bula_knowledge.api import build_deps
    from bula_knowledge.db import Database
    from bula_knowledge.ingest.fetch import RawDocument, fetch_github, load_local
    from bula_knowledge.ingest.pipeline import ingest_source
    from bula_knowledge.ingest.sources import SOURCES, Source

    if args.rebuild:
        db = Database(settings.database_url, settings.data_dir)
        db.reset()
        db.close()
        print("índice apagado")

    deps = build_deps(settings)
    jobs: list[tuple[Source, str | None, list[RawDocument]]] = []
    try:
        if args.path:
            if not args.id:
                raise ValueError("--path exige --id")
            local = Source(
                id=args.id,
                title=args.title or args.id,
                repo="local",
                ref="local",
                license="não informada",
                prefix="",
                base_url=args.base_url,
            )
            jobs.append((local, None, load_local(args.path)))
        else:
            unknown = [name for name in (args.source or []) if name not in SOURCES]
            if unknown:
                raise ValueError(f"fonte desconhecida: {', '.join(unknown)}")
            token = settings.github_token.get_secret_value() if settings.github_token else None
            for name in args.source or list(SOURCES):
                source = SOURCES[name]
                print(f"{source.id}: baixando {source.repo}@{source.ref}...", flush=True)
                commit_sha, documents = fetch_github(source, token=token)
                jobs.append((source, commit_sha, documents))
        for job_source, job_sha, job_documents in jobs:
            report = ingest_source(deps.db, deps.embedder, job_source, job_documents, job_sha)
            print(json.dumps(asdict(report), ensure_ascii=False), flush=True)
    finally:
        deps.db.close()
    return 0


def _search(args: argparse.Namespace, settings: Settings) -> int:
    from bula_knowledge.api import build_deps
    from bula_knowledge.ingest.chunk import full_heading

    deps = build_deps(settings)
    try:
        chunks, took_ms = deps.searcher.search(
            args.query, k=args.k or settings.search_k, mode=args.mode
        )
        for position, chunk in enumerate(chunks, start=1):
            print(f"{position}. [{chunk.source}] {full_heading(chunk.title, chunk.heading)}")
            print(f"   {chunk.url}")
            print(
                f"   rrf={chunk.score:.4f} vetorial={chunk.vector_rank} "
                f"textual={chunk.lexical_rank}"
            )
        print(f"{len(chunks)} trechos em {took_ms} ms ({args.mode})")
    finally:
        deps.db.close()
    return 0


def _context(args: argparse.Namespace, settings: Settings) -> int:
    from bula_knowledge.api import build_deps

    deps = build_deps(settings)
    try:
        result = deps.workflow.run(args.question)
        print(result.model_dump_json(indent=2, exclude={"chunks": {"__all__": {"content"}}}))
    finally:
        deps.db.close()
    return 0


def _sources(args: argparse.Namespace, settings: Settings) -> int:
    from bula_knowledge.api import build_deps, list_sources

    deps = build_deps(settings)
    try:
        for source in list_sources(deps.db):
            print(json.dumps(source.model_dump(), ensure_ascii=False))
    finally:
        deps.db.close()
    return 0


def _eval(args: argparse.Namespace, settings: Settings) -> int:
    from bula_knowledge.api import build_deps, list_sources
    from bula_knowledge.evaluation.retrieval import evaluate, load_golden, log_to_mlflow
    from bula_knowledge.tracking import configure_mlflow

    cases = load_golden(args.golden)
    if args.limit:
        cases = cases[: args.limit]
    deps = build_deps(settings)
    try:
        corpus = ", ".join(f"{s.id}@{s.ref}" for s in list_sources(deps.db))
        params = {
            "embedding_model": deps.embedder.name,
            "corpus": corpus,
            "rrf_k": str(settings.rrf_k),
            "candidates": str(settings.search_candidates),
            "lexical_min_coverage": str(settings.lexical_min_coverage),
            "golden": args.golden.name,
        }
        if not args.no_mlflow:
            configure_mlflow(settings)
        # Uma passada de aquecimento, para a latência não medir a carga do modelo.
        deps.searcher.search("warm up", k=1)
        if args.workflow:
            return _eval_workflow(args, settings, deps, cases, params)
        for mode in args.modes.split(","):
            # Sem isto, os modos seguintes achariam a pergunta já vetorizada e pareceriam
            # mais rápidos do que são.
            clear_cache = getattr(deps.embedder, "clear_cache", None)
            if clear_cache is not None:
                clear_cache()
            result = evaluate(deps.searcher, cases, k=args.k, mode=mode.strip())
            run_id = None if args.no_mlflow else log_to_mlflow(result, params)
            print(
                json.dumps({"mode": mode, "run_id": run_id, **result.metrics}, ensure_ascii=False)
            )
    finally:
        deps.db.close()
    return 0


def _eval_workflow(
    args: argparse.Namespace,
    settings: Settings,
    deps: Any,
    cases: list[Any],
    params: dict[str, str],
) -> int:
    from bula_knowledge.evaluation.retrieval import evaluate_workflow, log_workflow_to_mlflow

    if deps.workflow.mode != "llm":
        raise ValueError("--workflow mede o RAG com LLM: defina LLM_PROVIDER e a chave no .env")

    def progress(number: int, case: Any) -> None:
        print(
            f"[{number}/{len(cases)}] {case.id}: crua={case.rank_raw} "
            f"reescrita={case.rank_rewritten} triagem={case.rank_graded} "
            f"aprovados={case.kept} tentativas={case.attempts} "
            f"tokens={case.input_tokens}+{case.output_tokens} {case.took_ms:.0f} ms",
            file=sys.stderr,
            flush=True,
        )

    result = evaluate_workflow(deps.workflow, deps.searcher, cases, k=args.k, on_case=progress)
    params = {
        **params,
        "llm_provider": settings.llm_provider,
        "llm_model": str(settings.llm_fast_model or settings.llm_model),
        "llm_effort": str(settings.llm_effort),
        "rag_max_attempts": str(settings.rag_max_attempts),
    }
    run_id = None if args.no_mlflow else log_workflow_to_mlflow(result, params)
    print(json.dumps({"mode": "workflow", "run_id": run_id, **result.metrics}, ensure_ascii=False))
    return 0


_COMMANDS = {
    "serve": _serve,
    "ingest": _ingest,
    "search": _search,
    "context": _context,
    "sources": _sources,
    "eval": _eval,
}


if __name__ == "__main__":
    raise SystemExit(main())
