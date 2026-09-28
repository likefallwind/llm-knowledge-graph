from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

from . import (
    definitions,
    extraction,
    observations,
    resolution,
    sources,
    store,
    structure,
    validation,
    vocabulary,
)
from .llm import JSONLLM, is_quota_exhausted
from .models import (
    ClaimObservation,
    ChunkResult,
    ExtractionBatch,
    LoadedSource,
    SourcePassage,
    TextChunk,
)


logger = logging.getLogger(__name__)

CONSECUTIVE_FAILURE_PAUSE_THRESHOLD = 3
CONSECUTIVE_FAILURE_PAUSE_SECONDS = 600


class _ConsecutiveFailurePauser:
    def __init__(self, pause_seconds: float | None = None) -> None:
        self.count = 0
        self.pause_seconds = (
            CONSECUTIVE_FAILURE_PAUSE_SECONDS if pause_seconds is None else pause_seconds
        )

    def record_success(self) -> None:
        self.count = 0

    def record_failure(self, *, immediate: bool = False) -> None:
        """Count one failed Chunk, pausing once failures look like an outage.

        ``immediate`` is for failures that are already conclusive on their own —
        an exhausted quota needs no corroboration from two more dead Chunks.
        """

        self.count += 1
        if not immediate and self.count < CONSECUTIVE_FAILURE_PAUSE_THRESHOLD:
            return
        reason = (
            "额度耗尽"
            if immediate
            else f"连续 {CONSECUTIVE_FAILURE_PAUSE_THRESHOLD} 个 Chunk 失败"
        )
        logger.warning(
            "%s，暂停 %d 秒后继续",
            reason,
            self.pause_seconds,
        )
        time.sleep(self.pause_seconds)
        self.count = 0


def process_catalog(
    conn: sqlite3.Connection,
    llm: JSONLLM,
    catalog_path: str | Path,
    *,
    source_limit: int | None = None,
    start_chunk: int = 0,
    max_chunks: int | None = None,
    chunk_chars: int = 8000,
    overlap_chars: int = 500,
    max_entities: int = 50,
    max_claims: int = 30,
    chunk_workers: int = 1,
    judge_workers: int = 1,
    relation_workers: int = 1,
    stop_on_error: bool = False,
    synthesize_definitions: bool = False,
    definition_limit: int | None = None,
    summarize_sections: bool = True,
    summary_limit: int | None = None,
    summary_workers: int = 1,
    simple_llm: JSONLLM | None = None,
    source_keys: list[str] | None = None,
    work_selection: set[tuple[int, int, str]] | None = None,
    source_cache: dict[str, LoadedSource] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    failure_pause_seconds: float | None = None,
) -> dict[str, Any]:
    if chunk_workers < 1 or relation_workers < 1:
        raise ValueError("chunk_workers 和 relation_workers 必须至少为 1")
    if max_entities < 1 or max_claims < 1:
        raise ValueError("max_entities 和 max_claims 必须至少为 1")
    fast_llm = simple_llm or llm
    specs = sources.load_catalog(catalog_path)
    if source_keys:
        missing = set(source_keys) - {spec.key for spec in specs}
        if missing:
            raise ValueError(f"未知 source key: {sorted(missing)}")
        specs = [spec for spec in specs if spec.key in source_keys]
    if source_limit is not None:
        specs = specs[: max(0, source_limit)]
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    selection = work_selection if work_selection is not None else set()
    selected_source_ids: set[int] = set()
    for spec in specs:
        if max_chunks is not None and len(selection) >= max_chunks:
            admitted_ids = {item[0] for item in selection}
            existing_ids = {
                int(row[0]) for row in conn.execute(
                    "SELECT id FROM sources WHERE source_key=?", (spec.key,)
                )
            }
            if not admitted_ids.intersection(existing_ids):
                continue
        try:
            if source_cache is not None and spec.key in source_cache:
                loaded = source_cache[spec.key]
            else:
                loaded = sources.load_source(spec)
                if source_cache is not None:
                    source_cache[spec.key] = loaded
            source_id, is_new_version = store.add_source(conn, loaded)
            selected_source_ids.add(source_id)
            chunks = sources.chunk_text(
                loaded.content,
                max_chars=chunk_chars,
                overlap_chars=overlap_chars,
                **({"headings": spec.headings} if spec.headings is not None else {}),
            )
            if not chunks:
                raise ValueError(f"语料没有可处理的 Chunk: {spec.name}")
            if spec.expected_chunks is not None and len(chunks) != spec.expected_chunks:
                raise ValueError(f"Chunk 数量 {len(chunks)} 与 expected_chunks={spec.expected_chunks} 不符")
            all_passages = {
                passage.passage_id: passage
                for chunk in chunks
                for passage in chunk.passages
            }
            structure.sync_source_structure(
                conn, source_id, all_passages.values()
            )
            summary_result = {
                "processed": 0,
                "skipped": 0,
                "failed": 0,
            }
            if summarize_sections:
                summary_result = structure.summarize_source(
                    conn,
                    fast_llm,
                    source_id,
                    model=_model_name(fast_llm),
                    limit=summary_limit,
                    workers=summary_workers,
                )
                if summary_result["failed"]:
                    raise RuntimeError(
                        f"Section 摘要失败 {summary_result['failed']} 个；补跑成功后再抽取"
                    )
            source_result = {
                "source": spec.name,
                "source_id": source_id,
                "new_version": is_new_version,
                "processed_chunks": 0,
                "skipped_chunks": 0,
                "before_start_chunks": 0,
                "entities": 0,
                "claims": 0,
                "assertions": 0,
                "evidence": 0,
                "entity_observations": 0,
                "claim_observations": 0,
                "entity_cap_hit_chunks": [],
                "rejected": [],
                "section_summaries": summary_result,
                "expected_chunks": len(chunks),
                "source_key": spec.key,
                "content_sha256": loaded.content_hash,
            }
            work_items: list[tuple[TextChunk, str]] = []
            current_hashes = {}
            for chunk in chunks:
                section_id = structure.section_id_for_passages(
                    conn, source_id, (item.passage_id for item in chunk.passages),
                )
                context = structure.context_for_section(conn, section_id)
                current_hashes[chunk.index] = _processing_hash(
                    chunk.content_hash, model=_model_name(llm),
                    simple_model=_model_name(fast_llm), max_entities=max_entities,
                    max_claims=max_claims, section_context=context,
                )
                if chunk.index < max(0, start_chunk):
                    source_result["before_start_chunks"] += 1
                    continue
                processing_hash = current_hashes[chunk.index]
                if store.progress_done(
                    conn, source_id, chunk.index, processing_hash
                ):
                    source_result["skipped_chunks"] += 1
                    continue
                task_key = (source_id, chunk.index, processing_hash)
                if task_key not in selection:
                    if max_chunks is not None and len(selection) >= max_chunks:
                        continue
                    selection.add(task_key)
                contextual_chunk = replace(
                    chunk,
                    location=(
                        f"{chunk.location}\n{context}" if context else chunk.location
                    ),
                )
                work_items.append((contextual_chunk, processing_hash))

            failure_pauser = _ConsecutiveFailurePauser(failure_pause_seconds)
            for chunk, processing_hash, batch, extraction_error in (
                _extract_chunks_ordered(
                    llm,
                    work_items,
                    max_entities=max_entities,
                    max_claims=max_claims,
                    workers=chunk_workers,
                )
            ):
                try:
                    if extraction_error is not None:
                        raise extraction_error
                    result = process_chunk(
                        conn,
                        llm,
                        source_id=source_id,
                        chunk_index=chunk.index,
                        text=chunk.text,
                        passages=chunk.passages,
                        location=chunk.location,
                        max_entities=max_entities,
                        max_claims=max_claims,
                        judge_workers=judge_workers,
                        relation_workers=relation_workers,
                        batch=batch,
                        simple_llm=fast_llm,
                    )
                    store.save_progress(
                        conn,
                        source_id,
                        chunk.index,
                        processing_hash,
                        status="done",
                        result=result.as_dict(),
                    )
                    failure_pauser.record_success()
                    if on_progress:
                        on_progress(source_id, chunk.index, "done")
                    source_result["processed_chunks"] += 1
                    for key in (
                        "entities",
                        "claims",
                        "assertions",
                        "evidence",
                        "entity_observations",
                        "claim_observations",
                    ):
                        source_result[key] += getattr(result, key)
                    if result.entity_cap_hit:
                        source_result["entity_cap_hit_chunks"].append(chunk.index)
                    source_result["rejected"].extend(result.rejected)
                except Exception as exc:
                    conn.rollback()
                    store.save_progress(
                        conn,
                        source_id,
                        chunk.index,
                        processing_hash,
                        status="failed",
                        error=str(exc),
                    )
                    failure = {
                        "source": spec.name,
                        "chunk": chunk.index,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    if on_progress:
                        on_progress(source_id, chunk.index, "failed")
                    if stop_on_error:
                        raise
                    failure_pauser.record_failure(
                        immediate=is_quota_exhausted(exc)
                    )
            source_result["done_chunks"] = sum(
                store.progress_done(conn, source_id, chunk.index, current_hashes[chunk.index])
                for chunk in chunks
            )
            completed.append(source_result)
        except Exception as exc:
            failures.append({"source": spec.name, "error": str(exc)})
            if stop_on_error:
                raise
    definition_synthesis: dict[str, Any] = {
        "processed": [],
        "skipped": [],
        "failures": [],
        "remaining": 0,
    }
    if synthesize_definitions:
        entity_ids = sorted({
            int(row[0]) for source_id in selected_source_ids
            for row in conn.execute(
                "SELECT DISTINCT entity_id FROM entity_observations "
                "WHERE source_id=? AND entity_id IS NOT NULL", (source_id,)
            )
        })
        definition_synthesis = definitions.synthesize_pending(
            conn, llm, entity_ids=entity_ids, limit=definition_limit
        )
        definition_failures = [
            {"stage": "definition_synthesis", **item}
            for item in definition_synthesis["failures"]
        ]
        failures.extend(definition_failures)
        if stop_on_error and definition_failures:
            raise RuntimeError(definition_failures[0]["error"])
    return {
        "completed": completed,
        "failures": failures,
        "definition_synthesis": definition_synthesis,
    }


def process_chunk(
    conn: sqlite3.Connection,
    llm: JSONLLM,
    *,
    source_id: int,
    chunk_index: int = 0,
    text: str,
    passages: tuple[SourcePassage, ...],
    location: str,
    max_entities: int = 50,
    max_claims: int = 30,
    judge_workers: int = 1,
    relation_workers: int = 1,
    batch: ExtractionBatch | None = None,
    simple_llm: JSONLLM | None = None,
) -> ChunkResult:
    fast_llm = simple_llm or llm
    if batch is None:
        batch = extraction.extract(
            llm,
            text,
            passages=passages,
            location=location,
            max_entities=max_entities,
            max_claims=max_claims,
        )
    result = ChunkResult(
        rejected=list(batch.rejected),
        entity_cap_hit=len(batch.entities) >= max_entities,
    )
    local_candidates: dict[str, set[int]] = {}
    extraction_model = _model_name(llm)

    entity_observation_ids: list[int] = []
    for observation in batch.entities:
        observation_id, created = observations.add_entity_observation(
            conn,
            source_id=source_id,
            chunk_index=chunk_index,
            observation=observation,
            extraction_model=extraction_model,
        )
        entity_observation_ids.append(observation_id)
        result.entity_observations += int(created)

    observation_ids: list[int] = []
    for claim in batch.claims:
        observation_id, created = observations.add_claim_observation(
            conn,
            source_id=source_id,
            chunk_index=chunk_index,
            claim=claim,
            extraction_model=extraction_model,
        )
        observation_ids.append(observation_id)
        result.claim_observations += int(created)
    # Grounded observations survive later identity, validation, or model failures.
    conn.commit()

    for observation_id, observation in zip(
        entity_observation_ids, batch.entities
    ):
        resolved = resolution.resolve_observation(conn, llm, observation)
        observations.save_entity_resolution(
            conn,
            observation_id,
            resolved,
            resolver_model=extraction_model,
        )
        vocabulary.resolve_observation_types(
            conn,
            fast_llm,
            observation_id,
            observation,
            model=_model_name(fast_llm),
        )
        entity_id = resolved.entity_id
        # Extraction aliases are suggestions until the identity resolver has
        # accepted them.  Reusing raw suggestions here would bypass the
        # resolver for Claim endpoints inside the same Chunk.
        keys = (observation.name, *store.aliases_for(conn, entity_id))
        for name in keys:
            local_candidates.setdefault(store.reference_key(name), set()).add(
                entity_id
            )
        if store.add_evidence(
            conn,
            source_id=source_id,
            source_text=observation.source_text,
            model_quote=observation.model_quote,
            passage_ids=observation.passage_ids,
            passage_version=extraction.PASSAGE_VERSION,
            location=observation.location,
            polarity="support",
            extraction_model=extraction_model,
            extraction_prompt_version=extraction.EXTRACTION_PROMPT_VERSION,
            observed_entity_type=observation.entity_type,
            entity_id=entity_id,
        ):
            result.evidence += 1
        result.entities += 1

    local = {
        key: next(iter(entity_ids))
        for key, entity_ids in local_candidates.items()
        if len(entity_ids) == 1
    }
    observations.resolve_endpoint_ids(conn, observation_ids, local=local)
    # Normalize relations only after endpoint resolution so the normalizer sees
    # the final canonical endpoint names together with the complete Assertion.
    relation_results: dict[int, tuple[ClaimObservation, vocabulary.RelationResolution]] = {}
    relation_inputs = []
    for observation_id in observation_ids:
        row = observations.get_observation(conn, observation_id)
        if row is None:
            continue
        claim = observations.as_claim(conn, row)
        relation_inputs.append((observation_id, claim))
    for observation_id, claim, relation_result in _normalize_relations_ordered(
        conn, fast_llm, relation_inputs, workers=relation_workers,
    ):
        relation_results[observation_id] = (claim, relation_result)
        conn.execute(
            """UPDATE claim_observations
               SET relation=?,relation_type_id=?,relation_kind=?,
                   updated_at=CURRENT_TIMESTAMP WHERE id=?""",
            (
                relation_result.canonical_name,
                relation_result.relation_type_id,
                relation_result.relation_kind,
                observation_id,
            ),
        )
        vocabulary.save_relation_resolution(
            conn,
            observation_id,
            claim.raw_relation or claim.relation,
            relation_result,
            model=_model_name(fast_llm),
        )
    observations.prepare_assertions(conn, observation_ids)
    rows = [
        observations.get_observation(conn, observation_id)
        for observation_id in observation_ids
    ]
    missing = [
        row
        for row in rows
        if row is not None
        and str(row["assertion_fingerprint"])
        and observations.current_judgment(
            conn, int(row["id"]), validator_model=extraction_model
        )
        is None
    ]
    judgments = _judge_claims(
        llm,
        [observations.as_claim(conn, row) for row in missing],
        workers=judge_workers,
    )
    for row, (verdict, reason) in zip(missing, judgments):
        observations.save_judgment(
            conn,
            int(row["id"]),
            validator_model=extraction_model,
            verdict=verdict,
            reason=reason,
        )
        observation_id = int(row["id"])
        relation_entry = relation_results.get(observation_id)
        expected = "supports" if str(row["polarity"]) == "support" else "contradicts"
        if verdict == expected and relation_entry is not None:
            original_claim, relation_result = relation_entry
            vocabulary.finalize_relation_resolution(
                conn,
                observation_id,
                original_claim.raw_relation or original_claim.relation,
                relation_result,
                model=_model_name(fast_llm),
            )
    # Cache relation judgments independently of whether endpoints exist yet.
    conn.commit()

    for observation_id in observation_ids:
        materialized = observations.materialize(
            conn, observation_id, validator_model=extraction_model
        )
        outcome = materialized["outcome"]
        if outcome == "materialized":
            result.claims += int(materialized["claim_created"])
            result.assertions += int(materialized["assertion_created"])
            result.evidence += int(materialized["evidence_created"])
        elif outcome in {"pending_endpoint", "pending_relation"}:
            row = observations.get_observation(conn, observation_id)
            result.pending.append(
                {
                    "stage": (
                        "endpoint_resolution"
                        if outcome == "pending_endpoint"
                        else "relation_resolution"
                    ),
                    "observation_id": observation_id,
                    "subject": str(row["subject_name"]),
                    "relation": str(row["relation"]),
                    "object": str(row["object_name"]),
                }
            )
        elif outcome in {"not_supported", "blocked"}:
            detail = {"observation_id": observation_id, **materialized}
            result.not_materialized.append(detail)
    # Any Entity or verified alias learned in this Chunk may unlock older
    # observations.  This deterministic replay uses cached judgments only.
    observations.resolve_and_materialize_cached(conn)
    conn.commit()
    return result


def _normalize_relations_ordered(
    conn: sqlite3.Connection,
    llm: JSONLLM,
    items: list[tuple[int, ClaimObservation]],
    *,
    workers: int,
) -> Iterator[tuple[int, ClaimObservation, vocabulary.RelationResolution]]:
    """Parallelize requests only; preserve observation order and stage boundaries."""
    if workers < 1:
        raise ValueError("relation_workers 必须至少为 1")
    if workers == 1 or len(items) < 2:
        for observation_id, claim in items:
            yield observation_id, claim, vocabulary.resolve_relation(conn, llm, claim)
        return

    # No RelationType/alias/Claim is finalized during this stage. Each request
    # sees exactly the candidates it would see in the serial loop, including
    # raw-name-specific exact matches. The SQLite connection stays here.
    jobs = [
        (observation_id, claim, vocabulary._relation_candidates(
            conn, claim.raw_relation or claim.relation,
        )) for observation_id, claim in items
    ]

    def run(job):
        observation_id, claim, candidates = job
        started = time.monotonic()
        logger.info("Relation normalization request started observation=%s", observation_id)
        try:
            result = vocabulary.resolve_relation_with_candidates(llm, claim, candidates)
            return observation_id, claim, result
        finally:
            logger.info("Relation normalization request ended observation=%s seconds=%.3f",
                        observation_id, time.monotonic() - started)

    logger.info("Relation normalization batch size=%s workers=%s", len(jobs), workers)
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        # map raises the first error in INPUT order, not completion order.
        # Writes and all subsequent judging/finalization remain in the caller.
        yield from executor.map(run, jobs)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _extract_chunks_ordered(
    llm: JSONLLM,
    work_items: list[tuple[TextChunk, str]],
    *,
    max_entities: int,
    max_claims: int,
    workers: int,
) -> Iterator[tuple[TextChunk, str, ExtractionBatch | None, Exception | None]]:
    """Prefetch extraction concurrently while yielding original Chunk order.

    At most ``workers`` futures exist at once.  No worker receives a database
    connection; entity resolution and every SQLite mutation stay in the caller.
    """

    def run(chunk: TextChunk) -> ExtractionBatch:
        return extraction.extract(
            llm,
            chunk.text,
            passages=chunk.passages,
            location=chunk.location,
            max_entities=max_entities,
            max_claims=max_claims,
        )

    if workers == 1:
        for chunk, processing_hash in work_items:
            try:
                yield chunk, processing_hash, run(chunk), None
            except Exception as exc:
                yield chunk, processing_hash, None, exc
        return

    executor = ThreadPoolExecutor(max_workers=workers)
    pending = deque()
    items = iter(work_items)

    def submit_next() -> bool:
        try:
            chunk, processing_hash = next(items)
        except StopIteration:
            return False
        pending.append((chunk, processing_hash, executor.submit(run, chunk)))
        return True

    try:
        for _ in range(min(workers, len(work_items))):
            submit_next()
        while pending:
            chunk, processing_hash, future = pending.popleft()
            try:
                batch, error = future.result(), None
            except Exception as exc:
                batch, error = None, exc
            submit_next()
            yield chunk, processing_hash, batch, error
    finally:
        executor.shutdown(cancel_futures=True)


def _judge_claims(
    llm: JSONLLM,
    claims: list[ClaimObservation],
    *,
    workers: int,
) -> list[tuple[str, str]]:
    if workers < 1:
        raise ValueError("judge_workers 必须至少为 1")
    if workers == 1 or len(claims) < 2:
        return [validation.judge_claim(llm, claim) for claim in claims]
    with ThreadPoolExecutor(max_workers=min(workers, len(claims))) as executor:
        return list(
            executor.map(
                lambda claim: validation.judge_claim(llm, claim),
                claims,
            )
        )


def _model_name(llm: JSONLLM) -> str:
    config = getattr(llm, "config", None)
    model = getattr(config, "model", "")
    return str(model or llm.__class__.__name__)


def _processing_hash(
    chunk_hash: str,
    *,
    model: str,
    simple_model: str,
    max_entities: int,
    max_claims: int,
    section_context: str = "",
) -> str:
    config = {
        "chunk_hash": chunk_hash,
        "passage_version": extraction.PASSAGE_VERSION,
        "extraction_prompt_version": extraction.EXTRACTION_PROMPT_VERSION,
        "entity_prompt_version": extraction.ENTITY_PROMPT_VERSION,
        "relation_prompt_version": extraction.RELATION_PROMPT_VERSION,
        "section_summary_prompt_version": structure.SUMMARY_PROMPT_VERSION,
        "relation_normalizer_version": vocabulary.RELATION_NORMALIZER_VERSION,
        "type_normalizer_version": vocabulary.TYPE_NORMALIZER_VERSION,
        "resolution_prompt_version": resolution.RESOLUTION_PROMPT_VERSION,
        "validator_prompt_version": validation.VALIDATION_PROMPT_VERSION,
        "model": model,
        "simple_model": simple_model,
        "max_entities": max_entities,
        "max_claims": max_claims,
    }
    if section_context:
        config["section_context"] = section_context
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
