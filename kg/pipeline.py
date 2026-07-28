from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from . import extraction, resolution, sources, store, validation
from .llm import JSONLLM
from .models import ChunkResult, SourcePassage


def process_catalog(
    conn: sqlite3.Connection,
    llm: JSONLLM,
    catalog_path: str | Path,
    *,
    source_limit: int | None = None,
    max_chunks: int | None = None,
    chunk_chars: int = 8000,
    overlap_chars: int = 500,
    max_entities: int = 20,
    max_claims: int = 30,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    specs = sources.load_catalog(catalog_path)
    if source_limit is not None:
        specs = specs[: max(0, source_limit)]
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    remaining_chunks = max_chunks
    for spec in specs:
        if remaining_chunks is not None and remaining_chunks <= 0:
            break
        try:
            loaded = sources.load_source(spec)
            source_id, is_new_version = store.add_source(conn, loaded)
            chunks = sources.chunk_text(
                loaded.content,
                max_chars=chunk_chars,
                overlap_chars=overlap_chars,
            )
            source_result = {
                "source": spec.name,
                "source_id": source_id,
                "new_version": is_new_version,
                "processed_chunks": 0,
                "skipped_chunks": 0,
                "entities": 0,
                "claims": 0,
                "evidence": 0,
                "rejected": [],
            }
            for chunk in chunks:
                if remaining_chunks is not None and remaining_chunks <= 0:
                    break
                processing_hash = _processing_hash(
                    chunk.content_hash,
                    model=_model_name(llm),
                    max_entities=max_entities,
                    max_claims=max_claims,
                )
                if store.progress_done(
                    conn, source_id, chunk.index, processing_hash
                ):
                    source_result["skipped_chunks"] += 1
                    continue
                if remaining_chunks is not None:
                    remaining_chunks -= 1
                try:
                    result = process_chunk(
                        conn,
                        llm,
                        source_id=source_id,
                        text=chunk.text,
                        passages=chunk.passages,
                        location=chunk.location,
                        max_entities=max_entities,
                        max_claims=max_claims,
                    )
                    store.save_progress(
                        conn,
                        source_id,
                        chunk.index,
                        processing_hash,
                        status="done",
                        result=result.as_dict(),
                    )
                    source_result["processed_chunks"] += 1
                    for key in ("entities", "claims", "evidence"):
                        source_result[key] += getattr(result, key)
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
                    if stop_on_error:
                        raise
            completed.append(source_result)
        except Exception as exc:
            failures.append({"source": spec.name, "error": str(exc)})
            if stop_on_error:
                raise
    return {"completed": completed, "failures": failures}


def process_chunk(
    conn: sqlite3.Connection,
    llm: JSONLLM,
    *,
    source_id: int,
    text: str,
    passages: tuple[SourcePassage, ...],
    location: str,
    max_entities: int = 20,
    max_claims: int = 30,
) -> ChunkResult:
    batch = extraction.extract(
        llm,
        text,
        passages=passages,
        location=location,
        max_entities=max_entities,
        max_claims=max_claims,
    )
    result = ChunkResult(rejected=list(batch.rejected))
    local: dict[str, int] = {}
    extraction_model = _model_name(llm)

    for observation in batch.entities:
        resolved = resolution.resolve_observation(conn, llm, observation)
        entity_id = resolved.entity_id
        keys = (observation.name, *observation.aliases)
        for name in keys:
            local[store.normalize_name(name)] = entity_id
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

    for claim in batch.claims:
        subject_id = _endpoint(conn, local, claim.subject)
        object_id = _endpoint(conn, local, claim.object)
        if subject_id is None or object_id is None:
            result.rejected.append(
                f"Claim 端点无法唯一解析: {claim.subject} -> {claim.object}"
            )
            continue
        verdict, reason = validation.judge_claim(llm, claim)
        expected = "supports" if claim.polarity == "support" else "contradicts"
        if verdict != expected:
            result.rejected.append(
                f"Claim 证据裁决为 {verdict}: "
                f"{claim.subject} {claim.relation} {claim.object}; {reason}"
            )
            continue

        existing = store.find_claim(
            conn, subject_id, claim.relation, object_id
        )
        if claim.polarity == "oppose" and existing is None:
            result.rejected.append(
                "反对证据对应的 Claim 尚不存在，按首版规则暂不入库"
            )
            continue
        if existing is None:
            claim_id, created, error = store.upsert_claim(
                conn, subject_id, claim.relation, object_id
            )
            if claim_id is None:
                result.rejected.append(
                    f"Claim 未写入: {claim.subject} {claim.relation} "
                    f"{claim.object}; {error}"
                )
                continue
            if created:
                result.claims += 1
        else:
            claim_id = int(existing["id"])
        if store.add_evidence(
            conn,
            source_id=source_id,
            source_text=claim.source_text,
            model_quote=claim.model_quote,
            passage_ids=claim.passage_ids,
            passage_version=extraction.PASSAGE_VERSION,
            location=claim.location,
            polarity=claim.polarity,
            extraction_model=extraction_model,
            extraction_prompt_version=extraction.EXTRACTION_PROMPT_VERSION,
            validator_model=extraction_model,
            validator_prompt_version=validation.VALIDATION_PROMPT_VERSION,
            validator_verdict=verdict,
            validator_reason=reason,
            claim_id=claim_id,
        ):
            result.evidence += 1
    conn.commit()
    return result


def _endpoint(
    conn: sqlite3.Connection, local: dict[str, int], name: str
) -> int | None:
    key = store.normalize_name(name)
    if key in local:
        return local[key]
    matches = store.exact_entity_ids(conn, name)
    return matches[0] if len(matches) == 1 else None


def _model_name(llm: JSONLLM) -> str:
    config = getattr(llm, "config", None)
    model = getattr(config, "model", "")
    return str(model or llm.__class__.__name__)


def _processing_hash(
    chunk_hash: str,
    *,
    model: str,
    max_entities: int,
    max_claims: int,
) -> str:
    config = {
        "chunk_hash": chunk_hash,
        "passage_version": extraction.PASSAGE_VERSION,
        "extraction_prompt_version": extraction.EXTRACTION_PROMPT_VERSION,
        "resolution_prompt_version": resolution.RESOLUTION_PROMPT_VERSION,
        "validator_prompt_version": validation.VALIDATION_PROMPT_VERSION,
        "model": model,
        "max_entities": max_entities,
        "max_claims": max_claims,
    }
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
