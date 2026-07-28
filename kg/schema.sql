PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('schema_version', '4');

-- A row is one immutable version of one logical source.
CREATE TABLE IF NOT EXISTS sources (
    id           INTEGER PRIMARY KEY,
    source_key   TEXT NOT NULL,
    name         TEXT NOT NULL,
    source_type  TEXT NOT NULL,
    uri          TEXT NOT NULL DEFAULT '',
    version      TEXT NOT NULL,
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    language     TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_key, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_sources_key ON sources(source_key, created_at);

-- Entity 没有单值 entity_type。类型是 mention 级的观察，记在
-- evidence.observed_entity_type 上；Entity 层的类型表示是这些观察的汇总
-- （type profile，见 store.type_profile）。一个词确实可能同时是多个类型，
-- 例如「深度学习」既是一族做法也是一个研究方向。
CREATE TABLE IF NOT EXISTS entities (
    id              INTEGER PRIMARY KEY,
    canonical_name  TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    definition      TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Aliases are an Entity attribute, not another knowledge-object type.
CREATE TABLE IF NOT EXISTS entity_aliases (
    id              INTEGER PRIMARY KEY,
    entity_id       INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_id, normalized_name)
);

CREATE INDEX IF NOT EXISTS idx_aliases_name
ON entity_aliases(normalized_name, entity_id);

CREATE TABLE IF NOT EXISTS claims (
    id         INTEGER PRIMARY KEY,
    subject_id INTEGER NOT NULL REFERENCES entities(id),
    relation   TEXT NOT NULL CHECK(relation IN (
        'is_a', 'part_of', 'prerequisite_of'
    )),
    object_id  INTEGER NOT NULL REFERENCES entities(id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK(subject_id != object_id),
    UNIQUE(subject_id, relation, object_id)
);

CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject_id, relation);
CREATE INDEX IF NOT EXISTS idx_claims_object ON claims(object_id, relation);

CREATE TABLE IF NOT EXISTS evidence (
    id           INTEGER PRIMARY KEY,
    target_key   TEXT NOT NULL,
    entity_id    INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    claim_id     INTEGER REFERENCES claims(id) ON DELETE CASCADE,
    source_id    INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    excerpt      TEXT NOT NULL,
    model_quote  TEXT NOT NULL DEFAULT '',
    -- 本次观察判出的类型。仅对 Entity Evidence 有意义；空串表示未记录
    -- （schema 4 之前的历史行，无法回填——我们不知道当时判的是什么）。
    observed_entity_type TEXT NOT NULL DEFAULT '' CHECK(observed_entity_type IN (
        '', 'resource', 'criterion', 'data', 'task', 'solution', 'concept'
    )),
    passage_ids  TEXT NOT NULL DEFAULT '[]',
    passage_version TEXT NOT NULL DEFAULT 'source-passages-1',
    extraction_model TEXT NOT NULL DEFAULT '',
    extraction_prompt_version TEXT NOT NULL DEFAULT '',
    validator_model TEXT NOT NULL DEFAULT '',
    validator_prompt_version TEXT NOT NULL DEFAULT '',
    validator_verdict TEXT NOT NULL DEFAULT '',
    validator_reason TEXT NOT NULL DEFAULT '',
    excerpt_hash TEXT NOT NULL,
    location     TEXT NOT NULL DEFAULT '',
    polarity     TEXT NOT NULL CHECK(polarity IN ('support', 'oppose')),
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK(
        (entity_id IS NOT NULL AND claim_id IS NULL)
        OR (entity_id IS NULL AND claim_id IS NOT NULL)
    ),
    UNIQUE(target_key, source_id, excerpt_hash, polarity)
);

CREATE INDEX IF NOT EXISTS idx_evidence_entity ON evidence(entity_id);
-- idx_evidence_observed_type 建在 db._migrate_evidence 里：本脚本对已存在的表
-- 是 no-op，而该索引依赖的列要等迁移才补上，写在这里会先于列执行而失败。
CREATE INDEX IF NOT EXISTS idx_evidence_claim ON evidence(claim_id);
CREATE INDEX IF NOT EXISTS idx_evidence_source ON evidence(source_id);

-- Operational resume bookkeeping. It is deliberately not a graph state machine.
CREATE TABLE IF NOT EXISTS source_progress (
    source_id   INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_hash  TEXT NOT NULL,
    status      TEXT NOT NULL CHECK(status IN ('done', 'failed')),
    result      TEXT NOT NULL DEFAULT '{}',
    error       TEXT NOT NULL DEFAULT '',
    updated_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(source_id, chunk_index, chunk_hash)
);
