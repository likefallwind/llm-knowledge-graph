PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('schema_version', '1');

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

CREATE TABLE IF NOT EXISTS entities (
    id              INTEGER PRIMARY KEY,
    canonical_name  TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    definition      TEXT NOT NULL,
    entity_type     TEXT NOT NULL CHECK(entity_type IN (
        'resource', 'criterion', 'data', 'task', 'solution', 'concept'
    )),
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

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
