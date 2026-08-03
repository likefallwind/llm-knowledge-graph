PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO schema_meta(key,value) VALUES ('schema_version','9');

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    source_key TEXT NOT NULL,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    uri TEXT NOT NULL DEFAULT '',
    version TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_key,content_hash)
);
CREATE INDEX IF NOT EXISTS idx_sources_key ON sources(source_key,created_at);

-- Deterministic document structure.  These rows are provenance/navigation,
-- never semantic Entity or Claim objects.
CREATE TABLE IF NOT EXISTS source_sections (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    section_key TEXT NOT NULL,
    parent_id INTEGER REFERENCES source_sections(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    depth INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    path_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id,section_key)
);
CREATE INDEX IF NOT EXISTS idx_source_sections_parent
ON source_sections(source_id,parent_id,ordinal);

CREATE TABLE IF NOT EXISTS source_passages (
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    passage_id TEXT NOT NULL,
    section_id INTEGER REFERENCES source_sections(id) ON DELETE SET NULL,
    content_hash TEXT NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    location TEXT NOT NULL,
    PRIMARY KEY(source_id,passage_id)
);
CREATE INDEX IF NOT EXISTS idx_source_passages_section
ON source_passages(section_id,passage_id);

CREATE TABLE IF NOT EXISTS section_summaries (
    id INTEGER PRIMARY KEY,
    section_id INTEGER NOT NULL REFERENCES source_sections(id) ON DELETE CASCADE,
    input_fingerprint TEXT NOT NULL,
    summarizer_model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    summary TEXT NOT NULL,
    supporting_passage_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(section_id,input_fingerprint,summarizer_model,prompt_version)
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    definition TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS entity_aliases (
    id INTEGER PRIMARY KEY,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_id,normalized_name)
);
CREATE INDEX IF NOT EXISTS idx_aliases_name
ON entity_aliases(normalized_name,entity_id);

CREATE TABLE IF NOT EXISTS entity_observations (
    id INTEGER PRIMARY KEY,
    observation_key TEXT NOT NULL UNIQUE,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    section_id INTEGER REFERENCES source_sections(id) ON DELETE SET NULL,
    chunk_index INTEGER NOT NULL,
    name TEXT NOT NULL,
    reference_key TEXT NOT NULL,
    definition TEXT NOT NULL,
    -- Kept as the first raw label for compatibility; not a whitelist.
    observed_entity_type TEXT NOT NULL DEFAULT '',
    raw_type_labels TEXT NOT NULL DEFAULT '[]',
    aliases TEXT NOT NULL DEFAULT '[]',
    source_text TEXT NOT NULL,
    model_quote TEXT NOT NULL DEFAULT '',
    passage_ids TEXT NOT NULL DEFAULT '[]',
    passage_version TEXT NOT NULL DEFAULT 'source-passages-2',
    location TEXT NOT NULL DEFAULT '',
    extraction_model TEXT NOT NULL DEFAULT '',
    extraction_prompt_version TEXT NOT NULL DEFAULT '',
    entity_id INTEGER REFERENCES entities(id) ON DELETE SET NULL,
    resolution_outcome TEXT NOT NULL DEFAULT '' CHECK(resolution_outcome IN (
        '', 'same', 'new', 'uncertain'
    )),
    resolution_reason TEXT NOT NULL DEFAULT '',
    candidate_entity_ids TEXT NOT NULL DEFAULT '[]',
    resolver_model TEXT NOT NULL DEFAULT '',
    resolver_prompt_version TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_entity_observations_entity
ON entity_observations(entity_id,id);
CREATE INDEX IF NOT EXISTS idx_entity_observations_section
ON entity_observations(section_id,id);
CREATE INDEX IF NOT EXISTS idx_entity_observations_reference
ON entity_observations(reference_key,id);

CREATE TABLE IF NOT EXISTS entity_type_vocab (
    id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS entity_type_aliases (
    id INTEGER PRIMARY KEY,
    type_id INTEGER NOT NULL REFERENCES entity_type_vocab(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    UNIQUE(type_id,normalized_name)
);
CREATE INDEX IF NOT EXISTS idx_entity_type_aliases_name
ON entity_type_aliases(normalized_name,type_id);
CREATE TABLE IF NOT EXISTS entity_observation_types (
    id INTEGER PRIMARY KEY,
    observation_id INTEGER NOT NULL REFERENCES entity_observations(id) ON DELETE CASCADE,
    raw_label TEXT NOT NULL,
    type_id INTEGER NOT NULL REFERENCES entity_type_vocab(id),
    outcome TEXT NOT NULL CHECK(outcome IN ('same','new','uncertain')),
    normalizer_model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(observation_id,raw_label,normalizer_model,prompt_version)
);

CREATE TABLE IF NOT EXISTS entity_definition_syntheses (
    id INTEGER PRIMARY KEY,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    observation_fingerprint TEXT NOT NULL,
    synthesizer_model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    definition TEXT NOT NULL,
    supporting_observations TEXT NOT NULL DEFAULT '[]',
    rejected_candidates TEXT NOT NULL DEFAULT '[]',
    limitation TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_id,observation_fingerprint,synthesizer_model,prompt_version)
);

-- KGGen-style open predicate vocabulary.  relation_kind is a non-restrictive
-- navigation/validation facet; canonical_name remains open-ended.
CREATE TABLE IF NOT EXISTS relation_types (
    id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    relation_kind TEXT NOT NULL DEFAULT 'other' CHECK(relation_kind IN (
        'is_a','part_of','prerequisite_of','other'
    )),
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS relation_aliases (
    id INTEGER PRIMARY KEY,
    relation_type_id INTEGER NOT NULL REFERENCES relation_types(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    UNIQUE(relation_type_id,normalized_name)
);
CREATE INDEX IF NOT EXISTS idx_relation_aliases_name
ON relation_aliases(normalized_name,relation_type_id);

CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY,
    subject_id INTEGER NOT NULL REFERENCES entities(id),
    relation_type_id INTEGER NOT NULL REFERENCES relation_types(id),
    -- Denormalized canonical label retained for readable SQL/export.
    relation TEXT NOT NULL,
    object_id INTEGER NOT NULL REFERENCES entities(id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK(subject_id != object_id),
    UNIQUE(subject_id,relation_type_id,object_id)
);
CREATE INDEX IF NOT EXISTS idx_claims_subject
ON claims(subject_id,relation_type_id);
CREATE INDEX IF NOT EXISTS idx_claims_object
ON claims(object_id,relation_type_id);

CREATE TABLE IF NOT EXISTS claim_observations (
    id INTEGER PRIMARY KEY,
    observation_key TEXT NOT NULL UNIQUE,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    section_id INTEGER REFERENCES source_sections(id) ON DELETE SET NULL,
    chunk_index INTEGER NOT NULL,
    subject_name TEXT NOT NULL,
    subject_reference_key TEXT NOT NULL,
    subject_entity_id INTEGER REFERENCES entities(id) ON DELETE SET NULL,
    raw_relation TEXT NOT NULL,
    relation TEXT NOT NULL,
    relation_type_id INTEGER REFERENCES relation_types(id) ON DELETE SET NULL,
    relation_kind TEXT NOT NULL DEFAULT 'other' CHECK(relation_kind IN (
        'is_a','part_of','prerequisite_of','other'
    )),
    object_name TEXT NOT NULL,
    object_reference_key TEXT NOT NULL,
    object_entity_id INTEGER REFERENCES entities(id) ON DELETE SET NULL,
    polarity TEXT NOT NULL CHECK(polarity IN ('support','oppose')),
    source_text TEXT NOT NULL,
    model_quote TEXT NOT NULL DEFAULT '',
    passage_ids TEXT NOT NULL DEFAULT '[]',
    passage_version TEXT NOT NULL DEFAULT 'source-passages-2',
    location TEXT NOT NULL DEFAULT '',
    extraction_model TEXT NOT NULL DEFAULT '',
    extraction_prompt_version TEXT NOT NULL DEFAULT '',
    claim_id INTEGER REFERENCES claims(id) ON DELETE SET NULL,
    materialization_error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_claim_observations_subject
ON claim_observations(subject_reference_key,subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_claim_observations_object
ON claim_observations(object_reference_key,object_entity_id);
CREATE INDEX IF NOT EXISTS idx_claim_observations_section
ON claim_observations(section_id,id);

CREATE TABLE IF NOT EXISTS relation_resolutions (
    id INTEGER PRIMARY KEY,
    observation_id INTEGER NOT NULL REFERENCES claim_observations(id) ON DELETE CASCADE,
    raw_relation TEXT NOT NULL,
    relation_type_id INTEGER NOT NULL REFERENCES relation_types(id),
    outcome TEXT NOT NULL CHECK(outcome IN ('same','new','uncertain')),
    candidate_relation_ids TEXT NOT NULL DEFAULT '[]',
    normalizer_model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(observation_id,normalizer_model,prompt_version)
);

CREATE TABLE IF NOT EXISTS claim_observation_judgments (
    id INTEGER PRIMARY KEY,
    observation_id INTEGER NOT NULL REFERENCES claim_observations(id) ON DELETE CASCADE,
    validator_model TEXT NOT NULL,
    validator_prompt_version TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK(verdict IN ('supports','contradicts','insufficient')),
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(observation_id,validator_model,validator_prompt_version)
);

CREATE TABLE IF NOT EXISTS relation_expansion_attempts (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    subject_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    object_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    context_fingerprint TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK(outcome IN ('relation','none','failed')),
    observation_id INTEGER REFERENCES claim_observations(id) ON DELETE SET NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id,subject_id,object_id,context_fingerprint,model,prompt_version)
);

CREATE TABLE IF NOT EXISTS entity_candidate_reviews (
    id INTEGER PRIMARY KEY,
    reference_key TEXT NOT NULL,
    evidence_fingerprint TEXT NOT NULL,
    reviewer_model TEXT NOT NULL,
    reviewer_version TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('same','new','uncertain')),
    entity_id INTEGER REFERENCES entities(id) ON DELETE SET NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_key,evidence_fingerprint,reviewer_model,reviewer_version)
);

CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY,
    target_key TEXT NOT NULL,
    entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    claim_id INTEGER REFERENCES claims(id) ON DELETE CASCADE,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    excerpt TEXT NOT NULL,
    model_quote TEXT NOT NULL DEFAULT '',
    observed_entity_type TEXT NOT NULL DEFAULT '',
    passage_ids TEXT NOT NULL DEFAULT '[]',
    passage_version TEXT NOT NULL DEFAULT 'source-passages-2',
    extraction_model TEXT NOT NULL DEFAULT '',
    extraction_prompt_version TEXT NOT NULL DEFAULT '',
    validator_model TEXT NOT NULL DEFAULT '',
    validator_prompt_version TEXT NOT NULL DEFAULT '',
    validator_verdict TEXT NOT NULL DEFAULT '',
    validator_reason TEXT NOT NULL DEFAULT '',
    excerpt_hash TEXT NOT NULL,
    location TEXT NOT NULL DEFAULT '',
    polarity TEXT NOT NULL CHECK(polarity IN ('support','oppose')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK((entity_id IS NOT NULL AND claim_id IS NULL)
       OR (entity_id IS NULL AND claim_id IS NOT NULL)),
    UNIQUE(target_key,source_id,excerpt_hash,polarity)
);
CREATE INDEX IF NOT EXISTS idx_evidence_entity ON evidence(entity_id);
CREATE INDEX IF NOT EXISTS idx_evidence_claim ON evidence(claim_id);
CREATE INDEX IF NOT EXISTS idx_evidence_source ON evidence(source_id);

CREATE TABLE IF NOT EXISTS source_progress (
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('done','failed')),
    result TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(source_id,chunk_index,chunk_hash)
);
