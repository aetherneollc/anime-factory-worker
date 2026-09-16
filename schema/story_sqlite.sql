-- story.sqlite is the only business source of truth per story.
-- All timestamps are ISO-8601 TEXT. Booleans are INTEGER 0/1.
-- JSON columns are UTF-8 TEXT.

CREATE TABLE IF NOT EXISTS story (
    id TEXT PRIMARY KEY,
    title TEXT,
    slug TEXT,
    world_mode TEXT NOT NULL,
    style_preset TEXT NOT NULL DEFAULT 'shinkai',
    langs TEXT NOT NULL,
    primary_lang TEXT NOT NULL DEFAULT 'zh',
    version TEXT NOT NULL DEFAULT '1.0.0',
    planned_episodes INTEGER,
    schema_version TEXT NOT NULL DEFAULT 'v1',
    canon_token_count INTEGER NOT NULL DEFAULT 0,
    kind TEXT NOT NULL DEFAULT 'series',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (world_mode IN ('fiction', 'period_flavor', 'historical_strict')),
    CHECK (kind IN ('film', 'series', 'short'))
);

CREATE TABLE IF NOT EXISTS episodes (
    episode_code TEXT PRIMARY KEY,
    title TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    synopsis TEXT,
    recap TEXT,
    script_text TEXT,
    duration_target_s REAL NOT NULL DEFAULT 600,
    duration_actual_s REAL,
    kind TEXT NOT NULL DEFAULT 'series',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (status IN ('pending', 'running', 'completed', 'blocked')),
    CHECK (kind IN ('film', 'series', 'short'))
);

CREATE TABLE IF NOT EXISTS scenes (
    id TEXT PRIMARY KEY,
    episode_code TEXT NOT NULL,
    seq INTEGER NOT NULL,
    location_id TEXT,
    interior_id TEXT,
    mood TEXT,
    time_of_day TEXT,
    season TEXT,
    synopsis TEXT,
    beats_json TEXT,
    interval_from_prev TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    id TEXT PRIMARY KEY,
    episode_code TEXT NOT NULL,
    scene_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    duration REAL,
    h3_mode TEXT,
    first_frame_prompt TEXT,
    h3_prompt TEXT,
    refs_json TEXT,
    plate_id TEXT,
    costume_ids_json TEXT,
    video_path TEXT,
    keyframe_path TEXT,
    last_frame_path TEXT,
    seed INTEGER,
    retry_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    shot_id TEXT,
    chain_id TEXT,
    chain_index INTEGER NOT NULL DEFAULT 0,
    generation_plan_id TEXT,
    approved_generation_id TEXT,
    creative_version INTEGER NOT NULL DEFAULT 1,
    CHECK (h3_mode IS NULL OR h3_mode IN ('ref2va', 'fl2va_first', 'fl2va_first_last')),
    CHECK (status IN ('pending', 'pending_chain', 'prepared', 'running', 'completed', 'failed', 'blocked'))
);

CREATE TABLE IF NOT EXISTS cuts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    beats_start INTEGER,
    beats_end INTEGER,
    seconds REAL,
    size TEXT,
    camera TEXT,
    characters_json TEXT,
    props_json TEXT,
    frame_prompt TEXT,
    keyframe_path TEXT
);

-- gender is validated by the application (anime_factory.tts.resolve_gender), not by a
-- CHECK here. Legacy rows may be NULL until backfilled; the voice-lock path requires an
-- explicit cast gender or a 1boy/1girl identity tag before it will synthesize speech,
-- it no longer defaults an unmarked speaker to male.
CREATE TABLE IF NOT EXISTS characters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    aka_json TEXT,
    identity_prompt TEXT,
    visual_lock_prompt TEXT,
    age INTEGER NOT NULL,
    alive INTEGER NOT NULL DEFAULT 1,
    current_costume_id TEXT,
    current_location_id TEXT,
    seed INTEGER,
    sheet_dir TEXT,
    gender TEXT
);

CREATE TABLE IF NOT EXISTS character_relationships (
    a_id TEXT NOT NULL,
    b_id TEXT NOT NULL,
    relation TEXT,
    status TEXT,
    since_episode TEXT,
    PRIMARY KEY (a_id, b_id)
);

-- lang is validated by the application against config/languages.json, not by a CHECK here.
-- A hardcoded CHECK (lang IN ('zh','en','ja')) turned "adding a language is one registry row"
-- into an IntegrityError the moment 'ko' was registered, and an existing story.sqlite keeps its
-- old CHECK anyway, so the constraint could never be the real gate.
-- lock_version increments only on an explicit voice-profile migration (new clone,
-- speaker swap). A non-empty voice_uri is otherwise preserved across episodes:
-- recomputing a stock_voice_uri must never silently overwrite an established lock.
-- profile_fingerprint is anime_factory.tts.profile_fingerprint(character_id, gender,
-- voice_uri, model, lock_version) — it identifies the locked voice identity, independent
-- of any single line's text, and is what cross-episode reuse checks against.
CREATE TABLE IF NOT EXISTS character_voice (
    character_id TEXT NOT NULL,
    lang TEXT NOT NULL,
    voice_uri TEXT,
    reference_audio TEXT,
    speed REAL,
    emotion TEXT,
    version TEXT,
    lock_version INTEGER NOT NULL DEFAULT 1,
    profile_fingerprint TEXT,
    PRIMARY KEY (character_id, lang)
);

-- One row per synthesized line wav. fingerprint is anime_factory.tts.line_fingerprint(...)
-- over (character_id, gender, voice_uri, model, speed, lang, cleaned_text, lock_version).
-- compose/produce only reuse an on-disk wav when its stored fingerprint still matches;
-- any drift (voice migration, text edit, speed change) invalidates the cached take.
-- hosted_fingerprint is the control plane's 16-hex tts_manifest fingerprint
-- (sha256("v1|model|voice_uri|text")[:16]) recorded when a hosted pre-GPU wav is
-- bridged into this table (anime_factory.tts_hosted) — both fingerprints are kept;
-- the worker's richer local fingerprint always remains the reuse gate.
CREATE TABLE IF NOT EXISTS line_audio (
    segment_id TEXT NOT NULL,
    lang TEXT NOT NULL,
    character_id TEXT,
    fingerprint TEXT NOT NULL,
    hosted_fingerprint TEXT,
    wav_path TEXT,
    duration REAL,
    sample_rate INTEGER,
    content_hash TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (segment_id, lang)
);

CREATE TABLE IF NOT EXISTS locations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    aka_json TEXT,
    type TEXT,
    parent_id TEXT,
    plate_variant TEXT,
    damage_state TEXT,
    season TEXT,
    coord_json TEXT,
    plate_prompt TEXT
);

-- Canon graph (JSON export mirrors these). Not stuffed into factions.
CREATE TABLE IF NOT EXISTS geo_edges (
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'walk',
    distance_km REAL,
    travel_time TEXT,
    PRIMARY KEY (from_id, to_id, mode)
);

CREATE TABLE IF NOT EXISTS geo_interiors (
    id TEXT PRIMARY KEY,
    name TEXT,
    parent TEXT,
    scene_id TEXT,
    asset_path TEXT
);

CREATE TABLE IF NOT EXISTS factions (
    id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    members_json TEXT
);

CREATE TABLE IF NOT EXISTS props (
    id TEXT PRIMARY KEY,
    name TEXT,
    holder TEXT,
    state TEXT,
    identity_prompt TEXT,
    asset_dir TEXT,
    CHECK (state IS NULL OR state IN ('intact', 'damaged', 'destroyed', 'lost'))
);

CREATE TABLE IF NOT EXISTS timeline_events (
    id TEXT PRIMARY KEY,
    date TEXT,
    title TEXT,
    kind TEXT NOT NULL,
    episode TEXT,
    body TEXT,
    sources_json TEXT,
    invented INTEGER NOT NULL DEFAULT 0,
    CHECK (kind IN ('real', 'fictional'))
);

CREATE TABLE IF NOT EXISTS foreshadowing (
    id TEXT PRIMARY KEY,
    planted_episode TEXT,
    must_payoff_by TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    text TEXT,
    payoff_episode TEXT,
    CHECK (status IN ('open', 'paid', 'abandoned'))
);

CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    path TEXT,
    prompt TEXT,
    seed INTEGER,
    fingerprint TEXT,
    parent_id TEXT,
    costume_id TEXT,
    character_id TEXT,
    scene_id TEXT,
    prop_id TEXT,
    selected_image_id TEXT,
    history_json TEXT,
    created_at TEXT,
    CHECK (kind IN ('character_sheet', 'scene_plate', 'prop', 'costume_derive', 'keyframe'))
);

CREATE TABLE IF NOT EXISTS memory (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    text TEXT,
    episode_code TEXT,
    object_id TEXT,
    embedding BLOB,
    created_at TEXT NOT NULL,
    CHECK (kind IN ('episode_summary', 'event', 'scene_desc', 'arc_fragment', 'foreshadow_text'))
);

CREATE TABLE IF NOT EXISTS qc_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_code TEXT,
    segment_id TEXT,
    gate TEXT NOT NULL,
    verdict TEXT NOT NULL,
    details_json TEXT,
    created_at TEXT NOT NULL,
    CHECK (gate IN ('deterministic', 'text', 'visual')),
    CHECK (verdict IN ('pass', 'retry', 'fail'))
);

CREATE TABLE IF NOT EXISTS continuity (
    episode_code TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS glossary (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    zh TEXT,
    en TEXT,
    ja TEXT,
    CHECK (kind IN ('character', 'location', 'item', 'term'))
);

-- v2.0 production layer: Shot ≠ Segment. H3 units are segments on a chain.
CREATE TABLE IF NOT EXISTS productions (
    id TEXT PRIMARY KEY,
    story_id TEXT,
    kind TEXT NOT NULL DEFAULT 'series',
    film_id TEXT,
    series_id TEXT,
    season_id TEXT,
    status TEXT,
    target_duration REAL,
    primary_language TEXT,
    languages_json TEXT,
    style_preset TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (kind IN ('film', 'series', 'short'))
);

CREATE TABLE IF NOT EXISTS sequences (
    id TEXT PRIMARY KEY,
    production_id TEXT,
    episode_code TEXT NOT NULL,
    seq INTEGER NOT NULL,
    title TEXT,
    synopsis TEXT
);

CREATE TABLE IF NOT EXISTS beats (
    id TEXT PRIMARY KEY,
    sequence_id TEXT,
    scene_id TEXT,
    seq INTEGER NOT NULL,
    purpose TEXT,
    text TEXT
);

CREATE TABLE IF NOT EXISTS shots (
    id TEXT PRIMARY KEY,
    episode_code TEXT NOT NULL,
    scene_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    duration REAL,
    purpose TEXT,
    size TEXT,
    camera TEXT,
    h3_mode TEXT,
    character_id TEXT,
    match_cut INTEGER NOT NULL DEFAULT 0,
    chain_id TEXT,
    first_frame_prompt TEXT,
    h3_prompt TEXT,
    refs_json TEXT,
    plate_id TEXT,
    costume_ids_json TEXT,
    line_json TEXT,
    eyeline TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS generation_plans (
    id TEXT PRIMARY KEY,
    shot_id TEXT,
    strategy TEXT,
    segment_duration REAL,
    segment_count INTEGER,
    provider TEXT,
    model TEXT,
    h3_mode TEXT,
    reference_policy TEXT,
    continuity_policy TEXT,
    repair_policy TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generation_tasks (
    id TEXT PRIMARY KEY,
    plan_id TEXT,
    segment_id TEXT,
    seq INTEGER,
    status TEXT,
    seed INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generation_results (
    id TEXT PRIMARY KEY,
    segment_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    path TEXT NOT NULL,
    seed INTEGER,
    h3_mode TEXT,
    status TEXT,
    qc_verdict TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS repair_tasks (
    id TEXT PRIMARY KEY,
    target_type TEXT,
    target_id TEXT,
    failure_type TEXT,
    qc_report_id INTEGER,
    diagnosis TEXT,
    repair_strategy TEXT,
    old_prompt TEXT,
    new_prompt TEXT,
    attempt INTEGER NOT NULL DEFAULT 1,
    max_attempt INTEGER NOT NULL DEFAULT 2,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    completed_at TEXT
);

-- SFX cue resolution: Freesound is always attempted before MOSS-SoundEffect v2.
-- One row per resolved cue (a board/segment SFX need, or a shared ambience bed).
-- provenance_json carries the full audit trail: query/tags/scores for Freesound
-- hits (creator, source_url, license, sample_rate, freesound id) or the MOSS
-- seed/prompt/duration/commit for a fallback generation. content_hash is the
-- sha256 of the normalized PCM WAV, used for cache dedup and provenance.
-- r2_key follows anime_factory.r2_paths sfx helpers: episodes/<EP>/audio/sfx/...
-- for episode-local cues, shared/sfx/... for reusable/ambience assets. Model
-- weights are never written to R2 (see anime_factory.sfx_moss).
CREATE TABLE IF NOT EXISTS sfx_cues (
    id TEXT PRIMARY KEY,
    episode_code TEXT,
    segment_id TEXT,
    cue_key TEXT NOT NULL,
    prompt TEXT,
    tags_json TEXT,
    duration_target REAL,
    bus TEXT NOT NULL DEFAULT 'sfx',
    source TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    license TEXT,
    creator TEXT,
    source_url TEXT,
    content_hash TEXT,
    sample_rate INTEGER,
    duration_actual REAL,
    r2_key TEXT,
    local_path TEXT,
    shared INTEGER NOT NULL DEFAULT 0,
    provenance_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (bus IN ('sfx', 'ambience')),
    CHECK (status IN ('pending', 'resolved', 'failed', 'rejected'))
);
