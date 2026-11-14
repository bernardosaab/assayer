-- 0001: initial schema. See ADR-0003 (SQLite), ADR-0004 (attempt durability), ADR-0006 (ledger).

CREATE TABLE run (
    run_id          TEXT PRIMARY KEY,
    eval_name       TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    config_hash     TEXT NOT NULL,
    budget_micros   INTEGER NOT NULL CHECK (budget_micros >= 0),
    status          TEXT NOT NULL CHECK (status IN ('open', 'halted', 'complete')),
    replay_of       TEXT REFERENCES run (run_id),
    meta_json       TEXT NOT NULL
);

-- Snapshot of every fixture at run creation: a run stays reproducible if YAML changes later.
CREATE TABLE run_case (
    run_id          TEXT NOT NULL REFERENCES run (run_id),
    case_id         TEXT NOT NULL,
    ordinal         INTEGER NOT NULL,
    fixture_hash    TEXT NOT NULL,
    fixture_json    TEXT NOT NULL,
    PRIMARY KEY (run_id, case_id)
);

CREATE TABLE attempt (
    attempt_id      INTEGER PRIMARY KEY,
    run_id          TEXT NOT NULL,
    case_id         TEXT NOT NULL,
    repeat          INTEGER NOT NULL CHECK (repeat >= 0),
    status          TEXT NOT NULL
                    CHECK (status IN ('pending', 'in_progress', 'succeeded', 'harness_error')),
    output_json     TEXT,
    error_kind      TEXT,
    error_msg       TEXT,
    try_count       INTEGER NOT NULL DEFAULT 0,
    started_at      TEXT,
    finished_at     TEXT,
    UNIQUE (run_id, case_id, repeat),
    FOREIGN KEY (run_id, case_id) REFERENCES run_case (run_id, case_id)
);
CREATE INDEX ix_attempt_run_status ON attempt (run_id, status);

-- Reserve-then-settle cost ledger. Committed spend per run is
--   settled -> actual, reserved/orphaned -> reserved, released -> 0.
CREATE TABLE cost_ledger (
    entry_id            INTEGER PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES run (run_id),
    attempt_id          INTEGER REFERENCES attempt (attempt_id),
    judgement_key       TEXT,
    purpose             TEXT NOT NULL CHECK (purpose IN ('agent', 'simulator', 'judge')),
    model               TEXT NOT NULL,
    status              TEXT NOT NULL
                        CHECK (status IN ('reserved', 'settled', 'released', 'orphaned')),
    reserved_micros     INTEGER NOT NULL CHECK (reserved_micros >= 0),
    actual_micros       INTEGER CHECK (actual_micros >= 0),
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    session_id          TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    settled_at          TEXT
);
CREATE INDEX ix_ledger_run ON cost_ledger (run_id, status);
CREATE INDEX ix_ledger_attempt ON cost_ledger (attempt_id);

CREATE TABLE check_result (
    attempt_id      INTEGER NOT NULL REFERENCES attempt (attempt_id),
    check_id        TEXT NOT NULL,
    check_version   TEXT NOT NULL,
    passed          INTEGER NOT NULL CHECK (passed IN (0, 1)),
    detail_json     TEXT NOT NULL,
    PRIMARY KEY (attempt_id, check_id, check_version)
);

CREATE TABLE judgement (
    judgement_id    INTEGER PRIMARY KEY,
    attempt_id      INTEGER NOT NULL REFERENCES attempt (attempt_id),
    judge_id        TEXT NOT NULL,
    judge_version   TEXT NOT NULL,
    evidence_hash   TEXT NOT NULL,
    judge_model     TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('done', 'harness_error')),
    verdict_final   TEXT,
    verdict_raw     TEXT,
    reasks          INTEGER NOT NULL DEFAULT 0,
    dropped         INTEGER NOT NULL DEFAULT 0,
    verdict_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    UNIQUE (attempt_id, judge_id, judge_version)
);
CREATE INDEX ix_judgement_cache ON judgement (judge_id, judge_version, evidence_hash, status);

CREATE TABLE grade (
    attempt_id      INTEGER NOT NULL REFERENCES attempt (attempt_id),
    grader_version  TEXT NOT NULL,
    outcome         TEXT NOT NULL CHECK (outcome IN ('pass', 'fail', 'harness_error')),
    reasons_json    TEXT NOT NULL,
    PRIMARY KEY (attempt_id, grader_version)
);

-- Human labels for judge calibration (the judge's judge).
CREATE TABLE calibration (
    attempt_id      INTEGER NOT NULL REFERENCES attempt (attempt_id),
    judge_id        TEXT NOT NULL,
    judge_version   TEXT NOT NULL,
    labeller        TEXT NOT NULL,
    human_verdict   TEXT NOT NULL CHECK (human_verdict IN ('pass', 'fail')),
    note            TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (attempt_id, judge_id, judge_version, labeller)
);
