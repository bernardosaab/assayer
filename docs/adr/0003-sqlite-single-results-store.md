# 0003. SQLite as the single results store, WAL mode, one writer thread

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

Every durable fact of an eval lives in the store: runs, the fixture snapshot, attempts and their
outputs, the cost ledger, check results, judgements, grades and calibration labels. The store
must survive a `kill -9`, hold an exact budget while many asyncio tasks (and sometimes two
processes) reserve money at once, and be queryable for reports and diffs. It runs on a laptop
and in CI, where nobody wants to operate a database server.

## Considered options

1. JSONL files per run, rewritten or appended by the harness.
2. A server database (Postgres).
3. One SQLite file in WAL mode, with every access going through a single connection.

## Decision

Option 3. `SqliteStore` (`src/assayer/adapters/store/sqlite_store.py`) implements every store
port against one file (`store_path`, default `.assayer/results.sqlite` in `assayer.toml`).

- `_connect` sets `journal_mode=WAL`, `foreign_keys=ON`, `busy_timeout=5000`,
  `synchronous=NORMAL`, and `isolation_level=None` so transactions are explicit.
- One connection, one thread: a `ThreadPoolExecutor(max_workers=1)`. Every public method
  submits a small synchronous function through `_call` (`run_in_executor`), so writes from
  concurrent tasks are serialized without locks in application code.
- Read-modify-write operations (`_create_or_load_run`, `_ensure_attempts`, `_claim_attempt`,
  `_reserve`, `_recover_interrupted`, `_retry_errors`) run inside `_tx()`, which issues
  `BEGIN IMMEDIATE`. A second process therefore waits on the write lock instead of reading a
  stale budget. A lock held past `busy_timeout` becomes `StoreLocked`, a harness error
  (`store_locked`).
- Invariants live in the schema (`migrations/0001_init.sql`): `CHECK` constraints on every
  status column, `UNIQUE (run_id, case_id, repeat)` on `attempt`, money as non-negative
  `INTEGER` micro-dollars.
- `migrator.py` applies numbered `NNNN_name.sql` files forward-only. Each migration runs in its
  own `BEGIN IMMEDIATE` and re-reads `schema_version` inside it, so two processes opening a
  fresh file cannot apply the same migration twice.

The budget invariant is tested across both kinds of concurrency in
`tests/property/test_ledger_invariants.py`: 200 concurrent `reserve` calls against a
1,000-micro budget, and two spawned processes that share one budget and together get exactly
`budget / amount` successful reservations.

## Consequences

**Positive:** no service to run; a run's whole history is one file you can copy or attach to a
bug report; ACID transactions and schema constraints enforce the attempt and ledger invariants;
several processes on the same machine can work on one run safely.

**Negative / trade-offs:**

- Single machine only. WAL depends on shared memory, so the file must not sit on a network
  filesystem (NFS/SMB). Distributed workers would need a different store.
- Every store call goes through one thread, reads included. A slow report query delays attempt
  claims and cost reservations queued behind it.
- `synchronous=NORMAL` in WAL mode can lose the last committed transactions on power loss or
  an OS crash (a process crash is safe). After such a loss the ledger could undercount spend.
- `_claim_attempt` uses `UPDATE ... RETURNING`, which needs SQLite 3.35 or newer. An older
  system library fails at the first claim.
- Outputs, verdicts and run metadata are stored as JSON `TEXT` (`output_json`,
  `verdict_json`, `meta_json`), so SQL cannot query inside them without the json1 functions.
