# Postmortem: Checkout latency spike — 2024-02-14

Severity: SEV-2. Duration: 14:05–15:40 UTC. Author: Rui (payments team).

## Impact

p99 checkout latency rose from 800ms to 9s. Approximately 6% of checkout attempts timed out. No data loss.

## Root cause

The payments service connection pool to Postgres was capped at 20 connections. A Valentine's Day traffic surge (3.1x baseline) exhausted the pool; requests queued at the pool rather than failing fast.

## Resolution

Raised the pool cap to 80 and enabled the pool's 500ms acquire timeout so overload sheds load instead of queueing.

## Action items

1. Add pool saturation alerts at 70% utilization (owner: Rui, done 2024-02-20).
2. Load-test checkout at 4x baseline before seasonal events (owner: Dana).
3. Evaluate pgbouncer for connection multiplexing (owner: Rui) — see ADR-011.
