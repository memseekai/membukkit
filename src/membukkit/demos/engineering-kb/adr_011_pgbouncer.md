# ADR-011: Adopt pgbouncer for payments Postgres — 2024-03-12

Status: Accepted. Deciders: Rui, Dana, platform team.

## Context

The February 14 checkout incident showed the payments service saturates its direct Postgres connection pool under surge traffic. Raising per-service caps does not scale: Postgres degrades past ~500 total connections, and we run 9 service replicas.

## Decision

Deploy pgbouncer in transaction-pooling mode in front of the payments Postgres primary. Services connect to pgbouncer with generous local pools; pgbouncer multiplexes onto 40 server connections.

## Consequences

- Prepared statements require `max_prepared_statements` tuning (transaction mode caveat).
- Session-level features (advisory locks, LISTEN/NOTIFY) must not be used by payments code — enforced by a lint rule.
- Rollout completed 2024-03-28; surge headroom validated at 5x baseline in the load test on 2024-04-15.
