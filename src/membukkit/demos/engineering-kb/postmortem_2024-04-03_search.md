# Postmortem: Search returning stale results — 2024-04-03

Severity: SEV-3. Duration: ~36 hours (detected late). Author: Imani (search team).

## Impact

Newly published products did not appear in search results for up to 36 hours. Detected by a merchant complaint, not by monitoring.

## Root cause

The indexing worker's Kafka consumer group silently stopped committing offsets after a broker rolling restart on April 1st. The worker kept running (health check green) but reprocessed the same batch in a loop.

## Resolution

Restarted the consumer group; added an end-to-end freshness probe (publish a canary product, assert it is searchable within 5 minutes).

## Action items

1. Freshness probe in production monitoring (owner: Imani, done 2024-04-10).
2. Alert on consumer group offset lag > 10k (owner: Tom, done 2024-04-08).
3. Health check must verify offset progress, not just process liveness (owner: Tom).
