# ADR-014: Standardize on Kafka for inter-service events — 2024-05-06

Status: Accepted. Deciders: Imani, Tom, platform team.

## Context

Three ad-hoc mechanisms exist for service-to-service events: Kafka (search indexing), Redis pub/sub (notifications), and direct HTTP webhooks (billing). Each has separate retry, monitoring, and schema conventions; the April search incident showed how weak our Kafka operational practices were even on the main path.

## Decision

All new inter-service events go through Kafka with schema-registry-enforced Avro schemas. Redis pub/sub and internal webhooks are deprecated for new use; existing users migrate by Q4 2024.

## Consequences

- Notifications team migrates first (June); billing webhooks last (October) due to the invoicing freeze in September.
- Every consumer group must ship offset-lag alerts from day one (lesson from the 2024-04-03 postmortem).
- Platform team owns a shared consumer library with mandatory freshness probes.
