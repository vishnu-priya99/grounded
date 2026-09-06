# Postmortem — 2026-03-09 EMEA API degradation

| | |
| --- | --- |
| **Incident ID** | INC-2026-0311 |
| **Severity** | Sev-1 |
| **Duration** | 47 minutes (14:02–14:49 UTC) |
| **Regions affected** | EMEA only |
| **Customer impact** | Elevated error rates (peak 38% of requests) and p95 latency up to 9s on the EMEA API cluster. US and APAC unaffected. |
| **Author** | Platform on-call (secondary) |
| **Status** | Closed — action items tracked in PLAT-4821..4826 |

## Summary

At 14:02 UTC a routine deploy of `services/api` to the EMEA region rolled out a
configuration change that lowered the database connection-pool ceiling from 200
to 20. Under normal EMEA afternoon load the pool was exhausted within ~90 seconds,
causing request queueing, timeouts, and a partial outage. The change was rolled
back at 14:41; the cluster recovered by 14:49 once connection backlog drained.

## Timeline (UTC)

| Time | Event |
| --- | --- |
| 14:00 | Deploy pipeline for EMEA `api` starts (change: pool tuning + logging). |
| 14:02 | New pods healthy; config with `db_pool_max=20` live. |
| 14:04 | First alert: EMEA API 5xx rate > 5%. Auto-escalation timer starts. |
| 14:06 | On-call primary acknowledges (within the 15-minute escalation window). |
| 14:12 | #incident-emea-api opened; Incident Commander assigned. |
| 14:20 | Dashboards show connection-pool saturation; deploy suspected. |
| 14:33 | Decision to roll back; `make release REF=<prev-sha>` started. |
| 14:41 | Rollback pods healthy; error rate falling. |
| 14:49 | p95 latency and error rate back to baseline. Incident downgraded. |
| 15:30 | Customer status page updated to "resolved". |

## Root cause

The pool ceiling was parameterised in a shared config template. A find-and-replace
during the change edited the wrong key, and the value `20` (intended for a log
sampling rate) landed in `db_pool_max`. CI did not catch it because the config
schema allowed any positive integer and no load test runs against EMEA-scale
traffic in the pipeline.

## What went well

- Alerting fired within 2 minutes of impact.
- Acknowledgement and IC assignment were inside SLA.
- Rollback via `make release REF=` worked exactly as documented.

## What went wrong

- No schema bound on `db_pool_max` (should be ≥ 50).
- Staging load does not resemble EMEA peak, so the fault was invisible pre-prod.
- The config diff was hard to review — 300 lines of generated YAML.

## Action items

| ID | Action | Owner | Due |
| --- | --- | --- | --- |
| PLAT-4821 | Add min/max bounds to the config schema for pool sizes | Platform | 2026-03-20 |
| PLAT-4822 | Add an EMEA-scale load test to the deploy pipeline | Platform | 2026-04-03 |
| PLAT-4823 | Render config diffs as semantic key/value changes in PR | DevEx | 2026-04-10 |
| PLAT-4824 | Alert on connection-pool utilisation > 80% | Platform | 2026-03-18 |
| PLAT-4825 | Runbook: "API 5xx spike after deploy" | On-call | 2026-03-25 |
| PLAT-4826 | Postmortem review in the next platform all-hands | Platform Lead | 2026-03-13 |
