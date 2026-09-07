# Architecture and trust boundaries

## Components

| Component | Input | Output | Failure behavior |
|---|---|---|---|
| Vision classifier | RGB field image | calibrated label, confidence, alternatives | reject invalid image or abstain |
| Weather adapter | latitude, longitude | current measurements with timestamp/source | explicit unavailable context |
| Market forecaster | normalized historical CSV | latest price, seven-day estimate, trend | explicit unavailable context |
| Knowledge base | crop, predicted disease, language | source-tagged passages | empty evidence set |
| Generator | evidence plus structured context | typed private draft | deterministic safe template |
| Risk policy | question, prediction, draft, evidence | review decision and reasons | conservative review |
| Review store | private draft and audit metadata | pending/approved/rejected state | transactional conflict on repeat decision |
| Farmer API | image, question, location | safe response or neutral queue message | never exposes pending draft |

## Trust boundaries

- Farmer input is untrusted. Upload size and media type are checked; Pillow validates image bytes.
- Model output is probabilistic, not a diagnosis. Confidence is calibrated on validation data and compared with an abstention threshold.
- External weather and market data can be unavailable or stale. Their source and timestamp travel with the context.
- Retrieved passages are not automatically authoritative. Each record has an `expert_validated` flag.
- LLM output is untrusted. It is parsed into a typed schema, source IDs are allow-listed, and deterministic policy runs after generation.
- The review database contains private drafts. Farmer-facing status responses expose approved advice only.
- The reviewer API key is an MVP control, not production-grade identity. Replace it with organizational SSO/RBAC and an immutable audit log.

## State machine

```text
image -> invalid/low confidence -> NEED_MORE_INFORMATION
image -> draft -> safe and validated -> READY
image -> draft -> risky/uncertain/unvalidated -> PENDING_REVIEW
PENDING_REVIEW -> agronomist approves edited text -> APPROVED
PENDING_REVIEW -> agronomist rejects -> REJECTED
```

## Scaling path

Replace local files with object storage, SQLite with PostgreSQL, in-process jobs with a queue, API-key review with SSO/RBAC, JSONL monitoring with an event pipeline, and synchronous inference with GPU model serving. These changes do not alter the public request/response schema or safety state machine.

