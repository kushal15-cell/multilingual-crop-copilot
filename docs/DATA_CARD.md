# Data card and alignment contract

## Image data

Expected organization is one class directory per label. Preserve the original source, attribution, and license separately. The preparation script normalizes pixels for exact-duplicate hashing, groups duplicates, and produces a manifest. It does not detect every near-duplicate or multiple photos of the same plant.

## Market data

Required grain is one row per `date + market + crop`. Prices use INR per quintal unless explicitly
changed. A provider-specific ingestion job must map spelling variants and units into this contract
before model training. The bundled bootstrap table contains seven Karnataka market names and is
marked `synthetic_demo_v2` on every row. It is only for software tests and pipeline demonstrations;
it is not mandi evidence and must remain visibly labeled in every user-facing output.

The training target is modal price seven days after each feature row. The chronological split purges a seven-day gap between training features and the test window so future target labels cannot leak across the boundary.

## Weather data

The live adapter obtains current context by latitude/longitude. The UI supplies approximate city
coordinates from a district/nearest-market selector and allows an optional precise override.
Historical price training expects weather already aligned by market/date. Define how each market
maps to coordinates and handle timezone/date boundaries explicitly.

## Cross-source alignment

Disease predictions are not joined to historical yield because the MVP lacks reliable disease severity, treatment, and resulting-yield labels. The market model therefore provides economic context only. It must not be described as the causal cost of the diagnosed disease.

## Sensitive information

Farm coordinates and voice recordings can be personal or commercially sensitive. The current API deletes temporary upload files after processing and monitoring excludes question/image content. Before a real pilot, define consent, encryption, retention, deletion, access, and data-residency policies.
