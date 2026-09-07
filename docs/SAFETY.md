# Advisory safety policy

## Automatically abstain

- Vision confidence below the configured threshold.
- Unreadable or too-small image.
- Unsupported language or malformed location.
- Model artifact is unavailable.

## Mandatory review

- Pesticide, fungicide, insecticide, herbicide, dosage, concentration, mixing, frequency, withholding/pre-harvest interval, or restricted-chemical content.
- Diagnostic confidence below the expert-review threshold.
- Advice grounded in any record not marked expert validated.
- Other locally defined high-risk categories added to `configs/risk_policy.yaml`.

## Reviewer expectations

The reviewer verifies the diagnosis limitations, regional registration and label requirements, crop stage, local weather, protective equipment, environmental risk, pre-harvest interval, and whether non-chemical measures are appropriate. Approval must identify the reviewer. Rejection requires a reason.

## Incident response

Disable advice generation, retain audit metadata under the approved retention policy, notify the system owner and agronomy lead, investigate model/data/prompt/provider changes, and require explicit sign-off before restoring service. Never use this system for poisoning emergencies; direct users to local emergency and poison-control services.

