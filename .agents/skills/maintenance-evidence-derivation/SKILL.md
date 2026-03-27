---
name: maintenance-evidence-derivation
description: Use when implementing or modifying derived maintenance-oriented evidence from raw telematics, such as workshop dwell, repair-window candidates, repair completion candidates, and first trip after workshop.
---

## When to use this

Use this skill when a task involves:
- deriving maintenance evidence from raw telematics
- adding rule-based repair-window logic
- computing workshop dwell summaries
- identifying first trip after workshop
- materializing maintenance-oriented helper columns or evidence tables

Do not use this skill for raw-signal generation unless the task explicitly includes derived maintenance features.

## Goal

Convert raw telematics into interpretable maintenance evidence that downstream agents, validators, and fine-tuning pipelines can use directly.

## Preferred derived features

Use project naming conventions if already established. Otherwise prefer:
- `vehicle_is_stationary`
- `time_inside_workshop_min`
- `repair_window_candidate`
- `repair_completion_candidate`
- `first_trip_after_workshop`
- `fault_active`
- `fault_cleared_after_service`

## Core reasoning patterns

### Workshop dwell
A workshop dwell interval is typically supported by:
- near or inside workshop geofence
- stationary behavior
- sufficient dwell duration

### Repair window candidate
A repair window candidate is typically supported by:
- workshop dwell
- low or zero speed
- ignition mostly off or reduced activity
- optional supporting fault or maintenance context

### Repair completion candidate
A repair completion candidate is typically supported by:
- prior workshop dwell
- later restart or resumed motion
- departure from workshop area or first outbound trip

### Avoid false positives
Do not treat every workshop pass-by as repair.
Require a meaningful dwell pattern and, where useful, a minimum stationary duration.

## Inputs this skill expects

Typical inputs:
- workshop-distance or geofence features
- speed and stationary features
- ignition behavior
- trip segmentation features
- optional fault or maintenance-event context
- deterministic thresholds from config

## Outputs this skill should support

- row-level derived evidence columns
- interval-level workshop visits or repair windows
- deterministic evidence for downstream tools and evaluations

## Implementation checklist

1. Identify authoritative raw input columns.
2. Make thresholds and heuristic rules explicit in config or constants.
3. Derive row-level evidence first, then interval-level summaries if needed.
4. Add tests that include:
   - true repair window
   - false workshop pass-by
   - repair completion after dwell and departure
   - no-completion case
5. Document the rule logic.

## Invariants

- evidence is deterministic for fixed inputs and config
- derived columns are logically consistent with raw telematics
- no repair completion without some prior candidate repair window unless explicitly allowed
- workshop pass-by without dwell should not be mislabeled as repair

## Files typically touched

- `src/telematics/derived_features.py`
- `tests/telematics/test_derived_features.py`
- schema docs for derived outputs

## Validation before finishing

Run:
- derived-feature unit tests
- one smoke scenario with workshop arrival, dwell, and departure
- one negative scenario where the vehicle passes near a workshop but never stops

## Definition of done

A change using this skill is done when:
- maintenance-evidence features are derived deterministically
- repair windows and completion logic are documented
- tests cover true and false positive patterns
- downstream code can consume the evidence without recomputing ad hoc logic
