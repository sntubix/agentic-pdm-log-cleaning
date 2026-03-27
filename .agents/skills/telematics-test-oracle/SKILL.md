---
name: telematics-test-oracle
description: Use when writing or updating tests for telematics generation, including deterministic seeds, signal ranges, cross-signal consistency, schema toggles, and end-to-end maintenance-aware smoke validations.
---

## When to use this

Use this skill when a task involves:
- adding unit tests for telematics signals
- adding integration tests for environment outputs
- defining invariants for state-driven telemetry
- regression checks for deterministic generation
- validating smoke scenarios end-to-end

This skill can be paired with other telematics skills whenever tests are part of the work.

## Goal

Make telematics changes safe, reproducible, and reviewable through strong invariants and targeted regression tests.

## Test layers to prefer

### 1. Unit tests
Use for:
- state-machine behavior
- GPS/geofence calculations
- motion and trip logic
- battery and coolant dynamics
- derived maintenance evidence

### 2. Integration tests
Use for:
- environment output schema
- feature toggles
- aligned maintenance-log + telematics export
- backward compatibility with old modes

### 3. Smoke scenarios
Use for:
- workshop arrival and departure
- battery replacement
- overheating repair
- implausible maintenance record

## Core invariants to encode

Always prefer concrete invariants over vague visual checks.

Examples:
- same seed + same config => identical output
- speed is non-negative
- engine hours are non-decreasing
- distance to workshop is non-negative
- `inside_workshop_geofence` matches the chosen threshold
- in-service periods do not show sustained driving
- off-state coolant cools toward ambient
- charging battery voltage is distinguishable from resting voltage
- workshop pass-by without dwell is not labeled as repair completion
- old pipeline still works when new signals are disabled

## Test-writing rules

1. Keep tests deterministic.
2. Use small synthetic fixtures when possible.
3. Assert relationships, not just existence of columns.
4. Include at least one negative case for derived maintenance logic.
5. Prefer stable numerical tolerances over brittle exact-float expectations unless exact determinism is guaranteed.

## Inputs this skill expects

Typical inputs:
- signal modules or derived-feature modules
- config fixtures
- deterministic seed
- known scenario templates
- export schemas

## Outputs this skill should support

- focused unit tests
- integration tests for schemas and toggles
- end-to-end smoke checks
- regression tests for deterministic output

## Implementation checklist

1. Identify the highest-risk invariants for the change.
2. Add the smallest unit tests that pin those invariants down.
3. Add integration coverage if schemas or exports changed.
4. Add or update a smoke scenario if the feature affects end-to-end reasoning.
5. Keep test names descriptive and failure messages informative.

## Files typically touched

- `tests/telematics/`
- `tests/integration/`
- test fixtures
- smoke configs if used in tests

## Validation before finishing

Run:
- the new targeted unit tests
- affected integration tests
- one deterministic smoke check if relevant

Follow repo-standard verification commands from `AGENTS.md` where present.

## Definition of done

A change using this skill is done when:
- key invariants are encoded in automated tests
- deterministic behavior is checked
- negative cases are covered where appropriate
- end-to-end behavior is exercised when the feature affects pipeline outputs
