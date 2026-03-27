---
name: telematics-benchmark-scenarios
description: Use when creating or modifying deterministic benchmark scenarios and smoke tests that exercise workshop-aware telematics, maintenance completion, battery events, coolant anomalies, or implausible maintenance records.
---

## When to use this

Use this skill when a task involves:
- adding benchmark configs
- creating smoke scenarios
- designing end-to-end telematics-aware maintenance cases
- producing small deterministic datasets for regression checks
- documenting scenario intent for paper experiments

Do not use this skill for low-level signal implementation unless the issue explicitly includes scenario creation.

## Goal

Create small, deterministic scenarios that prove the telematics stack is useful for maintenance reasoning.

## Canonical scenario templates

Prefer starting from these:

1. Workshop arrival, dwell, and departure
   - vehicle approaches workshop
   - remains stationary near or inside geofence
   - later resumes operation
   - should support repair completion inference

2. Battery replacement
   - weak battery behavior before service
   - service window at workshop
   - improved voltage behavior after service

3. Overheating-related repair
   - abnormal coolant behavior before service
   - service window
   - improved thermal behavior after service

4. Implausible maintenance record
   - maintenance claimed while vehicle keeps moving far from workshop
   - should support inconsistency detection

## Scenario design rules

- keep scenarios short enough for smoke testing
- make them deterministic
- make the causal story obvious
- prefer one strong pattern per scenario
- include a brief textual description in docs or config comments

## Inputs this skill expects

Typical inputs:
- latent-state templates
- enabled signals
- maintenance-log event templates
- seed
- benchmark or smoke config structure

## Outputs this skill should support

- runnable benchmark configs
- smoke datasets
- small regression fixtures
- documentation of scenario intent

## Implementation checklist

1. Pick one benchmark story per scenario.
2. Enable only the signals needed to demonstrate it clearly.
3. Keep the sequence short but expressive.
4. Add assertions or checks that verify the expected pattern.
5. Document why the scenario exists and what it should prove.

## Invariants

- scenarios are reproducible with fixed seed
- scenario intent is interpretable from generated outputs
- smoke scenarios run quickly
- each scenario has at least one explicit expected outcome

## Files typically touched

- `configs/benchmarks/`
- `tests/integration/`
- `docs/experiments.md`

## Validation before finishing

Run:
- the smoke scenario end-to-end
- any associated integration assertions
- a quick review of generated artifacts to confirm the intended pattern is visible

## Definition of done

A change using this skill is done when:
- deterministic telematics-aware scenarios exist
- they run end-to-end
- they validate at least one maintenance-relevant reasoning pattern
- their intent is documented
