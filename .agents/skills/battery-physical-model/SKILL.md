---
name: battery-physical-model
description: Use when implementing or modifying the battery-voltage signal, including lightweight physically-inspired behavior for resting, charging, load-induced drops, degradation, and maintenance-related recovery.
---

## When to use this

Use this skill when a task involves:
- adding or changing `battery_voltage_v`
- adding latent battery-health or SOC-like state
- simulating battery decay, charging, or replacement behavior
- validating maintenance events related to electrical faults or battery replacement

Do not use this skill for generic signal plumbing unless the change materially affects battery behavior.

## Goal

Provide a lightweight, physically-inspired battery-voltage model suitable for synthetic maintenance benchmarks.

This is not a full electrochemical simulation. The objective is interpretable dynamics that make maintenance reasoning plausible.

## Preferred signal

- `battery_voltage_v`

Optional latent variables:
- SOC-like state
- battery health parameter
- current operational mode

## Minimum expected behaviors

1. Resting / vehicle off:
   - voltage remains relatively stable
   - may decay slowly over long parked periods

2. Ignition on / charging:
   - voltage should be higher than resting level
   - charging regime should be distinguishable from off-state behavior

3. Load or degraded battery:
   - lower baseline voltage or stronger drop behavior is possible

4. After maintenance such as battery replacement:
   - battery-health parameters may improve
   - voltage profile should reflect improved behavior

## Modeling guidance

A simple state-based or equivalent-circuit-inspired approximation is enough.

Suggested approach:
- determine an operational mode from latent state and ignition
- update a small latent variable set deterministically
- emit voltage as a function of:
  - mode
  - battery health
  - SOC-like variable
  - optional load factor
  - small bounded noise if the repo convention allows it

## Plausibility guidance

Document the expected ranges used by the repo.
The exact numbers may depend on vehicle type and simulation assumptions, but the following qualitative relationships should hold:
- charging voltage > resting voltage
- severely degraded battery behavior is noticeably worse
- long off periods can slightly worsen resting behavior
- replacement improves post-event behavior

## Inputs this skill expects

Typical inputs:
- latent state sequence
- ignition signal
- timestep
- deterministic seed
- optional battery-health scenario parameters
- optional maintenance-event schedule

## Outputs this skill should support

- time-indexed `battery_voltage_v`
- optional latent battery-health trace if the repo exposes it
- plausible pre/post maintenance behavior

## Implementation checklist

1. Add or update the battery model module.
2. Keep assumptions explicit in code comments and docs.
3. Make behavior deterministic with fixed seed and config.
4. Support at least one maintenance-relevant scenario such as battery replacement.
5. Add tests for:
   - resting vs charging behavior
   - mild parked decay
   - degraded-battery scenario
   - post-replacement improvement

## Invariants

- voltage values remain within documented plausible bounds
- charging behavior is distinguishable from resting behavior
- same seed + same config => same trace
- maintenance-event parameter changes have visible, testable effects

## Files typically touched

- `src/telematics/signals/battery.py`
- config modules for battery parameters
- `tests/telematics/test_battery_model.py`

## Validation before finishing

Run:
- battery unit tests
- one scenario with a battery-replacement event
- deterministic regression checks

Inspect traces visually if the repo has a plotting helper, but do not rely only on plots; keep assertions in tests.

## Definition of done

A change using this skill is done when:
- `battery_voltage_v` is generated deterministically
- resting, charging, degraded, and repaired behaviors are distinguishable
- assumptions and ranges are documented
- tests cover maintenance-relevant cases
