---
name: telematics-state-machine
description: Use when implementing or modifying the latent vehicle-state process that drives telematics generation, including allowed states, transition rules, deterministic seeding, and state-conditioned signal behavior.
---

## When to use this

Use this skill when a task involves any of the following:
- adding or editing the latent telematics state machine
- defining new vehicle operating states
- changing transition rules between states
- wiring raw signals to depend on latent state
- making telematics generation more coherent across GPS, speed, ignition, or maintenance-related events

Do not use this skill for isolated schema updates, documentation-only edits, or benchmark-only tasks unless they require changing the actual state logic.

## Goal

Create a lightweight, deterministic, maintenance-aware state process that downstream signal generators can rely on.

The state machine is not a full vehicle simulator. It is a compact latent process that produces coherent behavior across telematics signals and maintenance events.

## Canonical states

Unless the issue explicitly changes them, prefer this default state vocabulary:

- `driving`
- `parked_normal`
- `approaching_workshop`
- `waiting_for_service`
- `in_service`
- `post_service_test`
- `returned_to_operation`

You may add states only when there is a clear maintenance or signal-generation need.

## Design rules

1. Keep the implementation deterministic for a fixed seed and config.
2. Make state transitions interpretable and easy to test.
3. Keep the state machine small; prefer derived features over excessive state proliferation.
4. Every state should have clear behavioral implications for at least one downstream signal.
5. Every new state must document:
   - expected predecessor states
   - expected successor states
   - typical duration behavior
   - expected signal profile

## Default behavioral expectations

- `driving`:
  - speed generally above zero
  - ignition on
  - GPS changes over time
- `parked_normal`:
  - speed near zero
  - ignition usually off
  - GPS roughly stable
- `approaching_workshop`:
  - distance to workshop tends to decrease
  - ignition on
  - movement still plausible
- `waiting_for_service`:
  - near workshop
  - stationary or nearly stationary
  - ignition may vary but is usually not sustained high-activity driving
- `in_service`:
  - inside workshop geofence or very near it
  - speed near zero
  - ignition mostly off
  - allows maintenance-event alignment
- `post_service_test`:
  - ignition on
  - limited movement may resume
  - workshop proximity may still be high initially
- `returned_to_operation`:
  - movement resumes more normally
  - distance from workshop may increase

## Transition guidance

Prefer transitions like:
- `driving -> parked_normal`
- `driving -> approaching_workshop`
- `approaching_workshop -> waiting_for_service`
- `waiting_for_service -> in_service`
- `in_service -> post_service_test`
- `post_service_test -> returned_to_operation`

Avoid unrealistic direct jumps unless the scenario explicitly requires them.

Question every transition that skips all workshop-related states when the benchmark scenario is intended to represent maintenance completion.

## Inputs this skill expects

Typical inputs:
- global seed
- timestep or sampling interval
- scenario config
- workshop definitions or references
- maintenance-event schedule or latent maintenance intent
- optional route/usage parameters

## Outputs this skill should support

The state machine should provide, directly or indirectly:
- latent state per timestep
- deterministic state sequence generation
- helper methods for downstream signals
- optional transition markers or state-change events

## Implementation checklist

1. Identify the current telematics entry point and generation flow.
2. Add or update the state-machine module without breaking old odometer-only behavior.
3. Expose a compact API that downstream signal modules can consume.
4. Keep configuration explicit and seeded.
5. Add unit tests for:
   - deterministic replay with same seed
   - at least 3 realistic state sequences
   - state invariants
6. Update docs if states or config surface changed.

## Invariants

- same seed + same config => same state sequence
- no impossible states
- no impossible transition sequence for the declared scenario
- state durations are non-negative and coherent
- old behavior remains available when the state machine is disabled or not used

## Files typically touched

- `src/telematics/state_machine.py`
- `src/telematics/__init__.py`
- scenario/config modules that instantiate telematics
- `tests/telematics/test_state_machine.py`

## Validation before finishing

Run the smallest relevant checks first, then broader ones if available:
- state-machine unit tests
- deterministic-seed regression test
- one smoke scenario using workshop-related states

Follow repo-standard verification commands from `AGENTS.md` when available.

## Definition of done

A change using this skill is done when:
- the state machine is deterministic
- downstream signals can consume latent state cleanly
- tests cover realistic sequences and invariants
- documentation for any new states or transitions is added
