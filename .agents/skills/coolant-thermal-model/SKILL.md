---
name: coolant-thermal-model
description: Use when implementing or modifying coolant-temperature generation using a lightweight first-order thermal model with ambient temperature, state-dependent heating, cooling, and optional overheating or repair effects.
---

## When to use this

Use this skill when a task involves:
- adding or changing `coolant_temp_c`
- introducing ambient temperature effects
- modeling heating during operation
- modeling cooling after stop
- creating overheating or cooling-system repair scenarios

Do not use this skill for unrelated schema work unless coolant behavior is part of the change.

## Goal

Provide a lightweight, physically-inspired coolant-temperature signal for maintenance-aware synthetic telemetry.

The model should be simple, deterministic, and interpretable.

## Preferred signal

- `coolant_temp_c`

Optional latent variables:
- ambient temperature
- thermal inertia or cooling coefficient
- heat-generation level
- optional fault or degradation parameter

## Minimum expected behaviors

1. Cold start:
   - temperature begins near ambient

2. Active operation:
   - temperature rises gradually
   - may stabilize in an operating band

3. Vehicle off:
   - temperature decays gradually toward ambient
   - no unrealistic instantaneous drop unless explicitly modeled as noise

4. Fault / overheating scenario:
   - temperature can rise abnormally high or cool inefficiently

5. After relevant repair:
   - abnormal thermal behavior may improve

## Modeling guidance

Use a first-order thermal update or similarly simple approximation.
At each step:
- estimate heat input from operational state
- update temperature toward the state-dependent equilibrium
- include ambient as the cooling baseline
- keep transitions smooth unless the scenario explicitly injects anomalies

## Inputs this skill expects

Typical inputs:
- latent state sequence
- ignition and motion context if available
- timestep
- ambient temperature or profile
- deterministic seed
- optional cooling-system fault parameter
- optional maintenance-event schedule

## Outputs this skill should support

- time-indexed `coolant_temp_c`
- plausible warm-up and cool-down dynamics
- optional fault and post-repair behavior

## Implementation checklist

1. Add or update the coolant model module.
2. Document the chosen thermal approximation.
3. Keep output deterministic for fixed seed and config.
4. Support at least one maintenance-relevant abnormal scenario.
5. Add tests for:
   - cold start near ambient
   - warm-up under operation
   - cool-down while off
   - optional overheating scenario
   - post-repair improvement if applicable

## Invariants

- temperature remains within documented plausible bounds
- warm-up and cool-down are smooth
- same seed + same config => same trace
- off-state cooling trends toward ambient
- active operation produces higher temperature than prolonged off-state, all else equal

## Files typically touched

- `src/telematics/signals/coolant.py`
- config modules for thermal parameters
- `tests/telematics/test_coolant_model.py`

## Validation before finishing

Run:
- coolant-model unit tests
- one overheating or thermal-repair scenario
- deterministic regression checks

If a plotting helper exists, inspect one trace, but keep correctness assertions in automated tests.

## Definition of done

A change using this skill is done when:
- `coolant_temp_c` behaves like a simple thermal process
- active, resting, faulty, and repaired behaviors are distinguishable
- assumptions are documented
- tests cover warm-up, cool-down, and fault behavior
