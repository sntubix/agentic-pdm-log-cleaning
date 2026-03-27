---
name: motion-and-trip-signals
description: Use when implementing or modifying speed, ignition, stationary logic, stop duration, trip segmentation, engine hours, or idle time for maintenance-aware telematics generation.
---

## When to use this

Use this skill when a task involves:
- `speed_kmh`
- `ignition_on`
- `stop_duration_min`
- `vehicle_is_stationary`
- `trip_id`
- trip start or end logic
- `engine_hours`
- `idle_time_min`

Do not use this skill for purely spatial workshop logic unless motion or trip behavior is being changed too.

## Goal

Generate motion and operating-state signals that make workshop reasoning and post-service inference reliable.

## Preferred fields

- `speed_kmh`
- `ignition_on`
- `vehicle_is_stationary`
- `stop_duration_min`
- `trip_id`
- optional `trip_started`
- optional `trip_ended`
- `engine_hours`
- `idle_time_min`

## Behavioral guidance

### Speed
- should be state-conditioned
- near zero when parked or in service
- positive during driving-like states
- simple, interpretable dynamics are preferred over complex traffic realism

### Ignition
- usually on when driving
- usually off during prolonged service
- may be on during post-service test or idling

### Stationary logic
`vehicle_is_stationary` should be derived from speed using a documented threshold.

### Stop duration
`stop_duration_min` should:
- accumulate while stationary
- reset when meaningful motion resumes
- remain deterministic

### Trip segmentation
Trips may be rule-based.
A new trip should start when motion resumes after a stationary period that satisfies the repo’s chosen threshold.

### Engine hours and idle time
- `engine_hours` should accumulate only while the vehicle is operational
- `idle_time_min` should accumulate when ignition is on and speed is near zero

## Inputs this skill expects

Typical inputs:
- latent state sequence
- timestep
- optional route or movement hints
- deterministic seed
- thresholds for stationary and trip segmentation

## Outputs this skill should support

- coherent speed and ignition sequences
- stop-duration tracking
- trip segmentation features
- usage accumulation features

## Implementation checklist

1. Inspect current telematics generation flow.
2. Add motion fields without breaking odometer-only compatibility.
3. Derive stationary and stop-duration features from documented thresholds.
4. Implement deterministic trip segmentation.
5. Add engine-hours and idle-time accumulation.
6. Write tests for:
   - driving vs parked vs in-service behavior
   - stop-duration accumulation and reset
   - trip boundaries
   - engine-hours monotonic accumulation

## Invariants

- speed is non-negative
- stop duration is non-negative
- engine hours are non-decreasing
- idle time is non-decreasing within a continuous idle interval
- driving-like states do not produce permanently zero speed
- in-service states do not produce sustained driving behavior
- fixed seed yields fixed output

## Files typically touched

- `src/telematics/signals/motion.py`
- `src/telematics/signals/ignition.py`
- `src/telematics/signals/trips.py`
- `tests/telematics/test_motion_signals.py`
- `tests/telematics/test_trip_segmentation.py`

## Validation before finishing

Run:
- motion and trip unit tests
- one smoke scenario with workshop dwell then resumed operation
- deterministic regression checks

Check especially that trip logic does not accidentally create a new trip at every short pause.

## Definition of done

A change using this skill is done when:
- motion, ignition, and trip logic are deterministic
- stop-duration and usage accumulators behave coherently
- tests cover key state-dependent behaviors
- threshold assumptions are documented
