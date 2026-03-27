---
name: workshop-geofence-signals
description: Use when implementing or modifying GPS, workshop coordinates, distance-to-workshop, geofence membership, and workshop-entry or exit behavior for maintenance-aware telematics generation.
---

## When to use this

Use this skill when a task involves:
- generating GPS coordinates
- adding workshop location registries
- computing distance to nearest workshop
- defining workshop geofences
- making vehicle movement consistent with workshop arrival, dwell, or departure

Do not use this skill for generic schema-only changes unless workshop-related location fields are being added or modified.

## Goal

Provide location-aware telematics that support maintenance reasoning:
- workshop visits
- workshop dwell periods
- repair-window candidates
- repair completion inference when combined with motion and ignition signals

## Core raw and derived fields

Prefer these names unless the repo has an established naming scheme:
- `gps_lat`
- `gps_lon`
- `distance_to_nearest_workshop_km`
- `inside_workshop_geofence`

Optional helper outputs:
- `nearest_workshop_id`
- `workshop_visit_id`

## Design rules

1. Keep spatial behavior coherent with latent state.
2. Prefer simple, interpretable movement over route realism.
3. Make workshop geometry configurable.
4. Document the distance metric and geofence rule.
5. Use consistent units.

## Behavioral expectations by state

- `driving`: GPS changes over time
- `approaching_workshop`: distance to workshop tends to decrease
- `waiting_for_service`: location remains near workshop
- `in_service`: location remains inside or very near workshop geofence
- `post_service_test`: may remain near workshop at first, then move away
- `returned_to_operation`: distance from workshop can increase

## Geofence guidance

Default assumptions:
- workshop registry is config-driven
- distance metric should be documented and consistent
- `inside_workshop_geofence` is a boolean derived from distance and a configurable threshold

Boundary behavior must be deterministic and tested.

## Inputs this skill expects

Typical inputs:
- workshop coordinates
- geofence radius or threshold
- latent state sequence
- timestep
- seed
- optional coarse movement parameters

## Outputs this skill should support

- time-indexed GPS coordinates
- workshop-distance feature
- geofence-membership feature
- stable behavior around workshop stay and departure

## Implementation checklist

1. Add or update the workshop registry/config interface.
2. Implement GPS generation consistent with latent state.
3. Compute workshop distance and geofence membership.
4. Keep the interface easy to consume from downstream derived-feature logic.
5. Add unit tests for:
   - inside / outside geofence
   - workshop boundary behavior
   - coherent approach / dwell / departure patterns
6. Update schema docs and config docs.

## Invariants

- coordinates exist for each telematics row when GPS is enabled
- distance values are non-negative
- `inside_workshop_geofence` is consistent with the chosen threshold
- workshop-related movement is compatible with latent state
- fixed seed yields fixed output

## Files typically touched

- `src/telematics/signals/gps.py`
- `src/telematics/workshop.py`
- config modules defining workshops
- `tests/telematics/test_gps_workshop.py`

## Validation before finishing

Run:
- geofence unit tests
- deterministic replay test
- one smoke scenario with workshop arrival and dwell

Then verify that downstream code can consume the location fields without ad hoc conversions.

## Definition of done

A change using this skill is done when:
- GPS and workshop proximity fields are generated deterministically
- workshop approach, dwell, and exit behavior are coherent
- boundary and distance calculations are tested
- schema and config documentation are updated
