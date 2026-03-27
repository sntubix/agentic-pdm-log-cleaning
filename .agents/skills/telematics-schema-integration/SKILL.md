---
name: telematics-schema-integration
description: Use when integrating new telematics signals or derived features into configs, output schemas, clean/noisy exports, and backward-compatible environment generation flows.
---

## When to use this

Use this skill when a task involves:
- adding new telematics columns to exported outputs
- updating config switches for signal enablement
- extending clean/noisy paired exports
- preserving backward compatibility while adding telemetry features
- documenting telematics output schemas

Do not use this skill for raw-signal modeling tasks unless the task also changes exports, schemas, or config surfaces.

## Goal

Integrate new telematics capabilities into the repo cleanly and predictably, without breaking existing experiments.

## Integration principles

1. Preserve old behavior when new features are disabled.
2. Use explicit config flags for optional signals and derived outputs.
3. Keep field names stable and documented.
4. Avoid notebook-only schema transformations.
5. Keep clean/noisy alignment explicit when applicable.

## Typical outputs to manage

- raw telematics tables
- derived-evidence tables
- aligned maintenance-log + telematics outputs
- clean/noisy paired datasets
- schema docs
- sample smoke outputs

## Inputs this skill expects

Typical inputs:
- list of new raw signals
- list of new derived features
- existing generator and export flow
- config structure
- existing downstream consumers

## Outputs this skill should support

- updated config entries
- updated export schemas
- backward-compatible output behavior
- docs describing fields and toggles

## Implementation checklist

1. Identify all export points and downstream consumers.
2. Add config gates for each new signal family when appropriate.
3. Update schema definitions and serializers.
4. Keep existing pipelines working when new features are off.
5. Add or update integration tests for:
   - old behavior with features off
   - new behavior with features on
   - schema stability and required columns
6. Update docs and sample configs.

## Invariants

- feature toggles are honored
- required identifiers and timestamps remain stable
- same config => same schema order and content unless documented otherwise
- backward compatibility holds when new signals are disabled

## Files typically touched

- generator/export modules
- config files
- schema definitions
- `tests/integration/`
- `docs/experiments.md` or schema docs

## Validation before finishing

Run:
- integration tests
- one smoke generation with features on
- one smoke generation with features off

Verify that downstream code does not need hidden notebook transformations to consume the outputs.

## Definition of done

A change using this skill is done when:
- new telematics fields are integrated into exports cleanly
- old pipelines continue to work when features are off
- config and schema docs are updated
- integration tests cover both enabled and disabled modes
