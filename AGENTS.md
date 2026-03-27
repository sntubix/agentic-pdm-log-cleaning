# AGENTS.md

## Repository purpose

This repository implements the continuation of the maintenance-log cleaning research for fleet predictive maintenance.

The main objectives are:
- generate paired maintenance-log and telematics environments,
- inject controlled noise into logs and telematics,
- support tool-grounded repair and validation,
- export aligned clean/noisy supervision data,
- benchmark zero-shot and fine-tuned agents.

Unless an issue explicitly says otherwise, extend the existing codebase incrementally. Do not rewrite working modules without a concrete need.

## Working style

- Prefer small, composable modules over large monolithic files.
- Keep implementations deterministic given explicit seeds.
- Favor interpretable logic over opaque heuristics.
- Preserve backward compatibility whenever new features are disabled.
- Keep raw signals and derived features separate.
- Do not move core logic into notebooks.

## Current telematics direction

The current telematics roadmap extends odometer-only telemetry into maintenance-aware multi-signal telemetry.

Prioritize these signals and capabilities:
1. workshop-aware location features,
2. motion and operating-state features,
3. derived maintenance evidence,
4. physically-inspired maintenance signals,
5. schema integration and benchmark scenarios.

When implementing telematics features, prefer state-driven generation over independent per-signal random generation.

## Skills to use

Use the repository skills in `.agents/skills` when relevant.

### `telematics-state-machine`
Use for:
- latent operational state processes,
- state transitions,
- state-conditioned generation,
- deterministic scenario sequencing.

### `workshop-geofence-signals`
Use for:
- GPS generation,
- workshop coordinates,
- distance-to-workshop features,
- geofence entry and exit behavior.

### `motion-and-trip-signals`
Use for:
- speed,
- ignition,
- stop duration,
- trip segmentation,
- engine hours,
- idle time.

### `battery-physical-model`
Use for:
- battery voltage generation,
- SOC-like latent variables,
- charging/discharging behavior,
- battery replacement or degradation scenarios.

### `coolant-thermal-model`
Use for:
- coolant temperature generation,
- ambient-temperature handling,
- heating and cooling dynamics,
- overheating-related scenarios.

### `maintenance-evidence-derivation`
Use for:
- repair-window inference,
- repair-completion inference,
- workshop dwell aggregation,
- post-service movement evidence.

### `telematics-schema-integration`
Use for:
- output schema changes,
- config gating,
- aligned export changes,
- backward-compatibility handling,
- documentation updates.

### `telematics-benchmark-scenarios`
Use for:
- smoke configurations,
- end-to-end synthetic scenarios,
- benchmark data generation,
- telematics-aware evaluation cases.

### `telematics-test-oracle`
Use for:
- deterministic tests,
- invariants,
- plausible-range assertions,
- cross-signal consistency checks,
- schema and integration validation.

## Planning rules

For tasks that affect multiple modules or require more than a small localized patch:
- write or update a short plan before editing code,
- keep the plan concrete and file-oriented,
- update the plan if the implementation approach changes.

Use `PLANS.md` only when the task is large, cross-cutting, or expected to require multiple iterations.

A plan should include:
- goal,
- files likely to change,
- implementation steps,
- validation commands,
- open assumptions.

## Engineering constraints

- No hidden randomness. Every stochastic component must accept a seed or deterministic RNG.
- No silent schema changes. Document added or renamed fields.
- No undocumented config flags.
- No new heavy dependencies unless clearly justified in the PR or issue.
- Keep physical models lightweight, documented, and easy to inspect.
- Prefer rule-based or first-order models over complex simulators.

## Testing and validation

Before finishing a task:
- run the most targeted tests for changed modules,
- run broader integration or smoke tests when schemas or pipelines change,
- verify deterministic behavior when seeds are fixed,
- verify backward compatibility when new features are disabled.

At minimum:
- add or update unit tests for every new generator or derived-feature module,
- add integration coverage when outputs or configs change,
- include at least one scenario-level validation for telematics features.

## Review expectations

When opening or updating a PR:
- summarize what changed,
- list files touched,
- list tests run,
- list assumptions and limitations,
- call out any backward-compatibility risk,
- note any config or schema changes explicitly.

## Commands

Use repository commands if present. Prefer stable wrapper commands over ad hoc commands.

Typical command categories:
- setup/install
- lint/format
- unit tests
- integration tests
- smoke benchmarks

If the repository already defines command wrappers such as `make`, `just`, `uv`, or task runners, use those instead of inventing new entry points.

## File placement guidance

Prefer these patterns unless the repository already uses a different convention:
- signal generators under `src/telematics/signals/`
- state logic under `src/telematics/`
- derived features under `src/telematics/derived_features.py` or similar
- tests under `tests/telematics/`
- benchmark configs under `configs/benchmarks/`
- schema/config docs under `docs/`

## What to avoid

- Do not implement unrelated refactors during feature work.
- Do not replace simple deterministic logic with unnecessarily complex models.
- Do not mix raw telematics generation with downstream benchmark labeling in the same module.
- Do not break old outputs when a feature can be gated by config.
- Do not skip tests for generated signals just because they are synthetic.

## Definition of done

A task is done only when:
- implementation is complete,
- tests are added or updated,
- relevant validation commands pass,
- docs/config/schema updates are included when needed,
- the result is reproducible with fixed seeds,
- and the change stays within the issue scope.
