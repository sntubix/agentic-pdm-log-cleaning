# Roadmap

## Purpose
This roadmap translates the paper continuation into a repo plan that can be executed incrementally by Codex through GitHub issues and pull requests.

## Planning principles
- Keep Paper 1 working as the stable baseline.
- Prefer extension over rewrite.
- Use one issue per implementable feature.
- Make every task reproducible, testable, and config-driven.
- Separate long-lived repo guidance from execution tasks:
  - `AGENTS.md` for durable instructions
  - `paper1_scope.md` and `paper2_scope.md` for project boundaries
  - GitHub Issues/Project for execution tracking

---

## Milestone 0 — Baseline freeze

### Goal
Create a safe handoff point before new development starts.

### Tasks
- Tag the Paper 1 repository state, e.g. `paper1-final`
- Create a `paper2-main` branch
- Add `paper1_scope.md`, `paper2_scope.md`, and `roadmap.md`
- Add or update `AGENTS.md`
- Ensure the original smoke benchmark still runs

### Exit criteria
- Paper 1 baseline is reproducible
- Codex can read repo instructions and run the baseline

---

## Milestone 1 — Paired environment generation

### Epic
Generate aligned maintenance-log and telematics environments for Paper 2.

### Why
Paper 2 explicitly moves from maintenance-log-only cleaning toward multi-source PdM data curation.

### Candidate sub-issues
1. Add a paired environment schema
2. Extend the synthetic generator to emit aligned telematics streams
3. Add environment metadata and provenance tracking
4. Add deterministic environment serialization/export
5. Add smoke configs for paired data generation

### Deliverables
- new paired environment data model
- config file(s)
- tests for alignment and reproducibility

---

## Milestone 2 — Controlled noise framework extension

### Epic
Implement source-local and coupled-noise operators.

### Why
Paper 2 centers on controlled log and sensor noise, especially coupled-noise conditions.

### Candidate sub-issues
1. Add telematics-only noise operators
2. Add coupled identifier corruption across log and telematics
3. Add coupled temporal corruption across log and telematics
4. Add multi-label perturbation metadata
5. Add benchmark fixtures covering isolated vs coupled noise

### Deliverables
- new noise operator modules
- labels/metadata schema
- deterministic tests for each operator family

---

## Milestone 3 — Tool-grounded agent layer

### Epic
Extend the agent environment so it can reason across multiple sources with ontology support.

### Why
The Paper 2 abstract emphasizes tool-grounded agents using a minimal API and ontology context.

### Candidate sub-issues
1. Define minimal multi-source read API
2. Add ontology/context lookup tool
3. Add evidence logging and trace storage
4. Extend record-level action application logic
5. Add regression tests for agent-tool interaction

### Deliverables
- tool interfaces
- trace format
- tests using small synthetic fixtures

---

## Milestone 4 — Fine-tuning dataset builder

### Epic
Export automatically generated clean/noisy supervision pairs.

### Why
Fine-tuning is a central differentiator of Paper 2 relative to Paper 1.

### Candidate sub-issues
1. Define training example schema
2. Build exporter from generated environments
3. Add dataset versioning and manifest files
4. Add split generation for train/val/test
5. Add smoke dataset artifact generation

### Deliverables
- exported JSONL/CSV/parquet dataset format
- manifests with config/seed hashes
- validation tests for dataset integrity

---

## Milestone 5 — Fine-tuning pipeline

### Epic
Train and evaluate fine-tuned agents on the synthetic supervision data.

### Candidate sub-issues
1. Add training config schema
2. Add model adapter/training entrypoint
3. Add checkpoint management
4. Add inference wrapper for fine-tuned models
5. Add evaluation hooks compatible with the benchmark runner

### Deliverables
- train script(s)
- eval-compatible inference interface
- smoke fine-tuning run

---

## Milestone 6 — Benchmarking and ablations

### Epic
Compare zero-shot and fine-tuned agents under multiple noise settings.

### Candidate sub-issues
1. Extend benchmark runner for Paper 2 tasks
2. Add isolated-vs-coupled-noise ablation
3. Add zero-shot vs fine-tuned comparison
4. Add per-noise reporting tables
5. Add paper-ready result export

### Deliverables
- benchmark configs
- aggregated metrics files
- tables/plots for the paper

---

## Milestone 7 — Hardening and paper support

### Epic
Make the repo easy for Codex and humans to use repeatedly.

### Candidate sub-issues
1. Improve `AGENTS.md`
2. Add make targets for setup, test, smoke, benchmark
3. Add docs for experiment reproduction
4. Add changelog/release notes for Paper 2 milestones
5. Add PR templates for Codex tasks

### Deliverables
- stable commands
- updated docs
- cleaner developer workflow

---

## Suggested GitHub Project structure

### Parent issues (epics)
- Epic: Paired environment generation
- Epic: Controlled noise framework extension
- Epic: Tool-grounded agent layer
- Epic: Fine-tuning dataset builder
- Epic: Fine-tuning pipeline
- Epic: Benchmarking and ablations
- Epic: Hardening and paper support

### Recommended labels
- `epic`
- `paper1-baseline`
- `paper2`
- `generator`
- `noise`
- `agent`
- `ontology`
- `finetune`
- `benchmark`
- `docs`
- `good-first-codex-task`

---

## Suggested issue template for Codex-ready tasks

```md
## Problem
What limitation or missing capability does this issue address?

## Goal
What should exist after this issue is completed?

## In scope
- item
- item

## Out of scope
- item
- item

## Acceptance criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Likely files
- path/a
- path/b

## Commands to run
make test
make smoke
```

---

## Recommended order of execution
1. Freeze Paper 1 baseline
2. Build paired environment generator
3. Add first coupled-noise operators
4. Extend tool-grounded agent layer
5. Export fine-tuning datasets
6. Add fine-tuning pipeline
7. Run ablations and finalize reporting

This order gives you a usable benchmark early while keeping the fine-tuning work attached to a stable data-generation and evaluation stack.
