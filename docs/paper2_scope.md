# Paper 2 Scope

## Working title
Agentic Data Cleaning for Fleet Predictive Maintenance Under Controlled Log and Sensor Noise

## Purpose
This document defines the target scope for the continuation paper and the corresponding implementation work to be delegated to Codex.

## Research objective
Extend the first paper from maintenance-log cleaning alone to **multi-source PdM data cleaning** in which maintenance logs and telematics streams are generated jointly, corrupted in a controlled way, and cleaned by tool-grounded agents. The continuation also studies whether **fine-tuning on automatically generated clean/noisy pairs** improves performance compared with zero-shot agents.

## Main research questions
1. Can agentic cleaning scale from single-source maintenance-log curation to multi-source fleet PdM data curation?
2. Can controlled paired log/telematics environments expose more realistic coupled-noise conditions than Paper 1?
3. Does fine-tuning on synthetic clean/noisy pairs improve cleaning quality over zero-shot prompting?
4. Which types of coupled corruption remain difficult even after tool grounding and fine-tuning?

## Core contributions in scope
1. A synthetic fleet data generator that emits **paired maintenance-log and telematics datasets**.
2. A controlled noise framework that can corrupt one or both sources, including **coupled-noise conditions**.
3. Tool-grounded agents that operate **record by record** through a minimal cleaning API and use ontology context to validate or repair entries.
4. Automatic generation of **clean/noisy aligned pairs** for supervised or instruction-style fine-tuning.
5. A benchmark comparing **zero-shot** and **fine-tuned** agents under increasingly difficult noise settings.

## Inheritance from Paper 1
Paper 2 is an extension of Paper 1, not a rewrite. The following must remain available unless an issue explicitly states otherwise:
- original maintenance-log generation
- original maintenance-log noise taxonomy
- record-level cleaning actions
- zero-shot benchmarking pipeline
- reproducibility through explicit configuration and seeds

## New capabilities in scope

### 1. Paired environment generation
The environment should generate aligned sources such as:
- fleet registry
- maintenance log
- telematics or sensor streams
- optional ontology/context tables
- clean/noisy alignment metadata

Each generated environment should be reproducible from a config and a seed.

### 2. Controlled coupled noise
The continuation paper studies conditions where noise is not isolated to a single maintenance record field. The framework should support:
- source-local noise affecting only the maintenance log
- source-local noise affecting only telematics
- coupled noise affecting both sources in a consistent way
- labels that identify the applied perturbation(s)

### 3. Tool-grounded cleaning agents
The agent should continue to act through a minimal API, but now with stronger grounding over multiple sources. The implementation should support:
- read-only tools over the relevant data sources
- ontology/context retrieval for value validation
- structured outputs for accept/reject/update-style actions
- logging of tool traces and final actions

### 4. Fine-tuning dataset export
The system should automatically export aligned supervision examples derived from generated environments, such as:
- corrupted input record or record bundle
- retrieved evidence/context
- expected cleaning action
- expected field repair(s)
- optional rationale or metadata for offline analysis

### 5. Fine-tuning and evaluation pipeline
The repository should support:
- dataset versioning for clean/noisy pairs
- train/validation/test splits
- model-specific fine-tuning adapters or training scripts
- evaluation against the zero-shot baseline
- per-noise and per-coupled-noise reporting

## Deliverables in scope
- config-driven paired environment generator
- coupled-noise operators
- tool-grounded agent interface extensions
- fine-tuning dataset builder
- training/evaluation scripts
- benchmark reports and paper-ready tables/figures

## Explicitly out of scope
Unless a GitHub issue says otherwise, the following are out of scope:
- production-grade streaming infrastructure
- deployment UI or dashboard work
- full replacement of the Paper 1 architecture
- unrelated refactors done only for style
- dependence on private datasets for core reproducibility
- broad MLOps platform work unrelated to the paper

## Definition of done for Paper 2 tasks
A Paper 2 task is done only if:
- the implementation extends the existing baseline cleanly
- tests are added or updated
- a smoke configuration runs end to end
- docs/configs are updated
- outputs are reproducible with fixed seeds
- the result is traceable to at least one Paper 2 research objective

## Recommended implementation boundaries for Codex
Codex should work through small, independent issues corresponding to one vertical slice each. Good task boundaries include:
- paired environment generator
- one coupled-noise operator family
- one ontology-aware validation module
- fine-tuning data export format
- one benchmark runner extension

## Success criteria for the paper
Paper 2 should make it possible to demonstrate that:
- multi-source agentic cleaning is feasible in a controlled PdM setting
- coupled noise is harder than isolated noise
- fine-tuning yields measurable gains over zero-shot baselines
- the repository can reproduce the paper’s experiments from configuration files
