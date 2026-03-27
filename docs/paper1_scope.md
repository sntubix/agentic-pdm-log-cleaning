# Paper 1 Scope

## Title
Cleaning Maintenance Logs with LLM Agents for Improved Predictive Maintenance

## Purpose
This document captures the implementation and research scope of the first paper so the repository has a stable baseline before starting the continuation work.

## Research objective
Evaluate whether LLM-based agents can support predictive maintenance (PdM) data cleaning pipelines, with a primary focus on noisy maintenance logs. The paper studies whether agents can detect, reject, or repair corrupted log records by reasoning over structured enterprise-style data sources.

## Problem setting
The first paper studies maintenance logs as a critical data source for downstream PdM models. The logs are affected by common errors such as typos, missing values, identifier mismatches, and incorrect dates. The study evaluates LLM agents on six distinct noise types and assesses whether they can clean records in a zero-shot, tool-augmented setting.

## Core contributions in scope
1. Define a taxonomy of common noise patterns for automotive PdM maintenance logs.
2. Provide an open-source synthetic data generation framework for fleet-related maintenance data with controlled noise injection.
3. Benchmark multiple LLMs on maintenance-log cleaning tasks.
4. Analyze strengths, limitations, and adaptation needs for industrial PdM settings.

## Data sources in scope
The first paper uses a synthetic fleet environment composed of four data sources:
- Fleet registry
- Sensor data
- Service operations catalog
- Maintenance log

In the first paper, the maintenance log is the primary noisy target. The fleet registry and sensor data are generated as clean reference sources that the agent can query to validate or repair maintenance records.

## Noise taxonomy in scope
The first paper covers the following noise types:
- M0: no noise
- M1: vehicle identifier misalignment
- M2: out-of-fleet vehicles
- M3: invalid categorical values
- M4: missing values
- M5: digital system test entries
- M6: wrong end dates

## Agent task in scope
The agent processes one maintenance record at a time and must choose exactly one of the following actions:
- `accept`
- `reject`
- `update`

The implementation assumes a record-level cleaning workflow with optional tool use over the available data sources.

## Benchmark setting in scope
- Synthetic data generation with configurable per-noise proportions
- Zero-shot prompting
- Record-by-record processing
- Evaluation across multiple generated environments
- Metrics centered on action correctness and repair correctness

## Repository responsibilities in scope
The repository for Paper 1 should support:
- synthetic fleet data generation
- controlled maintenance-log noise injection
- relational or table-like access to enterprise data sources
- agent execution against record-level cleaning tasks
- experiment logging and result aggregation
- reproducible benchmarking with explicit seeds/configuration

## Explicitly out of scope
The following items are **not** part of the Paper 1 baseline:
- joint corruption of maintenance logs and telematics streams
- fine-tuning on clean/noisy aligned pairs
- multi-record memory or cross-record persistent reasoning
- production deployment concerns such as streaming infrastructure
- evaluation on authentic partner logs inside the public repo
- broad refactors unrelated to the synthetic-data benchmark

## Baseline constraints to preserve
When extending the repository for Paper 2, the following Paper 1 behavior should remain available:
- generation of clean fleet reference data
- generation of noisy maintenance logs under the original taxonomy
- record-level cleaning via `accept`, `reject`, and `update`
- zero-shot benchmarking pipeline
- reproducible experiment execution

## Why this scope matters
Paper 1 is the reference implementation. It establishes the baseline generator, the original maintenance-log noise taxonomy, and the zero-shot agent benchmark. Paper 2 should extend this baseline rather than replace it.
