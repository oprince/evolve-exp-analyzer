# Evolve Loop Analysis — Requirements

## 1. Purpose

A tool to analyze evolve loop experiment traces and answer: **"What works well and what doesn't for this experiment?"**

The right mental model is **DL training diagnostics**: you never just look at final accuracy — you look at the loss curve shape, train/val gap, gradient norms, learning rate schedule effects, per-class performance, compute utilization. Each signals something different about whether the *process* is healthy, independent of the final number. The same diagnostic framework should exist for evolutionary optimization runs.

The tool takes the raw outputs of any evolutionary experiment (SkyDiscover, ShinkaEvolve, OpenEvolve, or any future tool), analyzes the optimization *trajectory*, and produces a structured diagnostic report with rated dimensions, concrete evidence from the logs, and actionable recommendations. The goal is to surface signals that a human would catch if they went through the logs line by line — automatically.

---

## 2. Operation Modes

The tool operates in two modes that share the same analysis logic but differ in when and how analysis is triggered.

### 2.1 Post-mortem Analysis

Analysis runs **after the evolve loop has completed**. The full record set is available at analysis time.

**Characteristics:**
- Batch ingestion: all records loaded at once from JSONL file or framework checkpoint
- Full retrospective analysis: stagnation periods have known start, end, and recovery; convergence curve is complete; ceiling analysis sees the full tail
- No latency pressure: analysis can take minutes
- Single pass: all 10 quantitative analyzers, all 6 LLM judge steps, full report synthesis
- Invoked manually by the researcher after a run

**Primary output:** Structured diagnostic report + static dashboard

### 2.2 In-flight Analysis

Analysis runs **while the evolve loop is executing**. Records arrive incrementally; the tool operates as a non-blocking sidecar.

**Characteristics:**
- Incremental ingestion: tail-reads new records from a live JSONL file or polls the framework's database
- Partial analysis: stagnation periods may be ongoing (`end_iteration` unknown); convergence curves are rolling estimates; ceiling analysis predicts forward from partial history
- Real-time alerting: stagnation alerts fire as soon as the threshold is crossed, not at run end
- Scheduled LLM judges: root-cause analysis triggers on threshold crossing; high-frequency judges (compliance, mutation quality) run on sampled or batched records to avoid cost runaway
- Non-blocking: the analyzer never delays the evolve loop iteration — all LLM calls dispatch asynchronously
- Automatic transition to full post-mortem report on run completion signal

**Primary output:** Live dashboard + real-time alert log + full post-mortem report on completion

### 2.3 Shared Surface

The analysis logic is **identical in both modes**. All 10 quantitative analyzers are pure functions on `List[dict]` — calling them on a partial record list produces valid partial results with less certainty. The two modes differ only in how records are ingested and when LLM judges are triggered:

| Component | Post-mortem | In-flight |
|---|---|---|
| All 10 quantitative analyzers | ✅ Full batch | ✅ Incremental (same code) |
| All 6 LLM judge prompts | ✅ Single pass at end | ✅ Triggered on threshold / periodic |
| Historical DB + comparative analysis | ✅ Full | ✅ Available throughout |
| Report synthesizer | ✅ Final report | ✅ Partial reports + final on completion |
| Dashboard | ✅ Static | ✅ Live refresh |
| Batch loader | ✅ | — |
| Tail reader / DB poller | — | ✅ |
| Alert dispatcher + deduplication state | — | ✅ |
| LLM judge scheduler | — | ✅ |

### 2.4 LLM Endpoint Isolation

When running in-flight alongside an evolve framework, the analyzer **must use a dedicated LLM configuration** — separate from the framework's own LLM endpoint — so that analyzer judge calls do not compete with the framework's mutation calls for rate limits, quota, or latency headroom.

**Required isolation mechanisms (in order of preference):**

1. **Separate API key** — independent RPM/TPM quota pool from the same provider
2. **Different model tier** — analyzer uses a cheaper/faster model for high-frequency judge steps; the evolve framework is not affected
3. **Different provider** — framework uses OpenAI; analyzer uses Anthropic (or vice versa); rate limits are on entirely separate systems
4. **Dedicated base URL** — analyzer points to a local inference server (Ollama, vLLM), a dedicated Azure OpenAI deployment, or a custom proxy; framework's cloud API quota is completely unaffected

**Per-step model selection** is a first-class requirement. High-frequency judge steps (semantic compliance, per-mutation quality) must support a cheaper model override independent of the model used for low-frequency, high-value steps (stagnation root cause, artifact clustering). Running a frontier model on every iteration in a 500-iteration run is cost-prohibitive and unnecessary.

**Budget cap:** the analyzer must enforce a configurable maximum LLM cost per run. If the cap is reached, LLM judge steps stop gracefully; deterministic quantitative analysis continues unaffected.

---

## 3. Analytical Dimensions

Each dimension is like a diagnostic metric in DL training. Each produces a rating, a one-line summary, key evidence with iteration pointers, a historical comparison (once calibration data exists), and concrete recommendations.

### 2.1 Evolutionary Effectiveness

- Track `score_delta` per iteration (child_score − parent_score)
- Identify breakthrough iterations (score jump above a configurable threshold)
- Correlate effectiveness with mutation type (diff / full rewrite / cross), LLM model used, island / population, and parent selection strategy

### 2.2 Convergence Dynamics

The equivalent of the loss curve. Answers: Is the optimization converging? How fast? Is it monotonic or oscillating? Where are the phase transitions?

**Quantitative signals:**
- Best-so-far curve over iterations
- Rolling mean and variance of scores (window configurable)
- Change-point detection — when did the trajectory fundamentally shift?
- Convergence rate: improvement per iteration, improvement per evaluation
- Time-to-best: what fraction of total compute was spent *after* the best solution was already found

*Example: flags that the best result came at iteration 50 of 51, but the trajectory was essentially flat from iteration 11 onward — 40 iterations of oscillation for negligible gain.*

### 2.3 Exploration vs. Exploitation Balance

The equivalent of learning rate schedule health. Answers: Is the system trying enough structurally different approaches? Is it spending too long refining a local optimum? Is it exploring too wildly without consolidating gains?

**Signals:**
- Structural diversity index: how different are successive candidates — not just parameter values but code structure and topology
- Time spent in "exploit" phases (small parameter tweaks around a known-good structure) vs. "explore" phases (fundamentally new approaches)
- Number of distinct strategy clusters tried
- Revert frequency: how often the system goes back to a previous known-good solution

*Example: shows the system explored ~5-6 structural families but spent most iterations doing parameter tweaks within a single one.*

### 2.4 Regression Analysis

The equivalent of training instability. Answers: How often do new iterations make things worse? How severe are the regressions? How quickly does the system recover?

**Signals:**
- Regression frequency: fraction of iterations that are worse than the previous best
- Regression severity distribution
- Mean recovery time: iterations to get back to pre-regression performance
- Death spiral detection: consecutive regressions (overlaps with stagnation, but distinct — regressions are active getting-worse, stagnation is flat)

### 2.5 Stagnation Detection

Detect sequences of consecutive iterations with no progress toward the goal.

**Alert threshold:** configurable, default = 10 consecutive non-improving iterations. Severity escalates at 2× threshold.

**Stagnation subtypes:**

| Type | Signal |
|---|---|
| Score plateau | `child_score ≤ parent_score` repeatedly |
| Identical output | Generated code is identical to parent code |
| Format failure loop | Same format error repeating across iterations |
| Compliance violation loop | Same constraint violated repeatedly |
| Micro-improvement trap | Improvement < ε per iteration (false progress) |

**Per stagnation period, the tool must capture:**
- Start and end iteration, length of streak
- Full sequence of what was attempted (iteration, mutation type, failure type, score delta)
- Dominant failure type within the streak
- Recovery event: which iteration broke the stagnation, what mutation type, model, paradigm shift
- Best score before stagnation started and after recovery

**Alert severity levels:**

| Condition | Severity |
|---|---|
| Streak ≥ threshold | Warning |
| Streak ≥ 2× threshold | Critical |
| Dominant failure = `identical_output` + streak ≥ threshold | Critical |
| Dominant failure = `compliance_violation` + streak ≥ threshold | High |
| Multiple stagnation periods in same run | Warning |

Each alert includes: sequence visualization, dominant failure type, LLM root cause analysis, and a concrete recommendation.

### 2.6 Efficiency

The equivalent of compute utilization. Answers: What's the cost per unit of improvement? Where is compute being wasted? What's the marginal cost curve?

**Signals:**
- Improvement per LLM call
- Improvement per dollar of LLM API cost
- Improvement per wall-clock hour
- Improvement per evaluator call / trial
- Fraction of total compute spent in the "productive" phase (before the trajectory plateaued) vs. the "wasted" phase after best-so-far stopped improving
- Pareto frontier of cost vs. performance across iterations

*Example: most improvement came in iterations 0–3 (~8 LLM calls). Iterations 4–50 added marginal improvement at a cost of 120 more LLM calls — a 20× worse cost-efficiency ratio in the second phase.*

### 2.7 Generalization / Robustness

The equivalent of train/val gap. Answers: Does the solution overfit to the evaluation setup? How sensitive is it to perturbations?

**Signals:**
- Performance gap between eval-time score and full-scale verification (if available)
- Per-workload / per-benchmark decomposition: is the improvement concentrated in one workload?
- Parameter sensitivity: how much does score change with small perturbations to the solution?
- Cross-seed variance (if multiple seeds available)

*Example: a gap between development evaluation and full verification is a red flag. Improvement concentrated in one sub-benchmark while others barely move indicates a narrow solution.*

### 2.8 Instruction Compliance

Did the LLM code-writer respect the instructions it was given?

**Deterministic checks (zero LLM cost):**

| Check | What it detects |
|---|---|
| EVOLVE-BLOCK boundary | Did the mutation touch code outside the marked block? |
| Format validity | Is the diff parseable / is the full rewrite a complete file? |
| API preservation | Were function signatures maintained? |

**LLM-as-judge (semantic compliance):**
- Given the system message constraints and the generated code, were all constraints respected?
- Identify specific violations: which constraint was broken and where in the code
- Rate overall compliance: `fully_compliant` / `partially_compliant` / `non_compliant`

### 2.9 Evaluator Output Analysis

The evaluator is a first-class input — not just a source of scores.

**Failure mode classification (deterministic):**

| Mode | Signal |
|---|---|
| `crash` | Evaluator raised an exception |
| `timeout` | Exceeded the time limit |
| `wrong_output` | Correctness metric = 0 |
| `worse` | Ran correctly but combined_score dropped |
| `partial` | Some sub-metrics improved, combined_score did not |
| `success` | combined_score improved |

**Sub-metric divergence:** when `combined_score` does not improve, detect whether individual sub-metrics did — surfaces hidden partial progress the primary score obscures.

**Cascade stage analysis:** identify at which stage (1 / 2 / 3) most failures occur. High Stage-1 failure rate indicates syntactically broken LLM output — a system message problem, not an evolution strategy problem.

**Evaluator artifact analysis (LLM-as-judge):** cluster recurring `stderr`, `build_warnings`, and `llm_feedback` across failed iterations. Identify recurring error patterns and produce actionable system message recommendations.

**Evaluator determinism check:** when `num_runs > 1`, flag iterations where score variance across runs is high. Stagnation caused by evaluator noise is a different problem than LLM quality.

### 2.10 Search Space Utilization

The equivalent of model capacity analysis. Answers: Is the search space well-sized? Are parameters hitting their bounds? Is the sampler actually exploring?

**Signals:**
- Parameter value distributions across top-K solutions
- Parameters repeatedly hitting bound constraints (signal that space is too narrow)
- Effective dimensionality: are some parameters redundant across all good solutions?
- Trial-to-parameter ratio (budget relative to complexity)
- For BO-based methods: acquisition function behavior, collapse detection

### 2.11 Meta-Analysis Quality

Unique to LLM-in-the-loop systems. Answers: Is the meta-analyzer's reasoning actually helping? Are its suggestions followed? Do followed suggestions lead to improvement?

**Signals:**
- Suggestion compliance rate: what fraction of LLM meta-recommendations were followed in the next iteration?
- Conditional improvement rate: improvement rate when suggestions are followed vs. when ignored
- Pattern reuse frequency: how often does the system revisit the same approach despite previous failure?
- Scratchpad / reasoning trace growth rate and compaction events (for tools that maintain one)
- Self-contradiction frequency: does the meta-analyzer recommend things it previously advised against?

*Example: suggestions were followed 60% of the time, and followed suggestions led to improvement only 30% of the time — this quantifies the LLM guidance value (or lack thereof).*

### 2.12 Ceiling and Stopping Criteria

The equivalent of capacity / underfitting analysis. Answers: Has the optimization hit a ceiling? Should it have stopped earlier? Is more compute likely to help?

**Signals:**
- Improvement rate trend: is marginal improvement declining monotonically?
- Statistical test for plateau: are recent iterations drawn from the same distribution as the flat region?
- Estimated probability of beating current best in N more iterations (extrapolation)
- Comparison to known theoretical bounds if available
- Detection of "accept terminal state" signals from the meta-analyzer that were ignored

*Example: the meta-analyzer suggested stopping around iteration 32 but the experiment ran to 51 — an automated system should flag that 19 iterations of compute were wasted.*

---

## 4. Alert System

Alerts must be actionable. Each alert contains:

1. Experiment identifier (run name, algorithm, model)
2. Stagnation or anomaly period (start, end, length)
3. Sequence of what was attempted (compact table)
4. Dominant failure type and secondary types
5. LLM-generated root cause explanation
6. Concrete recommendation (system message change, strategy change, evaluator change, early stopping)

---

## 5. Output and Reporting

### Structured Diagnostic Report

Each dimension produces a standardized block:

```
Dimension: Convergence Dynamics
Rating:    2/5  🔴
Summary:   Trajectory was flat from iteration 11 onward — 40 of 51 iterations
           produced no meaningful improvement over iteration 11.
Evidence:  Best-so-far at iter 11: 0.712. Best-so-far at iter 51: 0.731.
           Rolling variance (window=10) exceeded 0.05 from iter 15 onward.
Historical: Worse than 80% of past runs on this benchmark (median plateau onset: iter 28).
Recommendation: Add rolling improvement rate stopping criterion; would have saved
           ~35 iterations and ~$40 in this experiment.
```

### Cross-Dimension Interactions

A narrative section explaining how dimensions affect each other:
- *"Poor exploration-exploitation balance (dim 2.3) explains both the high regression rate (dim 2.4) and the low efficiency score (dim 2.6)"*
- *"Instruction compliance failures (dim 2.8) account for 40% of the stagnation period (dim 2.5)"*

### Per Stagnation Period
- Sequence table (iteration, mutation type, failure type, score delta)
- Root cause category: `LOCAL_OPTIMUM` / `INSTRUCTION_CONFUSION` / `APPROACH_EXHAUSTION` / `FORMAT_ISSUE` / `EVALUATOR_NOISE` / `OTHER`
- LLM explanation and recommendation

### Per Run (Aggregate)
- Total stagnation periods and % of iteration budget wasted
- Failure budget breakdown (% of iterations per failure mode)
- Cross-tabulation: compliance level × score improvement rate
- Evaluator artifact cluster summary
- Efficiency curve (cost vs. improvement over time)

### Executive Summary
A top-level paragraph covering: overall assessment, the two or three most impactful findings, the single most actionable recommendation, and estimated compute wasted.

### Dashboard Panels
- Score progression curve with stagnation periods highlighted, breakthroughs annotated, change-points marked
- Alert panel sorted by severity
- Mutation effectiveness by type, model, island
- Compliance × score cross-tabulation heatmap
- Failure budget stacked bar
- Cascade stage failure distribution
- Evaluator artifact clusters with recommendations
- Efficiency: improvement per LLM call / dollar over time
- Exploration vs. exploitation timeline

---

## 6. Comparative and Historical Analysis

The system must support cross-experiment learning:

- **Historical database** of past experiment analyses, queryable by benchmark, tool, model, and algorithm
- Each new report is compared against the historical distribution: *"Is this convergence rate typical? Is this regression frequency abnormal for this tool?"*
- Ratings are relative to historical norms once enough experiments are accumulated (early runs use absolute heuristics)
- Recurring patterns that appear across multiple experiments are promoted into standing recommendations

---

## 7. Design Constraints

### Tool-Awareness
A population-based method (OpenEvolve) has very different convergence dynamics than a serial refinement method. The analysis must be tool-aware — high regression rate may be normal and healthy in an evolutionary population but pathological in a serial refinement loop. Baselines and thresholds are parameterized per tool type.

### Graceful Degradation
Log quality varies across experiments and tools. The system must degrade gracefully:
- When only scores are available → deliver quantitative analysis (convergence, regression, efficiency)
- When code diffs are available → add structural diversity and compliance analysis
- When reasoning traces / scratchpads are available → add meta-analysis quality dimension
- When historical data is available → add comparative ratings

### Dimension Discovery
The 12 dimensions above are the initial fixed set. The system must include a **"Novel Observations"** section where the LLM can flag patterns that don't fit existing categories. Recurring novel observations may be promoted into formal dimensions over time.

---

## 8. Input Format

### Primary Format: JSONL

One record per iteration. Key fields:

| Field | Type | Required | Source |
|---|---|---|---|
| `iteration` | int | Yes | Framework |
| `parent_id` | uuid | Yes | Framework |
| `child_id` | uuid | Yes | Framework |
| `mutation_type` | string | Yes | Framework |
| `llm_model` | string | Yes | Framework |
| `llm_tokens_used` | int | If available | Framework |
| `llm_cost_usd` | float | If available | Framework |
| `wall_clock_seconds` | float | If available | Framework |
| `num_evaluations` | int | If available | Framework |
| `system_message` | string | Yes | Framework |
| `prompt` | string | Yes | Framework |
| `llm_response` | string | Yes | Framework |
| `reasoning_trace` | string | If available | Framework |
| `parent_code` | string | Yes | Framework |
| `child_code` | string | Yes | Framework |
| `diff` | string | If diff-based | Framework |
| `parent_score` | float | Yes | Framework |
| `child_score` | float | Yes | Evaluator |
| `score_delta` | float | Derived | Computed |
| `evaluation_status` | enum | Derived | Computed |
| `cascade_stage_failed` | int / null | If cascade enabled | Evaluator |
| `evaluator_metrics` | dict | Yes | Evaluator |
| `evaluator_artifacts` | dict | If available | Evaluator |
| `evolved_block_only` | bool | Derived | Compliance check |
| `format_valid` | bool | Derived | Compliance check |
| `signature_preserved` | bool | Derived | Compliance check |
| `stagnation_streak_id` | string | Derived | Stagnation detector |
| `streak_position` | int | Derived | Stagnation detector |
| `island_id` | int | If multi-island | Framework |
| `framework_stagnation_event` | string | If available | Framework adapter |

### Experiment Metadata Record

A separate header record (or sidecar JSON) per experiment:

| Field | Notes |
|---|---|
| `experiment_id` | Unique run identifier |
| `tool` | `skydiscover` / `shinkaevolve` / `openevolve` / other |
| `tool_type` | `population_based` / `serial_refinement` |
| `algorithm` | `adaevolve` / `evox` / `topk` / etc. |
| `baseline_score` | Score of the initial program |
| `objective_description` | What is being optimized |
| `stopping_criteria` | Max iterations, early stop config |
| `hardware` | Compute configuration |
| `total_iterations` | Actual iterations run |
| `total_llm_cost_usd` | If available |

### Supported Source Frameworks

| Framework | Adapter reads | Stagnation hints available |
|---|---|---|
| SkyDiscover | `checkpoints/checkpoint_N/` + run logs + AdaEvolve paradigm tracker | Yes — paradigm shift events |
| ShinkaEvolve | SQLite database (`dbase.py` schema) | Yes — dynamic island spawn events |
| OpenEvolve | Checkpoint directories + `evolution_trace.jsonl` (requires `evolution_trace.enabled: true`) | Partial — early stopping trigger |
| Generic | Direct JSONL | N/A |

---

## 9. Build Strategy

The tool is implemented as a **standalone tool** (`evolve-analyzer`), independent of CLEAR. Three LLM infrastructure modules are vendored from CLEAR at project creation and maintained independently. All other components are built from scratch. See Design document §1 for the full architectural decision and rationale.

The architecture adopts a **coordinator + specialist agents** design:
- Ingestion Agent (adapters + validation)
- Quantitative Analyzer (all deterministic metrics — no LLM cost)
- Qualitative Analyzer (LLM agent for code diffs, compliance, scratchpads)
- Comparative Analyzer (historical DB)
- Report Synthesizer (ratings + evidence + recommendations)

### Development Phases

**Phase 1 — Post-mortem Analysis**
Full batch analysis pipeline. All 10 quantitative analyzers, all 6 LLM judge steps, historical DB, report synthesis, static dashboard. Validates the entire analysis logic before any real-time concerns are introduced.

**Phase 2 — In-flight Analysis**
Thin incremental layer on top of Phase 1. Adds tail reader / DB poller, alert dispatcher with deduplication state, LLM judge scheduler, and live dashboard refresh. The analysis logic from Phase 1 is unchanged — Phase 2 controls *when* it is called, not *what* it does.
