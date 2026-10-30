---
date: '2026-10-07T00:00:00-04:00'
draft: false
title: 'Does Reasoning Help? A Paired MMLU Study on a Local 27B Model'
description: "I turned reasoning on for one local 27B model and re-ran a paired 1,000-question MMLU sample against its Part 1 baseline: +8.7 points, at roughly 14x the wall-clock time."
author: 'Jorgen Bergstrom'
tags: ['ML', 'LLM', 'llama.cpp', 'Benchmarking', 'Reasoning']
---

In [Part 1](/posts/mmlu_local_llms/) I benchmarked eight
local GGUF models on the full MMLU test set with reasoning switched **off**. That is,
the local LLMs were force to respond with a single letter.
In this follow-up article I am repeating the study using one of those models using
reasoning **on**.

---

## The question, and the trap

Part 1 ended on a caveat: those scores measured the ability of a model to answer a question directly,
with no room to deliberate, and a thinking-capable model would probably do
better if you let it think. So: how much better are the results when the model can reason?

The naive way to answer that is to turn reasoning on, run a few hundred
questions, and compare the accuracy to the Part 1 table. **Don't do that.** MMLU
questions vary enormously in difficulty, so unless you run the *same* questions
under both conditions, any difference you see is confounded by which questions
happened to be sampled. A reasoning run on the easy half of MMLU would look
great for reasons that have nothing to do with reasoning.

The fix is a **paired design**: answer each question twice, once each way. Then
you can count the questions that flipped, and test whether the flips are
lopsided with a **McNemar test**. That stays statistically meaningful even on a
subset, which matters because a full reasoning sweep on this hardware would
take about a week.

## The design

- **Model:** `qwen3.8-27B-IQ3_S` (~12 GB, a hybrid SSM + attention architecture,
  run fully on one RTX 5060 Ti with 16 GB).
- **Non-reasoning arm (control):** the *published* Part 1 predictions for this
  model — raw `/completion`, a grammar forcing one of `A|B|C|D`, one token —
  loaded and matched to each question by its `subject:idx` id. Reusing the
  original run means the control is literally the number from Part 1, and it
  halves the compute.
- **Reasoning arm:** `/v1/chat/completions` with
  `chat_template_kwargs = {"enable_thinking": true, "reasoning_effort": "xhigh"}`,
  server flags `--reasoning-format deepseek` (thoughts in `reasoning_content`,
  answer in `content`), `--reasoning-budget 8192`, context 131072, one slot.
- **Sample:** 1,000 questions, sampled round-robin across all 57 subjects so
  every subject is represented (~17–18 questions each), seeded for
  reproducibility.
- **Scoring:** greedy decoding, then parse the final `Answer: X` line. No
  grammar on this arm (see the pitfall below).
- **Test:** exact two-sided **McNemar** on the discordant pairs.

One honest framing up front: this compares two *methods*, not two settings of a
single switch. The control is a raw one-token completion; the reasoning arm uses
the chat template and an instruction to reason first. Some of the gap is
therefore method as well as mechanism. That is the honest way to state any
"vs. my original table" claim.

## The results

| Arm | Overall | Macro | Avg tokens | Avg latency |
|---|---:|---:|---:|---:|
| non-reasoning (Part 1 baseline) | **82.5%** | 82.6% | 1 | 2.9 s |
| reasoning (`xhigh`) | **91.2%** | 91.1% | 1,188 | 42.2 s |

Reasoning adds **+8.7 points** (95% CI **+6.3 to +11.1**). Exact McNemar
**p ≈ 3.7 × 10⁻¹³** (χ² = 49.6) across 1,000 paired questions.

|  | reasoning correct | reasoning wrong |
|---|---:|---:|
| **baseline correct** | 794 | 31 |
| **baseline wrong** | 118 | 57 |

So it isn't simply "more thinking, more right." Reasoning **fixed 118 questions
and broke 31** — a net **+87 correct answers**. Those 31 regressions are the
interesting part, and they're the reason to run a paired test rather than trust
a single accuracy number.

## Where reasoning earns its keep — and where it doesn't

Gains are concentrated exactly where you'd expect: **mathematics, formal logic,
and physics**. Each subject has only ~17–18 questions, so treat these as
illustrative, not precise.

| Subject | baseline | reasoning | delta |
|---|---:|---:|---:|
| high_school_mathematics | 8/18 | 18/18 | **+10** |
| college_mathematics | 11/18 | 17/18 | +6 |
| professional_accounting | 12/17 | 17/17 | +5 |
| machine_learning | 11/17 | 16/17 | +5 |
| elementary_mathematics | 13/18 | 18/18 | +5 |
| formal_logic | 12/18 | 17/18 | +5 |
| global_facts | 9/18 | 14/18 | +5 |
| abstract_algebra | 13/18 | 17/18 | +4 |
| college_computer_science | 14/18 | 18/18 | +4 |
| college_physics | 14/18 | 18/18 | +4 |
| high_school_physics | 14/18 | 18/18 | +4 |

The handful of regressions were all one or two questions, and clustered in
softer areas — the kind of question where letting a model talk itself into a
different answer can hurt:

| Subject | baseline | reasoning | delta |
|---|---:|---:|---:|
| marketing | 16/17 | 14/17 | −2 |
| human_aging | 15/17 | 14/17 | −1 |
| prehistory | 17/17 | 16/17 | −1 |
| management | 16/17 | 15/17 | −1 |
| us_foreign_policy | 16/17 | 15/17 | −1 |
| clinical_knowledge | 13/18 | 12/18 | −1 |
| electrical_engineering | 17/18 | 16/18 | −1 |

`prehistory` is a nice microcosm: the direct model was perfect, and reasoning
traded one right answer for a wrong one.

## What it costs

The score went up 8.7 points; the compute went up far more. The direct baseline
emits a single token in ~2.9 s per question. The reasoning arm averaged **1,188
tokens and 42 s** — roughly **14× the wall-clock**.

The average hides a brutal tail:

| | tokens | latency |
|---|---:|---:|
| median | 365 | 13.5 s |
| 75th percentile | 779 | 27.7 s |
| 90th percentile | 3,908 | 135.9 s |
| 95th percentile | 7,593 | 265.7 s |
| max | 8,581 | 301.4 s |

**99 of 1,000 traces exceeded 4,000 tokens, and 50 reached 8,000+ tokens against
the 8,192-token thinking budget.** Those long traces accounted for **58% of all
generated tokens**. Reasoning is cheap on most questions and extremely expensive
on a stubborn minority — and the stubborn minority is where a lot of the gain
lives. The whole 1,000-question study took **11.2 hours** on the one GPU.

## Two questions it got right and wrong

**Fixed — `high_school_mathematics:247`.** "An object moving along a line has
velocity `v(t) = t·cos(t) − ln(t+2)`, `0 ≤ t ≤ 10`. How many times does the
object reverse direction?" The direct model answered `B` (one). The reasoning
trace converted the question into finding sign changes of `v(t)` and answered
`C` (two), matching the gold answer:

> *"Reverse direction when velocity changes sign… `v(0) = −ln2 < 0`. Need to
> count sign changes… t·cos t oscillates between −t and t…"*

**Broke — `econometrics:19`.** "The characteristic roots of the MA process
`y_t = −3u_{t−1} + u_{t−2} + u_t` are…" The direct model answered `B` (1 and
0.5), correct. The reasoning trace got tangled in the possible sign conventions
for an MA characteristic polynomial, second-guessed itself, and answered `A`
(1 and 2):

> *"…characteristic roots are roots of `θ(z) = 1 + θ₁z + …`? Or sometimes roots
> of `1 − θ₁z − …`? Need to infer from options."*

That's the failure mode in miniature: deliberation helps when the path is
computable and the model just needs to walk it, and hurts when the model uses
the extra room to talk itself out of a correct, confident answer.

## The pitfall that nearly broke the study

My first attempt let thinking run unbounded and capped generation at 8,192
tokens. About **9% of questions hit the cap** — and a truncated trace has no
final `Answer:` line, so it scores as wrong. That is a silent bias against
reasoning, manufactured entirely by a config choice.

Worse, the model **over-thought**: one trace spent thousands of tokens
speculating about whether a word had been dropped from the question text,
without ever converging on an answer. You cannot fix that by raising the cap;
the model will happily loop at any limit.

The fix was the server's **reasoning budget** (`--reasoning-budget 8192`). When
the thinking phase reaches the budget, `llama.cpp` injects a message telling the
model to wrap up and give its final answer, instead of cutting it off mid-word.
Combined with a higher `--max-tokens` (16,384) for the answer, the final run
finished with **zero truncated traces and zero unparseable answers**.

If you take one engineering lesson from this: **an answer parser plus an
unbounded generation cap is a biased benchmark.** Decide what a model should do
when it thinks too long, and enforce it deliberately.

## What this does and doesn't say

- It's **one model, one quantization, 1,000 of 14,042 questions**. The paired
  design makes the +8.7 robust *for this sample*, but it is not a claim that
  every model gains equally, or that the leaderboard reorders.
- The Part 1 baseline scores 80.8% on the full test set and 82.5% on this
  1,000-question sample, so the sample is slightly easier. Do **not** compare
  the reasoning arm's 91.2% to the full-set 80.8%; compare paired deltas.
- It's a **method comparison** (raw grammar baseline vs. chat + reasoning
  instruction), as noted above.
- A subtle one: switching `enable_thinking` off is not a magic "no reasoning"
  button. In an earlier chat-based control, the model still reasoned *aloud* on
  the hard questions even with thinking disabled. Which is another reason the
  grammar-constrained Part 1 run makes a cleaner control.

## Reproducing this

Two sibling projects:

```
Benchmark_MMLU/                     # Part 1: the cross-model leaderboard
  benchmark_mmlu.py
  results/<model>/predictions.jsonl # per-question baseline, reused here
  blog_mmlu_local_llms.md

Benchmark_MMLU_reasoning/           # Part 2: this study
  benchmark_mmlu_reasoning.py       # paired harness (reasoning + baseline join)
  run_reasoning.sh
  run_detached.sh                   # setsid launcher for multi-hour runs
  results/qwen3.8-27B-IQ3_S/        # run.json, predictions.jsonl, server.log
  blog_mmlu_reasoning.md
```

A smoke test, then the study:

```bash
cd Benchmark_MMLU_reasoning
LIMIT=20 ./run_reasoning.sh                 # smoke test
LIMIT=1000 ./run_detached.sh                # ~11 h, resumable
```

The harness reads the Part 1 predictions automatically as the control
(`--baseline-results auto`), runs only the reasoning arm, and prints a paired
report. Re-running resumes where it left off. Everything — server lifecycle,
prompting, parsing, McNemar, per-subject deltas — is in the script.

## Where next

The honest sequel to Part 1's teaser is a **cross-model** reasoning table: does
turning reasoning on reorder the leaderboard, or does everyone move by roughly
the same amount? This study answers that for one model. Running it for the other
seven is the experiment I actually want. After that, the natural question is
whether the reasoning gain recovers the quantization gap between `IQ3_S` and
its more aggressively compressed `IQ3_XXS` sibling.

---

*Full disclosure: the numbers above come from a single consumer GPU and a
1,000-question sample of one quantization. The methodology — pair the questions,
reuse the published baseline, test the flips, and bound the thinking — transfers
directly. And the next time you see a local-LLM leaderboard, ask the question
Part 1 ended on: was reasoning on or off? On this model it was worth 8.7 points
and 14× the time.*
