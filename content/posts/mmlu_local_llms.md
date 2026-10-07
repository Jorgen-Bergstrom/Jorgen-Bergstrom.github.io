---
date: '2026-10-07T00:00:00-04:00'
draft: true
title: 'Benchmarking Local LLMs on MMLU: What 14,042 Questions Taught Me'
description: "I ran eight local GGUF models against the full MMLU test set — 14,042 questions across 57 subjects — on a single consumer GPU. Here is the harness, the method, and the results."
author: 'Jorgen Bergstrom'
bluesky: true
bluesky_image: /mmlu_local_llms_cover.webp
tags: ['ML', 'LLM', 'llama.cpp', 'Benchmarking']
---

## Introduction

I have a collected a handful of GGUF models sitting in `~/.cache/llama.cpp`.
I wanted a straight answer to the simple question: *how good are
these local models at recalling facts and doing simple reasoning?*

So I built a small harness that runs my local models against the **full MMLU test
set** (14,042 questions across 57 subjects) using `llama.cpp`, one model at a
time, and records every single prediction.

This post covers what MMLU is, how to get the questions, how the benchmark
works, the example questions, and the results. Note that I ran the test
in a non-reasoning mode.

This is **Part 1 of 2**. Here every model answers *directly*, in a single
token, with no room to deliberate. In **[Part 2 — Does Reasoning Help? A Paired
MMLU Study](/posts/mmlu_reasoning/)** I turn reasoning on for one of these
models, and put a number on what deliberation is worth — and what it costs.

## What is MMLU?

MMLU (Massive Multitask Language Understanding) is a multiple-choice benchmark
introduced by Hendrycks et al. in 2020. Each question has four options (A–D)
and belongs to one of 57 subjects, from `abstract_algebra` and `astronomy` to
`professional_law` and `virology`. It's a popular proxy for general knowledge
and reasoning.

The full test split is **14,042 questions**. The dataset also ships a small
**dev** split with exactly 5 questions per subject — the standard source for
**5-shot** prompting — and a **val** split.

## How to get the MMLU questions

There are two easy routes.

**Option 1 — the original dataset (CSV files).** The official repository is
[`hendrycks/test`](https://github.com/hendrycks/test), and the data tarball is
hosted at:

```bash
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xf data.tar
```

You end up with directories like `data/test/`, `data/dev/`, and `data/val/`,
each containing one CSV per subject (e.g. `astronomy_test.csv`). Every row has
the form:

```text
question, A, B, C, D, answer
```

**Option 2 — Hugging Face Datasets.** If you prefer Python:

```python
from datasets import load_dataset

# "all" gives every subject; you can also request a single subject
mmlu = load_dataset("cais/mmlu", "all")
print(mmlu["test"][0])
```

In my project the data lives under `mmlu/data/{test,dev,val,auxiliary_train}/`
with one `<_subject>_<split>.csv` file per subject — 57 files in `test/` alone.
Whatever route you take, make sure you keep the *dev* split around: it supplies
the 5-shot examples.

## Some example questions

Here are real questions from the test split, so you can see the flavour
(answers at the end of each block).

**Astronomy** - *answer: A*

> What is true for a type-Ia ("type one-a") supernova?
> A. This type occurs in binary systems.
> B. This type occurs in young galaxies.
> C. This type produces gamma-ray bursts.
> D. This type produces high amounts of X-rays.

**Abstract algebra** - *answer: D*

> Find all zeros in the indicated finite field of the given polynomial with
> coefficients in that field. `x^5 + 3x^3 + x^2 + 2x` in `Z_5`.
> A. 0
> B. 1
> C. 0,1
> D. 0,4

**Machine learning** - *answer: C*

> Statement 1 | RoBERTa pretrains on a corpus that is approximately 10x larger
> than the corpus BERT pretrained on.
> Statement 2 | ResNeXts in 2018 usually used tanh activation functions.
> A. True, True
> B. False, False
> C. True, False
> D. False, True

**Moral scenarios** - *answer: D*

> For which of these two scenarios does the main character (who uses I/me/my)
> do something clearly morally wrong, according to ordinary moral standards in
> the US as of 2020?
> Scenario 1 | I slammed on my breaks to miss the ball as it came in to the road.
> Scenario 2 | I taught my children to play the xylophone.
> A. Wrong, Wrong
> B. Wrong, Not wrong
> C. Not wrong, Wrong
> D. Not wrong, Not wrong

**Virology** - *answer: D*

> AIDS activism in the U.S. resulted in:
> A. Food and Drug Administration reforms
> B. Easier access to investigational drugs
> C. Changes in the way medicine was practiced in the U.S.
> D. All of the above

What makes this interesting is that MMLU spans trivia, formal math,
professional knowledge, and judgement calls in the same benchmark.

## How I ran the benchmark

Everything runs on a computer with a single **NVIDIA RTX 5060 Ti (16 GB)**
using `llama.cpp`'s `llama-server`. For each model, in sequence:

1. Start `llama-server` with the model's GGUF and launch flags.
2. For each test question, build a standard **5-shot prompt** using the 5 dev
   examples for that subject.
3. Decode **greedily** (`temperature = 0`) with a **GBNF grammar** that forces
   the output to be exactly one of `A | B | C | D`, capped at one token.
4. Score by exact letter match against the gold answer.
5. Stop the server and move to the next model.

A few design choices that matter:

- **Grammar-constrained single letters** make scoring exact and remove the
  "the model rambled and we guess-parsed it" problem.
- **Greedy decoding** makes runs reproducible.
- Requests run **4-way concurrent** (the server is started with `-np 4`).
- Runs are **resumable**: predictions stream to
  `results/<model>/predictions.jsonl`, so a crash doesn't cost the whole run.

## The results

Here is the final table. All runs completed the full 14,042 questions per model
with **zero failures**.

| Model | Overall | Macro avg | Correct/Answered |
|---|---:|---:|---:|
| qwen3.6-35B-A3B | **84.1%** | 84.9% | 11813 / 14042 |
| qwen3.5-35B-A3B | **83.8%** | 84.5% | 11772 / 14042 |
| qwen3.8-27B-IQ3_S | 80.8% | 81.8% | 11342 / 14042 |
| qwen3.8-27B-IQ3_XXS | 79.3% | 81.1% | 11130 / 14042 |
| qwen3.5-9B | 79.0% | 80.1% | 11096 / 14042 |
| qwen3-coder-30B-A3B | 76.7% | 78.7% | 10771 / 14042 |
| gemma4-26B-A4B | 60.9% | 62.0% | 8550 / 14042 |
| gemma4-12B | 60.5% | 60.6% | 8493 / 14042 |

*Overall = micro accuracy over all answered questions. Macro = unweighted mean
of the 57 per-subject accuracies. `IQ3` / `Q4` etc. are GGUF quantization
levels.*

### What I take away

- **The Qwen family dominates.** Six of eight models clear 76%, and the top two
  (both 35B-A3B Mixture-of-Experts) sit within 0.3 points of each other at
  ~84%. That gap is noise, not a real ranking.
- **Small doesn't mean weak.** `qwen3.5-9B` scores 79.0% — it beats the 30B
  `qwen3-coder` and all but matches the 27B `qwen3.8-XXS` (79.3%), despite
  being a fraction of the size.
- **Quantization barely matters here.** `qwen3.8-IQ3_S` beats its more
  aggressively quantized `IQ3_XXS` sibling by ~1.5 points — worth knowing if
  you're squeezing a model onto a small GPU.
- **Specialists pay a tax.** The code-tuned `qwen3-coder-30B-A3B` lands below
  the general-purpose models on a broad-knowledge benchmark.
- **The Gemma models are the outlier.** Both sit roughly 16 points below the
  weakest Qwen and 23 below the strongest, and the larger one buys almost
  nothing over the smaller one (60.9% vs 60.5%). Their error patterns also show
  a strong bias toward a single answer position: `gemma4-26B-A4B` picks **A**
  33.8% of the time, against 22.9% in the gold answers, while `gemma4-12B`
  over-predicts **D** (30.4% vs 26.9%). That's a sign they're leaning on surface
  statistics rather than the content.

## Note: these are non-reasoning scores

This is worth stating plainly: **none of these numbers involve reasoning or
"chain-of-thought."**

The harness calls the raw `/completion` endpoint and constrains the output to
a single letter with a grammar and a one-token budget. Each model generated
**exactly one token** — the answer. There was no room to think, so a model's
"thinking mode" was never exercised, even though some of these models have one.

That matters, because reasoning is a real, tunable feature on some of these
models. For example, the **Qwen3.8** chat template exposes:

- `enable_thinking` (on/off), and
- `reasoning_effort` with levels **`xhigh` (default), `medium`, `low`**
  (`high` is accepted as an alias for `xhigh`).

You enable it through the chat path, e.g. a `chat_template_kwargs` field on
`/v1/chat/completions`, or server flags like `--reasoning-effort xhigh` and
`--reasoning-format deepseek` (which puts the thoughts in a separate
`reasoning_content` field and leaves the answer in `content`).

Turned on, a model produces a visible trace before answering:

```text
reasoning_content: "We need answer user's simple multiple choice. Need final
                    one letter. Answer A."
content: "A"
```

So the honest framing is: **the table above measures each model's ability to
answer directly, in one shot, with no deliberation.** It's a fair and fully
reproducible comparison - but it is *not* a measure of reasoning ability, and
it's likely that thinking-capable models would score higher (or, occasionally,
worse through overthinking) in a reasoning-enabled mode. The `qwen3-coder`
model, notably, has no thinking mode at all, so a reasoning benchmark could
shift the ranking.

If you want the reasoning numbers, you have to change more than a flag: use
the chat endpoint so the template engages, allow a large generation budget,
and then parse the final letter out of the answer. That is exactly what
**[Part 2](/posts/mmlu_reasoning/)** does.


## Reproducing this

The harness is on GitHub:
[`Jorgen-Bergstrom/mmlu-local-benchmark`](https://github.com/Jorgen-Bergstrom/mmlu-local-benchmark).
The project layout is simple:

```text
benchmark_mmlu.py          # server lifecycle + 5-shot client + scoring
run_all_models.sh          # runs every registered model sequentially
mmlu/data/{test,dev,...}/  # the MMLU CSVs
results/<model>/           # run.json, predictions.jsonl, server.log
results/summary.md         # cross-model table
```

Requirements: a `llama.cpp` build with `llama-server`, Python 3 with
`requests`, and the MMLU CSVs. Everything else — server management, prompting,
scoring, and the summary table — is handled by the script. The repo carries the
per-model `run.json` results and the full per-question prediction sets (as
gzipped release assets) if you want to check my numbers or re-cut them.

**Next up:** [Part 2 — Does Reasoning Help? A Paired MMLU
Study](/posts/mmlu_reasoning/). Same questions, reasoning on, and a paired
statistical test on the difference. If you only read one takeaway from this
series, let it be this: a non-reasoning number and a reasoning number are not
interchangeable, and the gap between them is the whole story.
