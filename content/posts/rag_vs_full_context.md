---
date: '2026-11-05T00:00:00-05:00'
draft: true
title: 'Giving a Local LLM Your Own PDFs: Context Stuffing vs RAG'
description: "I asked a local Qwen model the same four questions about a 91-page engineering PDF three ways — no context, the whole PDF, and RAG. The failures were the useful part."
author: 'Jorgen Bergstrom'
bluesky: false
tags: ['ML', 'LLM', 'RAG', 'llama.cpp', 'Python']
categories: ['Local LLM']
---

## Introduction

A local LLM knows nothing about *your* files. Give it a spec sheet, a manual, or a set of research papers and it will answer confidently from whatever it memorized during training — even when that answer is wrong for your document. The fix is to put the relevant text in front of it. There are three common ways to do that, sitting on a spectrum of how much text you inject:

1. **No external context.** Cheap, instant, and limited to the model's training knowledge.
2. **Full-context ("stuff the whole document").** Paste the entire PDF into the prompt. Perfect recall — if it fits.
3. **Retrieval-Augmented Generation (RAG).** Split the document into chunks, embed them, and at query time inject only the few chunks that look relevant.

The trade-offs feel obvious, but they are more convincing with numbers. I asked a small local model the same four questions about a 91-page engineering PDF three ways. RAG used about **3.4% of the prompt tokens** of the full-document approach, and it answered two of the four questions as well as full-context did. It also failed the other two in ways that are more instructive than a clean sweep — and one baseline answer was flat-out wrong in a way that makes the whole case for grounding.

If you have not served a model locally before, my [llama.cpp setup post](/posts/install_llama_cpp_and_gemma/) covers the basics.

**The short version:**

| Approach | Prompt tokens / query | Latency / query | Grounded in the doc? |
|---|---:|---:|---|
| **1. No document** | **49** | 13–20 s (rambling, truncated) | No |
| **2. Whole PDF in prompt** | **53,116** | **33.6 s cold → ~5 s warm** | Yes |
| **3. RAG** | **1,785** | ~3–8 s | Mostly |

## The experiment

- **Model:** `Qwen3.5-9B` (Q4_K_M GGUF), served locally by `llama.cpp`.
- **Machine:** one consumer GPU (16 GB) plus CPU, with the model split across both.
- **Document:** *A Diverse Collection of Formulas, Definitions and Derivations in the Realm of Mechanical Engineering* (91 pages, ~147,000 characters, **≈51,400 tokens**).
- **Fairness:** same model, `temperature=0`, `max_tokens=512`, and "thinking" disabled for all three methods. Only the *context* changes.
- **Four questions**, each with a short, exact answer in the document:

| # | Question | What the document actually says |
|---|---|---|
| 1 | Can you explain König's Theorem in words? | Total KE = KE of the total mass moving at the centre-of-mass velocity + KE relative to the centre of mass (§4.4) |
| 2 | What is the so-called zeroth law of thermodynamics? | Two systems in thermal equilibrium with a third are in thermal equilibrium with each other (§5.8) |
| 3 | How many slip systems is there in a FCC material? | 12 — four {111} planes, three ⟨110⟩ directions each (§10.1) |
| 4 | What is the mathematical definition of the Burgers vector? | **b** = ∮_c (∂**u**/∂s) ds, *c* a right-handed circuit around the dislocation (§10.8) |

## Approach 1 — no document

This one is trivial to run:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant. Answer from your own knowledge."},
    {"role": "user", "content": question},
]
```

The model is genuinely good at this material. It correctly gave the zeroth law, the FCC count (12, with the 4×3 breakdown), and the Burgers-vector concept. But look at question 1:

> **"König's Theorem is a fundamental result in graph theory, specifically concerning bipartite graphs. It establishes a precise relationship between two specific types of subgraphs: matchings and vertex covers…"**

That is a *different* König's theorem — the graph-theory one about matchings and vertex covers. The mechanics theorem the document is about (kinetic energy decomposition) shares nothing with it but the name. The model did not say "I'm not sure which you mean"; it picked one and committed. **This is the whole argument for grounding in a single answer:** ungrounded knowledge can be confidently, specifically, and invisibly wrong for your document.

Two more costs people forget. Grounded answers are *shorter*: the ungrounded model **rambled past the 512-token cap on 3 of the 4 answers** and got truncated mid-sentence, while grounded answers finished in 40–170 tokens. And every ungrounded answer is the model's opinion, not your source of truth — a number that changed last week is recited just as confidently as one that did not.

## Approach 2 — paste the whole PDF into the prompt

The simplest possible "RAG": skip retrieval and hand the model everything.

```python
full_text = load_pdf("Solid_Mechanics_Qualifying_Equations.pdf")   # ~51k tokens

messages = [
    {"role": "system", "content": "Answer using the provided context."},
    {"role": "user", "content": f"Context:\n{full_text}\n\nQuestion: {question}"},
]
```

Full-context got **all four questions right**, including the real König's theorem, and it did so in 40–130 completion tokens. Recall is perfect — nothing can be missed because everything is present. It is the accuracy winner.

The bill is enormous. Every query ships **~53,100 prompt tokens** whether the answer is on page 1 or page 91. A cold prefill of the full document took **33.6 seconds** before generating a single token. (The `llama.cpp` server caches the prompt prefix, so subsequent queries dropped to ~3–8 s — in the recorded runs the full-context calls reported 52,604 cached prompt tokens. But a cold server, a different user, or a cache miss pays the full prefill every time.)

And there is a hard wall hiding behind the good result: this only works because the model was launched with a context window large enough to hold the whole document. A typical local setup with 8k–32k context **cannot do this at all** — the document overflows the window before you have asked your question. "Just paste the whole thing" quietly stops working the moment your corpus outgrows the window.

## Approach 3 — RAG

RAG makes the opposite bet: instead of sending everything, send only what is likely relevant. The pipeline has five stages, each a small module in `rag/`:

```text
Documents → Chunking → Embedding → Vector store → Retrieve top-k → Prompt → LLM
```

1. **Chunking** (`document_loader.py`) — split into overlapping token windows so facts spanning a boundary survive. I used the `cl100k_base` tokenizer: **512-token chunks, 128-token overlap**, dropping fragments under 64 tokens.
2. **Embedding** (`embedding.py`) — turn each chunk into a 384-d vector with `all-MiniLM-L6-v2`, L2-normalized so cosine similarity is a dot product.
3. **Vector store** (`vector_store.py`) — FAISS `IndexFlatIP`, carrying each chunk's text and metadata.
4. **Retrieval** (`retriever.py`) — embed the *question* the same way, take the top-k nearest chunks, and pack them into a context string (our run used `top_k=5` and a **2,000-token budget**).
5. **Prompt + generation** (`prompt_template.py`, `pipeline.py`) — wrap the chunks with the question and send it to the same local model.

You drive the whole thing in a few lines:

```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline(
    llm_api_key="llama-cpp",                    # dummy key; llama.cpp needs none
    llm_base_url="http://localhost:8080/v1",
    llm_model="unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-Q4_K_M.gguf",
    chunk_size=512, chunk_overlap=128, top_k=5,
    llm_max_tokens=4096,
    llm_disable_thinking=True,
)
pipeline.add_documents(["Solid_Mechanics_Qualifying_Equations.pdf"])   # or a directory
answer = pipeline.query("What is the mathematical definition of the Burgers vector?")
```

Indexing 91 pages took **about 2 seconds** (134 chunks), and each query pulled in roughly **1,785 prompt tokens** of context.

### What worked

**Q2 (zeroth law)** and **Q4 (Burgers vector)** came back tight and exact — including the document's own integral definition,

$$ \mathbf{b} = \oint_c \frac{\partial \mathbf{u}}{\partial s} ds $$

which the ungrounded model never reached before its 512-token cap. Grounded answers were concise and, crucially, *finished*.

### What failed — and why it is the most useful part

**Q1 (König's theorem) — retrieval miss.** RAG answered:

> *"Based on the provided context, König's Theorem is listed as section 4.4… the text provided only contains the heading for this section and does not include the definition."*

It refused rather than hallucinate — good behavior — but why did it not have the definition? The answer is a masterclass in how RAG actually breaks:

- The definition lives in **chunk 35**, at the very end of a formula-dense chunk about work and energy — so the sentence is *diluted* by several hundred tokens of unrelated text.
- A query about "König's Theorem" matches the **table of contents** (chunks 3–4, with the actual entry in chunk 4), which lists the phrase "König's Theorem" verbatim, far better than it matches the definition chunk.
- So dense search retrieved the *index entry* and never the *content*. Chunk 35 did not even make the top **10**. Shrinking chunks to 256 tokens made it worse, not better.

Two classic RAG hazards in one: a **table of contents can hijack retrieval**, and **terse definitions buried in dense text are hard to retrieve semantically**. The fixes are known — hybrid (BM25 + dense) search so exact terms like "König" match, a cross-encoder re-ranker, or just stripping the TOC from the index.

**Q3 (FCC slip systems) — context-budget miss.** RAG hedged:

> *"…the specific number of slip systems for an FCC material is not explicitly stated… standard materials science knowledge states that an FCC crystal has 12 slip systems…"*

Notice the tell: it fell back on **"standard knowledge"** — the very thing we were trying to avoid. The document says 12 plainly; the relevant chunk did not both make the top-5 and fit the 2,000-token budget. At 512 tokens per chunk, the budget only had room for **3 chunks** in practice. In a follow-up run, raising it to 8,000 tokens (so all retrieved chunks fit) fixed the answer immediately:

> *"Based on the provided context, an FCC material has 12 slip systems… four sets of {111} planes, three ⟨110⟩ directions each."*

So this failure was a **tuning** problem, not an embedding problem. `top_k` and the context budget are the first knobs to turn when RAG goes quiet.

## Putting the numbers together

**Prompt tokens per query** (this is what drives prefill time, KV-cache memory, and API cost):

```text
No document   ▏49
RAG           ███▏1,785
Full PDF      ██████████████████████████████████████████████████████████████ 53,116
```

RAG used **1,785 / 53,116 ≈ 3.4%** of the full-context prompt — a **~30× reduction**.

| Approach | Tokens / query | Cost / 1,000 queries\* | Answers correct (of 4) |
|---|---:|---:|---:|
| No document | 49 | USD 0.015 | 3 |
| **RAG** | 1,785 | **USD 0.54** | **2 clean + 1 partial** |
| Full PDF | 53,116 | USD 15.93 | 4 |

\*Illustrative at USD 0.30 per million input tokens.

**Latency** is subtler than it looks. Full-context is *dominated by prefill*: **33.6 s cold**, then ~3–8 s once the server's prefix cache is warm. RAG was ~3–8 s with no warm-up needed. The baseline's 18 s average is not a win — it is the model padding answers until it hit the cap.

## What I take away

1. **Grounding is not about cost; it is about correctness on *your* document.** The König hallucination — a real theorem, wrong for this document — is exactly the failure that RAG and full-context both prevent.
2. **For a single document, full-context is the accuracy champion.** If it fits your context window and you can stomach the prefill, it has perfect recall and the simplest code.
3. **RAG buys that reliability at ~1/30th the context cost, and scales** — but only if retrieval is good. Its failures here were 100% retrieval/tuning problems, not model problems.
4. **The failures are diagnosable.** One was a context-budget/top-k miss (fixable in a line); the other was a table-of-contents + terse-definition embedding miss (fixable with hybrid search or a re-ranker). Neither was mysterious.
5. **A grounded prompt refuses gracefully** — which is a feature. RAG openly said it did not have the definition instead of inventing one.

## Reproducing this

```bash
qwen3.5_9B                                   # local alias; starts llama.cpp on :8080
.venv/bin/python blog/compare_approaches.py  # three-way run
#   -> blog/results.json   (raw answers, tokens, latency, retrieved chunks)
#   -> blog/results.md     (human-readable)
.venv/bin/python examples/demo.py            # interactive demo
```

`blog/compare_approaches.py` runs all three methods over the question set and records each answer with the API's reported `prompt_tokens`, `completion_tokens`, and latency, plus which chunks actually fit the retrieval budget.

## Caveats

- Single-run results from one small model on one document. Treat them as illustrative, not a benchmark. Bigger models lean less on context, which narrows RAG's quality edge but not its cost edge.
- Embedding quality dominates RAG quality. `all-MiniLM-L6-v2` is small and fast; a stronger embedder or hybrid retrieval would likely have caught the König and FCC questions at default settings.
- The 2,000-token retrieval budget is deliberately modest; a larger budget or higher `top_k` changes the results. That *is* one of the lessons.
- For multiple PDFs, pass a list (or a directory) to `add_documents`; the index scales linearly and the per-query token math does not change.
