---
title: Smart MCQ Solver
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Smart MCQ Solver

A retrieval-augmented multiple-choice question solver. For any question,
it searches Wikipedia live for relevant context, then scores each answer
option with `google/flan-t5-base` by reading the model's log-likelihood
for each option letter as the first generated token — the same technique
used for the Kaggle Smart MCQ Solver Challenge's retrieval-augmented
pipeline, adapted here to work on any topic rather than a fixed question
set.

## How it works

1. **Retrieve** — the question text is sent to Wikipedia's public search
   API; the introductory extract of the top 3 matching articles is pulled
   as context. No local index or precomputed embeddings are needed.
2. **Score** — the retrieved context, the question, and the filled-in
   options are combined into a single prompt and passed through
   `flan-t5-base` in one forward pass. Rather than generating free text,
   the model's raw output logits for each option letter (A–E) are read
   directly and converted to a probability distribution with softmax.
3. **Rank** — options are sorted by probability; the top three are
   returned in the same space-separated format used for Kaggle MAP@3
   submissions.

## Notes

- Works with 2 to 5 options; leave unused option fields blank.
- Runs entirely on CPU — no GPU required for this model size.
- Retrieval quality depends on Wikipedia having relevant coverage of the
  question's topic; niche or very recent topics may return no context, in
  which case the model falls back on its own pretrained knowledge alone.

Built as part of the DL & GenAI course project (Smart MCQ Solver
Challenge). See the main project repository for the full Kaggle pipeline,
model comparisons, and report.
