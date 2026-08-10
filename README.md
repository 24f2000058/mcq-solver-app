---
title: Smart MCQ Solver
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
sdk_version: 6.22.0
---

# Smart MCQ Solver

A retrieval-augmented multiple-choice question solver. For any question,
it searches Wikipedia live for relevant context, then ranks the answer
options using `Qwen/Qwen2.5-3B-Instruct` running on Hugging Face's
ZeroGPU (time-sliced A100 access) — the same retrieval-augmented idea
behind the Kaggle Smart MCQ Solver Challenge pipeline, adapted here to
work on any topic rather than a fixed question set.

## How it works

1. **Retrieve** — the question text is sent to Wikipedia's public search
   API; the introductory extract of the top 3 matching articles is pulled
   as context. No local index or precomputed embeddings are needed, and
   this step runs on the Space's CPU container, not the GPU.
2. **Rank** — the retrieved context, the question, and the filled-in
   options are formatted through the model's chat template and passed to
   `Qwen2.5-3B-Instruct`, which is asked to return the option letters
   ranked most-to-least likely. This runs inside a `@spaces.GPU`-decorated
   function, so a physical GPU is only attached for the few seconds this
   call takes.
3. **Parse** — the model's free-text response is parsed for valid option
   letters with a regex, de-duplicated in order, and returned in the same
   space-separated format used for Kaggle MAP@3 submissions.

## ZeroGPU notes

- `@spaces.GPU(duration=30)` declares a tight, realistic time budget to
  the ZeroGPU scheduler — HF checks quota against this *declared* value,
  not actual runtime, so keeping it small avoids wasting quota and ranks
  higher in the scheduling queue.
- The model is loaded and moved to `"cuda"` at module level (Space
  startup), which is the standard ZeroGPU pattern — the physical GPU
  attaches transparently only during decorated calls.
- A `/health` endpoint is mounted alongside the Gradio UI via
  `gr.mount_gradio_app`. It does not call the GPU-decorated function, so
  pinging it costs **zero ZeroGPU quota** — it only keeps the Space's CPU
  container from sleeping (free-tier Spaces sleep after 48h idle). See
  `.github/workflows/keep-warm.yml` for a scheduled GitHub Actions ping
  every 5 hours.

## Notes

- Works with 2 to 5 options; leave unused option fields blank.
- Retrieval quality depends on Wikipedia having relevant coverage of the
  question's topic; niche or very recent topics may return no context, in
  which case the model falls back on its own pretrained knowledge alone.
- To use a lighter/faster model, swap `MODEL_NAME` in `app.py` to
  `Qwen/Qwen2.5-1.5B-Instruct`.

Built as part of the DL & GenAI course project (Smart MCQ Solver
Challenge). See the main project repository for the full Kaggle pipeline,
model comparisons, and report.
