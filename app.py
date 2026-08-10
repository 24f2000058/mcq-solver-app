"""
Smart MCQ Solver — RAG-powered Gradio app, ZeroGPU edition.

Retrieves live context from Wikipedia for any question, then asks
Qwen2.5-3B-Instruct to rank the options directly via its own instruction-
following (rather than a manual logit-reading trick, which is more fragile
across different tokenizers/chat templates). Inference runs inside a
@spaces.GPU-decorated function so it only claims a physical GPU for the
few seconds it's actually needed. A background thread periodically pings
this Space's own public URL to keep the CPU container from sleeping,
without touching any GPU quota.
"""

import re
import os
import time
import threading
import requests
import torch
import gradio as gr
import spaces
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"   
WIKI_API = "https://en.wikipedia.org/w/api.php"
OPTION_LETTERS = ["A", "B", "C", "D", "E"]
GPU_CALL_DURATION = 30   # seconds declared to the ZeroGPU scheduler 

print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
model.to("cuda")   # safe at module level on ZeroGPU — the physical GPU attaches transparently
                    # only during @spaces.GPU-decorated calls, this just prepares the model for it
model.eval()
print("Model loaded.")


def wikipedia_context(query: str, num_articles: int = 3, chars_per_article: int = 600) -> str:
    """Live retrieval: search Wikipedia, pull the intro extract of the top
    matching articles. No local index — works for any topic at query time.
    Runs on the CPU container, no GPU needed."""
    try:
        search_resp = requests.get(
            WIKI_API,
            params={
                "action": "query", "list": "search", "srsearch": query,
                "format": "json", "srlimit": num_articles,
            },
            timeout=8,
        )
        search_resp.raise_for_status()
        hits = search_resp.json().get("query", {}).get("search", [])
        titles = [h["title"] for h in hits]

        snippets = []
        for title in titles:
            extract_resp = requests.get(
                WIKI_API,
                params={
                    "action": "query", "prop": "extracts", "exintro": True,
                    "explaintext": True, "titles": title, "format": "json",
                },
                timeout=8,
            )
            extract_resp.raise_for_status()
            pages = extract_resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if extract:
                    snippets.append(extract[:chars_per_article])
        return " ".join(snippets)
    except requests.RequestException:
        return ""


@spaces.GPU(duration=GPU_CALL_DURATION)
def rank_options(context: str, question: str, options: dict) -> list:
    """The only GPU-touching step. Asks the instruct model directly for a
    ranked list of option letters via its chat template, then parses the
    response with a regex rather than relying on next-token logits — more
    robust across models than the manual scoring trick, at the cost of not
    returning calibrated probabilities (we report rank order instead)."""
    letters = [l for l in OPTION_LETTERS if options.get(l, "").strip()]
    if not letters:
        return []

    opts_block = "\n".join(f"{l}: {options[l]}" for l in letters)
    user_prompt = (
        f"Context: {context[:1500]}\n\n"
        f"Question: {question}\n"
        f"Options:\n{opts_block}\n\n"
        f"Rank these options from most to least likely to be correct. "
        f"Respond with ONLY the option letters separated by spaces, most likely first "
        f"(e.g. \"{letters[0]} {letters[1]}\"). Do not explain."
    )
    messages = [
        {"role": "system", "content": "You are a precise assistant that answers multiple-choice questions."},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=20, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # parse ranked letters out of the free-text response, de-duplicated, in order
    found = re.findall(r"\b([A-E])\b", generated)
    ranked = []
    for letter in found:
        if letter in letters and letter not in ranked:
            ranked.append(letter)
    # fallback: if parsing came up short, fill remaining slots in the model's
    # own original option order so the output is always well-formed
    for letter in letters:
        if letter not in ranked:
            ranked.append(letter)

    return ranked


def answer_mcq(question, a, b, c, d, e):
    if not question or not question.strip():
        return "Please enter a question.", "", ""
    options = {"A": a, "B": b, "C": c, "D": d, "E": e}
    if sum(1 for v in options.values() if v and v.strip()) < 2:
        return "Please fill in at least two options.", "", ""

    context = wikipedia_context(question)
    ranked = rank_options(context, question, options)

    top3 = ranked[:3]
    top3_str = " ".join(top3)
    detail = "\n".join(f"{i+1}. {letter}: {options[letter]}" for i, letter in enumerate(ranked))
    context_preview = (context[:500] + "...") if len(context) > 500 else context
    if not context_preview:
        context_preview = "(no Wikipedia context found for this question — answer is based on the model's own knowledge only)"

    return top3_str, detail, context_preview


with gr.Blocks(title="Smart MCQ Solver") as demo:
    gr.Markdown(
        "# Smart MCQ Solver\n"
        "Answers multiple-choice questions on **any topic** by retrieving live context "
        "from Wikipedia and ranking each option with Qwen2.5-3B-Instruct on a ZeroGPU-backed "
        "A100 — no fine-tuning on a fixed question set required."
    )
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Question", lines=3, placeholder="e.g. What is the capital of France?")
            a = gr.Textbox(label="Option A")
            b = gr.Textbox(label="Option B")
            c = gr.Textbox(label="Option C (optional)")
            d = gr.Textbox(label="Option D (optional)")
            e = gr.Textbox(label="Option E (optional)")
            submit = gr.Button("Answer", variant="primary")
        with gr.Column():
            top3_out = gr.Textbox(label="Top-3 prediction (Kaggle submission format)")
            detail_out = gr.Textbox(label="Full ranking", lines=6)
            context_out = gr.Textbox(label="Retrieved Wikipedia context (preview)", lines=6)

    submit.click(answer_mcq, inputs=[question, a, b, c, d, e], outputs=[top3_out, detail_out, context_out])

    gr.Examples(
        examples=[
            ["What is the largest planet in the solar system?", "Earth", "Jupiter", "Mars", "Venus", "Saturn"],
            ["Who wrote the play Romeo and Juliet?", "Charles Dickens", "William Shakespeare", "Jane Austen", "", ""],
        ],
        inputs=[question, a, b, c, d, e],
    )


def _self_ping_loop(interval_seconds: int = 5 * 60 * 60):
    """Runs in a background daemon thread. Periodically makes a real HTTP
    request to this Space's own public URL — the request goes out through
    HF's routing layer and back, which counts as genuine traffic and resets
    the 48h idle-sleep timer. Hits the root page, not the GPU-decorated
    function, so this never touches ZeroGPU quota. No-op if the Space's
    public hostname isn't available (e.g. running locally)."""
    host = os.environ.get("SPACE_HOST")
    if not host:
        space_id = os.environ.get("SPACE_ID")   # format: "username/spacename"
        if space_id and "/" in space_id:
            username, space_name = space_id.split("/", 1)
            host = f"{username}-{space_name}.hf.space".lower()

    if not host:
        print("Self-ping disabled: not running on a Hugging Face Space.")
        return

    url = f"https://{host}/"
    print(f"Self-ping enabled, will hit {url} every {interval_seconds/3600:.1f}h")

    while True:
        time.sleep(interval_seconds)
        try:
            resp = requests.get(url, timeout=15)
            print(f"Self-ping: {resp.status_code}")
        except requests.RequestException as e:
            print(f"Self-ping failed: {e}")


threading.Thread(target=_self_ping_loop, daemon=True).start()

demo.launch()
