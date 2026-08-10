"""
Smart MCQ Solver — RAG-powered Gradio app.

Retrieves live context from Wikipedia for any question, then scores each
answer option using google/flan-t5-base's next-token log-likelihood over
the option letters. No local model checkpoints or FAISS index required —
everything needed is downloaded once at Space startup (the flan-t5-base
weights) and every query hits Wikipedia's public API live.
"""

import requests
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"
WIKI_API = "https://en.wikipedia.org/w/api.php"
OPTION_LETTERS = ["A", "B", "C", "D", "E"]

print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()
DECODER_START_ID = model.config.decoder_start_token_id
print("Model loaded.")


def wikipedia_context(query: str, num_articles: int = 3, chars_per_article: int = 600) -> str:
    """Live retrieval: search Wikipedia, pull the intro extract of the top
    matching articles. No local index — works for any topic at query time."""
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


def score_options(context: str, question: str, options: dict) -> list:
    """Single forward pass through flan-t5-base; reads the model's logit
    for each option letter as the first generated token, rather than
    generating text and parsing it. Only scores options the user filled in,
    so this works for 2-way, 3-way, 4-way, or 5-way questions."""
    letters = [l for l in OPTION_LETTERS if options.get(l, "").strip()]
    if not letters:
        return []

    opts_block = "\n".join(f"{l}: {options[l]}" for l in letters)
    input_text = (
        f"Context: {context[:1500]}\n\n"
        f"Use the context above to answer this multiple-choice question.\n"
        f"Question: {question}\nOptions:\n{opts_block}\n"
        f"The correct answer is:"
    )

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    decoder_input_ids = torch.tensor([[DECODER_START_ID]])

    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
    logits = outputs.logits[0, 0, :]

    option_token_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in letters}
    raw_scores = torch.tensor([logits[option_token_ids[l]].item() for l in letters])
    probs = torch.softmax(raw_scores, dim=0)

    ranked = sorted(zip(letters, probs.tolist()), key=lambda x: -x[1])
    return ranked


def answer_mcq(question, a, b, c, d, e):
    if not question or not question.strip():
        return "Please enter a question.", "", ""
    options = {"A": a, "B": b, "C": c, "D": d, "E": e}
    if sum(1 for v in options.values() if v and v.strip()) < 2:
        return "Please fill in at least two options.", "", ""

    context = wikipedia_context(question)
    ranked = score_options(context, question, options)

    top3 = ranked[:3]
    top3_str = " ".join(letter for letter, _ in top3)
    detail = "\n".join(f"{letter}: {options[letter]}  \u2014  {prob:.1%}" for letter, prob in ranked)
    context_preview = (context[:500] + "...") if len(context) > 500 else context
    if not context_preview:
        context_preview = "(no Wikipedia context found for this question — answer is based on the model's own knowledge only)"

    return top3_str, detail, context_preview


with gr.Blocks(title="Smart MCQ Solver") as demo:
    gr.Markdown(
        "# Smart MCQ Solver\n"
        "Answers multiple-choice questions on **any topic** by retrieving live context "
        "from Wikipedia and scoring each option with a language model — no fine-tuning "
        "on a fixed question set required."
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
            detail_out = gr.Textbox(label="Ranked probabilities", lines=6)
            context_out = gr.Textbox(label="Retrieved Wikipedia context (preview)", lines=6)

    submit.click(answer_mcq, inputs=[question, a, b, c, d, e], outputs=[top3_out, detail_out, context_out])

    gr.Examples(
        examples=[
            ["What is the largest planet in the solar system?", "Earth", "Jupiter", "Mars", "Venus", "Saturn"],
            ["Who wrote the play Romeo and Juliet?", "Charles Dickens", "William Shakespeare", "Jane Austen", "", ""],
        ],
        inputs=[question, a, b, c, d, e],
    )

if __name__ == "__main__":
    demo.launch()
