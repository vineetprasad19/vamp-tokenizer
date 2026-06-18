# Token Counter

A tiny website that counts **OpenAI tokens** and **characters** in any text —
like the [OpenAI tokenizer](https://platform.openai.com/tokenizer), but it runs
**real Python in your browser**.

https://vineetprasad19.github.io/vamp-tokenizer/

It uses OpenAI's actual [`tiktoken`](https://github.com/openai/tiktoken) library
compiled to WebAssembly via [Pyodide](https://pyodide.org). There is **no
backend** — your text never leaves your device, so it works perfectly on GitHub
Pages (static hosting).

## Features

- Live **token count** and **character count** (plus word count) as you type
- Switch between encodings used by different model families:
  - `o200k_base` — GPT-4o, GPT-4.1, o1/o3/o4-mini
  - `cl100k_base` — GPT-4, GPT-3.5 Turbo, text-embedding-3
  - `p50k_base` — Codex, text-davinci-002/003
  - `r50k_base` — GPT-3 (davinci, curie, babbage, ada)
- Visualize how the text is split into tokens, and view the raw token IDs
- **Vector Graph** — embed the unique words in your text with a small in-browser
  AI model ([Transformers.js](https://github.com/huggingface/transformers.js),
  `all-MiniLM-L6-v2`) and see them grouped by meaning in a rotatable **3D** X/Y/Z
  scatter (PCA + k-means). Click *Generate*, then drag to rotate and scroll to
  zoom; the model (~25 MB) downloads once on first use.
- **Context window meter** — a bar showing how much of a model's context window
  (8K / 16K / 128K / 200K) your current text would fill.
- **Tokenizer comparison** — the same text tokenized by GPT-4o / GPT-4 / Codex /
  GPT-3 side by side, with multilingual & emoji example presets, showing why
  non-English text costs more tokens.
- **Next-token predictor** — runs `distilGPT-2` in the browser to show the top
  predicted **next tokens** with probability bars; click one to generate. A
  **temperature** slider reshapes the distribution (low = safe, high = creative),
  **top-k / top-p** sliders trim the candidate pool, and a **Sample** button draws
  a token from it — the core idea of how LLMs write text, plus how decoding works.
- **"LLMs explained like you're 5"** ([learn.html](learn.html)) — a plain-language
  FAQ page with real-life examples for every key concept.
- **Semantic search (mini-RAG)** — rank sentences by meaning against a question;
  the retrieve step behind "chat with your documents".
- **Sentiment classifier** — DistilBERT labels text positive/negative in-browser.
- **Image recognition (vision)** — a Vision Transformer names what's in an
  uploaded image (top-5 with confidence).
- **Speech-to-text** — record your mic and transcribe it with Whisper (tiny),
  in-browser.
- **Concept explainer cards** — collapsible "💡 What is …?" notes throughout that
  explain tokens, token IDs, embeddings, context windows, generation, RAG,
  classification, vision, and speech in plain language.

## How it works

GitHub Pages can only serve static files, so there is no Python *server*.
Instead, Python runs **in the visitor's browser**:

1. `app.js` boots [Pyodide](https://pyodide.org) and loads the `tiktoken` and
   `regex` packages (both ship with Pyodide).
2. The BPE vocab files in [`encodings/`](encodings/) are bundled in this repo.
   OpenAI's blob storage blocks browser (CORS) requests, so we ship the files
   ourselves and drop them into tiktoken's on-disk cache (see
   [`tokenizer.py`](tokenizer.py)). `tiktoken.get_encoding()` then reads them
   locally instead of hitting the network.
3. As you type, the text is tokenized and the counts/visualization update.

The first page load downloads the Pyodide runtime (~10 MB) and is then cached by
the browser.

## Run locally

Any static file server works (you can't use `file://` because of `fetch`):

```bash
# from this folder
python -m http.server 8000
# then open http://localhost:8000
```

## Deploy to GitHub Pages

1. Create a new repo (e.g. `token-counter`) and push this folder to it.
2. In the repo: **Settings → Pages → Build and deployment → Source: Deploy from
   a branch**, pick `main` and `/ (root)`, then **Save**.
3. Your site goes live at `https://<your-username>.github.io/token-counter/`.

## Updating the vocab files

The `.tiktoken` files were downloaded from OpenAI's public encodings:

```
https://openaipublic.blob.core.windows.net/encodings/<name>.tiktoken
```

They rarely change. If you ever need to refresh them, re-download the same file
names into `encodings/`.
