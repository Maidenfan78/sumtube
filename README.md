# SumTube – YouTube Video Summariser

A one‑file CLI that turns any public YouTube URL into a tidy, ~120‑word summary—without needing a browser, clipboard juggling, or an expensive GPU. It grabs captions when available, falls back to audio + Whisper‑cpp when they don’t, splits the transcript by GPT tokens, then fires the parts through the OpenAI Chat API in parallel.

Future releases will add an interactive “ask‑the‑video” box powered by retrieval‑augmented generation (RAG) so users can query facts like *“Which graphics card had the best price‑to‑performance?”* directly from long reviews.

---

## ✨ Features

* **Caption first, audio second** – uses `youtube-transcript-api`; if no captions, downloads audio with `yt-dlp` and transcribes locally via `whispercpp` wheels (no compiler needed).  
* **Token‑safe chunker** – splits transcripts with `tiktoken`, never overflowing the 128 k context window of GPT‑4o.  
* **Fully async pipeline** – leverages `openai.AsyncClient`, hitting OpenAI in parallel for 3‑4× speed‑ups on long videos.  
* **.env convenience** – project‑root `.env` is auto‑loaded by `python‑dotenv`; no need to touch system variables.  
* **Rich terminal output** – colourised summaries thanks to the `rich` library.  
* **MIT‑licensed & production‑ready** – permissive license lets you embed the summariser anywhere.

---

## 🚀 Quick Start

```bash
git clone https://github.com/<your-user>/sumtube.git
cd sumtube
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env               # store your OpenAI key
python yt_summariser.py https://youtu.be/dQw4w9WgXcQ
```

Example output:

```
Summary: The video explains ... (≈120 words)
```

> **Need an API key?** Create one at <https://platform.openai.com/account/api-keys> and paste it into `.env`.

---

## 🛠️ Installation Details

| Dependency              | Why                           | Windows Wheels |
|-------------------------|-------------------------------|----------------|
| `youtube-transcript-api`| caption grabber               | ✅ |
| `yt-dlp`                | audio‑only download           | ✅ |
| `whispercpp`            | local speech‑to‑text          | ✅ |
| `openai` ≥ 1.14         | Chat & Whisper API            | n/a |
| `tiktoken`              | token counting                | ✅ |
| `python-dotenv`         | load `.env` file              | ✅ |
| `rich`                  | coloured CLI output           | ✅ |

Python 3.8 → 3.12 recommended; many AI wheels don’t yet publish 3.13 binaries.

---

## 🏗️ How It Works

1. **ID extraction** – normalises both `youtu.be/<id>` and `watch?v=<id>` forms.  
2. **Transcription**  
   * **Captions:** `YouTubeTranscriptApi.get_transcript(video_id, languages=["en","en-US","auto"])`  
   * **Fallback STT:** `yt-dlp -x --audio-format m4a` → `Whisper.from_pretrained("tiny.en").transcribe()`  
3. **Chunk & summarise**  
   * Encode transcript with `tiktoken`; slice every ≤ 3000 tokens.  
   * Async‑call `gpt-4o` for each slice, then ask the model to fuse bullet‑point results into one paragraph.  

All temporary audio files are auto‑deleted after transcription.

---

## 📋 Roadmap

| Milestone                                   | Status |
|---------------------------------------------|--------|
| Interactive Q&A box with RAG (FAISS + LC)   | ⏳ |
| Web UI (FastAPI + HTMX)                     | ⏳ |
| Docker image & GitHub Action release        | ⏳ |
| GUI model selector & cost estimator         | ⏳ |

---

## 🤝 Contributing

1. Fork the repo and create a feature branch.  
2. Write tests (`pytest`).  
3. Submit a PR.

Please run `black` and `ruff` before pushing and ensure no secrets are committed.

---

## 📝 License

Released under the **MIT License** — see `LICENSE` for details.
