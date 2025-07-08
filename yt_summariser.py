"""
yt_summariser.py — Turn a YouTube URL into a neat 120-word summary.

Quick install
-------------
pip install youtube-transcript-api openai python-dotenv yt-dlp tiktoken rich whispercpp
# (whispercpp ships Windows wheels; no C++ tool-chain needed)

Environment
-----------
Create a .env file with:
OPENAI_API_KEY=sk-...
"""

# Disable Pylance rules that clash with dynamic imports and return types
# pyright: reportPrivateImportUsage=false
# pyright: reportCallIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportReturnType=false
# pyright: reportOptionalMemberAccess=false

import os, tempfile, subprocess, textwrap, asyncio
from typing import Optional, List

# ── third-party ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound  # type: ignore[reportPrivateImportUsage]
from openai import AsyncClient
import tiktoken
from whispercpp import Whisper  # type: ignore[reportCallIssue]
import rich

load_dotenv()
client = AsyncClient()                 # OPENAI_API_KEY pulled from env

# ── helpers ──────────────────────────────────────────────────────────────────
def get_video_id(url: str) -> str:
    """Accepts https://youtu.be/abc or https://www.youtube.com/watch?v=abc&t=1s."""
    from urllib.parse import urlparse, parse_qs
    p = urlparse(url)
    if p.hostname in ("youtu.be",):
        return p.path.lstrip("/")
    vid = parse_qs(p.query).get("v")
    if not vid:
        raise ValueError("Cannot detect video ID in URL")
    return vid[0]

# ── 1. TRANSCRIPTION ─────────────────────────────────────────────────────────
def transcript_from_youtube(url: str) -> Optional[str]:
    vid = get_video_id(url)
    try:
        txt = YouTubeTranscriptApi.get_transcript(vid, languages=["en","en-US","auto"])
        return " ".join(seg["text"] for seg in txt)
    except (TranscriptsDisabled, NoTranscriptFound):
        return None

def transcript_with_whisper_local(audio_path: str, model="tiny.en") -> str:
    w = Whisper.from_pretrained(model)
    return w.transcribe(audio_path)  # type: ignore[reportCallIssue]

def ensure_audio(url: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)
    subprocess.run(["yt-dlp","-x","--audio-format","m4a","-o",tmp.name,url], check=True, capture_output=True)
    return tmp.name

# ── 2. SUMMARISATION ─────────────────────────────────────────────────────────
enc = tiktoken.encoding_for_model("gpt-4o")
def token_chunks(text: str, max_tok=3000) -> List[str]:
    ids = enc.encode(text)
    for i in range(0, len(ids), max_tok):
        yield enc.decode(ids[i:i+max_tok])

async def summarise(text: str, model="gpt-4o") -> str:
    tasks = []
    for part in token_chunks(text):
        tasks.append(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content":"Summarise this transcript segment."},
                    {"role":"user","content":part}
                ],
                temperature=0.3,
            )
        )
    seg_summaries = [r.choices[0].message.content.strip() for r in await asyncio.gather(*tasks)]
    combo = "\n".join(seg_summaries)
    final = await client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"Condense to a single paragraph (≈120 words)."},
            {"role":"user","content":combo}
        ],
        temperature=0.3,
    )
    return textwrap.fill(final.choices[0].message.content.strip(),90)

# ── MAIN ─────────────────────────────────────────────────────────────────────
async def summarise_youtube(url: str, prefer="captions") -> str:
    txt = transcript_from_youtube(url) if prefer=="captions" else None
    if txt is None:
        audio = ensure_audio(url)
        txt = transcript_with_whisper_local(audio)
        os.remove(audio)
    if txt is None:
        raise RuntimeError("Transcript retrieval failed")
    return await summarise(txt)

if __name__ == "__main__":
    import sys
    if len(sys.argv)<2:
        rich.print("[red]Usage:[/red] python yt_summariser.py <youtube-url>")
        raise SystemExit(1)
    summary = asyncio.run(summarise_youtube(sys.argv[1]))
    rich.print(summary)
