# rag_qa.py
# pyright: reportCallIssue=false
# pyright: reportOptionalMemberAccess=false

import os
import numpy as np
import faiss
from openai import OpenAI
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Helper to load & chunk the transcript
def load_chunks(txt_path: Path, chunk_size: int = 500) -> list[str]:
    """
    Read a text transcript and split into chunks of approximately `chunk_size` words.
    """
    text = txt_path.read_text(encoding="utf-8")
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

# 2. Generate embeddings locally
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks: list[str]) -> np.ndarray:
    """
    Encode each chunk into a 384-dimensional vector using a local SBERT model.
    """
    return embed_model.encode(chunks, show_progress_bar=True)

# 3. Build the FAISS index
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Create an exact L2 index over the provided embeddings.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))  # type: ignore[reportCallIssue]
    return index

# 4. Querying & answer generation
def answer_question(
    question: str,
    chunks: list[str],
    index: faiss.IndexFlatL2,
    k: int = 5
) -> str:
    """
    Retrieve the most relevant chunks for `question` and ask OpenAI to answer.
    """
    # Embed the question into a (1, dim) array
    q_emb = embed_model.encode([question])
    q_emb = np.array(q_emb, dtype=np.float32)

    # Retrieve top-k chunks: search(x, k) → (distances, indices)
    distances, indices = index.search(q_emb, k)  # type: ignore[reportCallIssue]
    context = "\n\n".join(chunks[i] for i in indices[0])

    # Call the ChatCompletion API once with the retrieved context
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Answer based on the following video context."},
            {"role": "user",   "content": f"Context:\n{context}\n\nQ: {question}"}
        ],
        temperature=0.0,
    )
    # response.choices[0].message.content is always a string
    content = response.choices[0].message.content or ""
    return content.strip()

# 5. CLI orchestration
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ask a question about a YouTube transcript."
    )
    parser.add_argument("video_id", help="YouTube video ID (e.g. 'dQw4w9WgXcQ')")
    parser.add_argument("question", help="Your question about the video")
    parser.add_argument(
        "--chunks", type=int, default=500,
        help="Words per chunk for indexing"
    )
    args = parser.parse_args()

    # Load transcript file
    txt_file = Path("transcripts") / f"{args.video_id}.txt"
    if not txt_file.exists():
        raise FileNotFoundError(f"Transcript not found: {txt_file}")

    # Build index and answer
    chunks = load_chunks(txt_file, chunk_size=args.chunks)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    answer = answer_question(args.question, chunks, index, k=args.chunks)
    print(answer)
