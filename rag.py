"""
rag.py - the core of the RAG chatbot.

RAG = Retrieval-Augmented Generation. The steps in this file:
  1. read_pdf      -> read the text of each PDF page
  2. split_text    -> cut page text into small overlapping chunks
  3. build_index   -> turn chunks into embeddings and store them in ChromaDB
  4. retrieve      -> find the chunks most similar to the user's question
  5. ask           -> give those chunks + the question to the LLM (Gemini)
"""
import json
import os
import time

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

from headlines import extract_headlines, guess_title, rewind

load_dotenv()  # reads GEMINI_API_KEY and LLM_MODEL from the .env file

# ---- Settings you can experiment with ----
# Embeddings: ChromaDB creates them for us with its built-in free model (all-MiniLM-L6-v2),
# which runs on your laptop and needs no extra installs.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")  # change in .env if Google renames models
CHUNK_SIZE = 800     # characters per chunk
CHUNK_OVERLAP = 150  # characters shared between neighbouring chunks
TOP_K = 4            # how many chunks to retrieve for each question
DB_PATH = "chroma_db"
COLLECTION = "documents"
HEADLINES_PATH = "headlines.json"  # the topic list is saved here so it survives a restart
TITLE_PATH = "title.txt"           # the page title that matches your documents

def get_db():
    return chromadb.PersistentClient(path=DB_PATH)


# ---- Step 1: read the PDF ----
def read_pdf(file):
    """Return a list of (page_number, text) for a PDF path or file object."""
    rewind(file)
    reader = PdfReader(file)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages


# ---- Step 2: split text into chunks ----
def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Cut text into pieces of `size` characters; neighbours share `overlap` characters
    so an idea that sits on a boundary is not lost."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


# ---- Step 3: embed chunks and store them ----
def build_index(files):
    """files = list of (file_name, file_or_path). Returns the number of chunks stored."""
    db = get_db()
    try:
        db.delete_collection(COLLECTION)  # start fresh each time
    except Exception:
        pass
    collection = db.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, texts, metas, headlines = [], [], [], []
    for name, file in files:
        headlines.extend(extract_headlines(file, name))  # the headings shown as topics
        for page_number, page_text in read_pdf(file):
            for i, chunk in enumerate(split_text(page_text)):
                ids.append(f"{name}-p{page_number}-c{i}")
                texts.append(chunk)
                metas.append({"source": name, "page": page_number})

    if not texts:
        return 0

    # An embedding turns text into a list of numbers (similar meaning = similar numbers).
    # ChromaDB creates the embeddings for us when we add the documents.
    for i in range(0, len(texts), 500):
        collection.add(ids=ids[i:i + 500], documents=texts[i:i + 500], metadatas=metas[i:i + 500])
    with open(HEADLINES_PATH, "w", encoding="utf-8") as f:
        json.dump(headlines, f, ensure_ascii=False)
    with open(TITLE_PATH, "w", encoding="utf-8") as f:
        f.write(make_title([name for name, _ in files], headlines, texts))
    return len(texts)


def make_title(names, headlines, texts):
    """Ask the LLM for a short title that fits the documents.
    If that fails (busy, no key), use a title made from the headlines or file names."""
    fallback = guess_title(names, headlines)
    try:
        heads = "; ".join(h["text"] for h in headlines[:15])
        start = " ".join(texts[:6])[:1500]
        prompt = (
            "Write a short title (at most 7 words) that describes what these documents are about. "
            "Reply with the title only, no quotes.\n\n"
            f"Headings: {heads}\n\nStart of the text: {start}"
        )
        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(timeout=20_000),
        )
        response = client.models.generate_content(model=LLM_MODEL, contents=prompt)
        title = response.text.strip().splitlines()[0].strip(' "*#')
        if 3 <= len(title) <= 80:
            return title
    except Exception:
        pass
    return fallback


def load_title():
    """Return the saved title (empty if no documents were indexed yet)."""
    try:
        with open(TITLE_PATH, encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def load_headlines():
    """Return the saved list of headlines (empty if none were found yet)."""
    try:
        with open(HEADLINES_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def index_ready():
    try:
        return get_db().get_collection(COLLECTION).count() > 0
    except Exception:
        return False


# ---- Step 4: retrieve the most relevant chunks ----
def retrieve(question, k=TOP_K):
    collection = get_db().get_collection(COLLECTION)
    result = collection.query(query_texts=[question], n_results=k)  # Chroma embeds the question too
    chunks = []
    for text, meta in zip(result["documents"][0], result["metadatas"][0]):
        chunks.append({"text": text, "source": meta["source"], "page": meta["page"]})
    return chunks


# ---- Step 5: ask the LLM using only the retrieved chunks ----
def ask(question):
    """Return (answer, chunks_used)."""
    chunks = retrieve(question)
    context = "\n\n".join(
        f"[{c['source']}, page {c['page']}]\n{c['text']}" for c in chunks
    )
    prompt = (
        "You are a helpful study assistant. Answer the question using ONLY the context below. "
        "If the answer is not in the context, say \"I could not find this in the documents.\" "
        "Mention the source file and page number you used.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )
    # timeout is in milliseconds: if Google does not answer within 60 seconds, stop waiting
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options=types.HttpOptions(timeout=60_000),
    )

    # The free service is sometimes busy (error 503) or rate-limited (429).
    # Wait a little and try again instead of giving up straight away.
    waits = [3, 8, 15, 30]
    for attempt in range(len(waits) + 1):
        try:
            response = client.models.generate_content(model=LLM_MODEL, contents=prompt)
            return response.text, chunks
        except Exception as error:
            text = f"{type(error).__name__} {error}"
            busy = any(
                word in text
                for word in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "Timeout", "timed out", "DEADLINE")
            )
            if not busy or attempt == len(waits):
                raise
            time.sleep(waits[attempt])
