"""
headlines.py - finds the headlines (headings) inside a PDF so the app can show them as topics.

Two methods, tried in this order:
  1. Font size: text that is clearly bigger than the normal text is a headline.
  2. Line patterns (if the PDF has the same size everywhere): lines in CAPITALS,
     lines ending with a colon, numbered lines such as "1. Hydrated lime", or "Unit 3".

Only works on PDFs that contain real text. A scanned or handwritten PDF has no text to read.
"""
import re
from collections import Counter

from pypdf import PdfReader

MAX_HEADLINES = 80


def rewind(file):
    """Go back to the start of an uploaded file so it can be read again."""
    if hasattr(file, "seek"):
        file.seek(0)


def _pieces(page):
    """Return a list of (text, font_size) for every piece of text on a page."""
    pieces = []

    def visitor(text, cm, tm, font_dict, font_size):
        try:
            clean = " ".join(text.split())
            if clean:
                scale = (abs(tm[3]) or 1) * (abs(cm[3]) or 1)
                pieces.append((clean, round((font_size or 1) * scale, 1)))
        except Exception:
            pass

    try:
        page.extract_text(visitor_text=visitor)
    except Exception:
        pass
    return pieces


def _by_font_size(pages_pieces):
    """Headlines = text that is at least 15% bigger than the most common (body) size."""
    counts = Counter()
    for pieces in pages_pieces.values():
        for text, size in pieces:
            counts[size] += len(text)
    if not counts:
        return []
    body_size = counts.most_common(1)[0][0]

    found = []
    for page, pieces in pages_pieces.items():
        run, run_size = [], 0
        for text, size in pieces + [("", 0)]:  # the empty piece closes the last run
            is_headline = size >= body_size * 1.15 and len(text) >= 3
            if is_headline and (not run or size == run_size):
                run.append(text)  # same big size: it is the same headline
                run_size = size
                continue
            if run:
                found.append((page, " ".join(run)))
                run = []
            if is_headline:  # a different big size starts a new headline
                run, run_size = [text], size
    return found


def _looks_like_headline(line):
    if len(line.split()) > 10:
        return False
    if line.endswith(":"):
        return True
    letters = [c for c in line if c.isalpha()]
    if len(letters) >= 3 and all(c.isupper() for c in letters) and not line.endswith("."):
        return True
    if re.match(r"^(\d+|[IVXivx]+|[A-Za-z])[\.\)]\s+\S", line) and not line.endswith("."):
        return True
    if re.match(r"^(unit|chapter|module|topic|section)\s*[-:]?\s*\d+", line, re.I):
        return True
    return False


def _by_line_pattern(pages_text):
    found = []
    for page, text in pages_text.items():
        for line in text.splitlines():
            line = " ".join(line.split())
            if 3 <= len(line) <= 70 and _looks_like_headline(line):
                found.append((page, line))
    return found


def file_title(name):
    """'CEHT_unit-3.pdf' -> 'CEHT unit 3'"""
    stem = name.rsplit(".", 1)[0]
    return re.sub(r"[_\-]+", " ", stem).strip() or name


def guess_title(names, headlines):
    """A title made only from the content (no AI): the first headline of each PDF,
    or the file name if the PDF has no clear headline."""
    parts = []
    for name in names:
        mine = [h["text"] for h in headlines if h.get("source") == name]
        parts.append(mine[0] if mine and len(mine[0]) <= 60 else file_title(name))
    title = " | ".join(parts[:2])
    if len(parts) > 2:
        title += f" + {len(parts) - 2} more"
    return title


def extract_headlines(file, source=""):
    """Return a list like [{"source": "notes.pdf", "page": 1, "text": "SQL Basics"}, ...]."""
    rewind(file)
    reader = PdfReader(file)
    pages_pieces, pages_text = {}, {}
    for number, page in enumerate(reader.pages, start=1):
        pages_pieces[number] = _pieces(page)
        pages_text[number] = page.extract_text() or ""

    found = _by_font_size(pages_pieces)
    if len(found) < 2:  # font sizes did not reveal headings, try the line patterns
        found = _by_line_pattern(pages_text)

    result, seen = [], set()
    for page, text in found:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append({"source": source, "page": page, "text": text})
    return result[:MAX_HEADLINES]
