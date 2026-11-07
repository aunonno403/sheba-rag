"""Two chunkers over a document's pages: sentence-aware (default) and fixed-window (baseline).

A page is {"page": int | None, "text": str}. A chunk is {"text", "section", "page"}.
"""
import re

# A line starting with "১।", "১০।", "1." or "12." opens a numbered section (law sections, numbered steps).
SECTION_RE = re.compile(r"^\s*([০-৯]+|\d+)\s*[।.]", re.M)
# Sentence ends: danda / ? / ! anywhere, "." only before whitespace; newlines always break.
SENT_RE = re.compile(r"(?<=[।?!])\s+|(?<=\.)\s+|\n+")


def split_sections(text):
    """Return [(section_label | None, text)], cutting at numbered lines."""
    cuts = [(m.start(), m.group(1)) for m in SECTION_RE.finditer(text)]
    out, prev_pos, prev_label = [], 0, None
    for pos, label in cuts:
        if text[prev_pos:pos].strip():
            out.append((prev_label, text[prev_pos:pos].strip()))
        prev_pos, prev_label = pos, label
    if text[prev_pos:].strip():
        out.append((prev_label, text[prev_pos:].strip()))
    return out


def split_sentences(text):
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]


def _windows(text, size, overlap):
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


def sentence_chunks(pages, max_chars=1000):
    chunks = []
    for p in pages:
        for section, sec_text in split_sections(p["text"]):
            sents = []
            for s in split_sentences(sec_text):
                # a single over-long "sentence" (e.g. an OCR'd table row) is hard-split
                sents += _windows(s, max_chars, 0) if len(s) > max_chars else [s]
            cur = []
            for s in sents:
                if cur and len(" ".join(cur + [s])) > max_chars:
                    chunks.append({"text": " ".join(cur), "section": section, "page": p["page"]})
                    last = cur[-1]
                    cur = [last] if len(last) + 1 + len(s) <= max_chars else []  # one-sentence overlap
                cur.append(s)
            if cur:
                chunks.append({"text": " ".join(cur), "section": section, "page": p["page"]})
    return chunks


def fixed_chunks(pages, size=1000, overlap=150):
    chunks = []
    for p in pages:
        text = p["text"]
        heads = [(m.start(), m.group(1)) for m in SECTION_RE.finditer(text)]
        for i, w in enumerate(_windows(text, size, overlap)):
            start = i * (size - overlap)
            section = next((lab for pos, lab in reversed(heads) if pos <= start), None)
            if w.strip():
                chunks.append({"text": w, "section": section, "page": p["page"]})
    return chunks


CHUNKERS = {"sentence": sentence_chunks, "fixed": fixed_chunks}
