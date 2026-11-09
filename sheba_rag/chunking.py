"""Two chunkers over a document's pages: sentence-aware (default) and fixed-window (baseline).

A page is {"page": int | None, "text": str}. A chunk is {"text", "section", "page"}.
"""
import re

# A line starting with "১।", "১০।", "1. " or "12. " opens a numbered section (law sections, numbered steps).
# The "." must be followed by a space, so decimals like "২.১ নাগরিক সেবা" or "৮.২৫ একর" are not sections.
SECTION_RE = re.compile(r"^\s*([০-৯]+|\d+)\s*(?:।|\.(?=\s))", re.M)
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


def _label(sections):
    """Section label for a chunk: "৭", or "1–20" when it spans several short sections."""
    named = [s for s in sections if s]
    if not named:
        return None
    return named[0] if named[0] == named[-1] else f"{named[0]}–{named[-1]}"


def sentence_chunks(pages, max_chars=1000, min_chars=300):
    """Pack sentences into chunks of at most max_chars. A numbered section starts a new chunk, unless the
    current chunk is still under min_chars: then short sections (list items like "19. Sylhet") are packed
    together instead of becoming one tiny chunk each."""
    chunks = []
    for p in pages:
        cur = []  # (sentence, section) pairs

        def flush():
            chunks.append({"text": " ".join(s for s, _ in cur), "section": _label([sec for _, sec in cur]),
                           "page": p["page"]})

        for section, sec_text in split_sections(p["text"]):
            if cur and len(" ".join(s for s, _ in cur)) >= min_chars:
                flush()
                cur = []
            for s in split_sentences(sec_text):
                # a single over-long "sentence" (e.g. an OCR'd table row) is hard-split
                for part in _windows(s, max_chars, 0) if len(s) > max_chars else [s]:
                    if cur and len(" ".join(x for x, _ in cur + [(part, None)])) > max_chars:
                        flush()
                        last = cur[-1]
                        cur = [last] if len(last[0]) + 1 + len(part) <= max_chars else []  # one-sentence overlap
                    cur.append((part, section))
        if cur:
            flush()
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
