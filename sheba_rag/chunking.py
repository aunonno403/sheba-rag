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


def _is_row(line):
    return line.startswith(("-", "•"))


def _headings(lines):
    """For each line, the heading lines above it that it needs to make sense: [group heading, heading]
    for a list or table row ("- Express delivery: TK 8,625"), [group heading] for a heading. Prose gets [].
    A heading is a line directly above rows; a group heading is a line directly above a heading."""
    out, group, head = [], None, None
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if _is_row(line):
            out.append([h for h in (group, head) if h])
        elif _is_row(nxt):
            head = line
            out.append([group] if group else [])
        else:  # prose ends a table, unless this line is the group heading of the next one
            starts_group = i + 2 < len(lines) and _is_row(lines[i + 2])
            group, head = (line if starts_group else None), None
            out.append([])
    return out


def sentence_chunks(pages, max_chars=1000, min_chars=300):
    """Pack sentences into chunks of at most max_chars. A numbered section starts a new chunk, unless the
    current chunk is still under min_chars: then short sections (list items like "19. Sylhet") are packed
    together instead of becoming one tiny chunk each. A chunk that starts inside a list or table starts
    with that row's headings, so a price is never cut off from what it is the price of."""
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
            # a single over-long "sentence" (e.g. an OCR'd table row) is hard-split
            parts = [w for s in split_sentences(sec_text)
                     for w in (_windows(s, max_chars, 0) if len(s) > max_chars else [s])]
            for part, heads in zip(parts, _headings(parts)):
                if cur and len(" ".join(x for x, _ in cur + [(part, None)])) > max_chars:
                    while cur and cur[-1][0] in heads:  # a heading moves to the next chunk with its rows
                        cur.pop()
                    last = cur[-1] if cur else None
                    if cur:
                        flush()
                    if heads:
                        cur = [(h, section) for h in heads]
                    else:  # prose: one-sentence overlap (never a lone table row)
                        fits = last and not _is_row(last[0]) and len(last[0]) + 1 + len(part) <= max_chars
                        cur = [last] if fits else []
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
