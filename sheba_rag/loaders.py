"""Turn raw HTML/PDF files into pages of clean text.

Bangla government PDFs often have a broken text layer (Nikosh/SutonnyMJ fonts) or none
at all (scans). Such pages are detected by `needs_ocr` and OCR'd with Tesseract.
"""
import re
import subprocess
import unicodedata

import pymupdf
import trafilatura

from sheba_rag import config

BN_WORD = re.compile(r"[ঀ-৿]+")
# Correct Bangla never starts a word with a dependent vowel sign and never doubles one.
BROKEN_WORD = re.compile(r"^[া-ৌৗ]|[া-ৌ]{2}")
BROKEN_THRESHOLD = 0.02  # clean sources measured 0.0, broken PDFs 0.05-0.21
MIN_CHARS = 50


def nfc(text):
    return unicodedata.normalize("NFC", text)


def broken_ratio(text):
    words = BN_WORD.findall(nfc(text))
    if not words:
        return 0.0
    return sum(1 for w in words if BROKEN_WORD.search(w)) / len(words)


def needs_ocr(text):
    return len(text.strip()) < MIN_CHARS or broken_ratio(text) > BROKEN_THRESHOLD


def html_pages(html):
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    # trafilatura drops <strong> that sits outside a <p>; the NID FAQ puts every question there
    html = re.sub(r"<(/?)strong\b[^>]*>", r"<\1p>", html)
    text = trafilatura.extract(html, include_tables=True) or ""
    text = re.sub(r"\n\s*\n+", "\n", re.sub(r"[ \t]+", " ", text))  # bdlaws pages are full of tabs/blank lines
    return [{"page": None, "text": nfc(text.strip())}]


def ocr_page(page):
    png = page.get_pixmap(dpi=300).tobytes("png")
    # psm 4 (single column of variable-size text) keeps Bangla word spacing; psm 3 loses it
    out = subprocess.run(
        [config.TESSERACT_CMD, "stdin", "stdout", "-l", "ben+eng", "--psm", "4"],
        input=png, capture_output=True, check=True,
    )
    return out.stdout.decode("utf-8")


def pdf_pages(path, log=print):
    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, 1):
            text = page.get_text()
            if needs_ocr(text):
                text = ocr_page(page)
                if needs_ocr(text):
                    log(f"  skip page {i}: unreadable even after OCR")
                    continue
            pages.append({"page": i, "text": nfc(text)})
    return pages
