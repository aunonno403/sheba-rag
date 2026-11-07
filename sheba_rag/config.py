import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
# Windows consoles default to cp1252, which cannot print Bangla
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SOURCES = DATA / "sources.csv"
RAW_DIR = DATA / "raw"
TEXT_DIR = DATA / "text"
CHROMA_DIR = DATA / "chroma"

LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-3.5-flash-lite")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini/gemini-3.1-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
CHUNKER = os.getenv("CHUNKER", "sentence")
TOP_K = int(os.getenv("TOP_K", "5"))
# Below this top-1 cosine score we answer "I don't know" without calling the LLM. Only a guard
# against clearly off-topic questions: answerable and unanswerable scores overlap (see README).
# Tuned for bge-m3; e5 scores sit much higher (0.75+), so 0.40 is a no-op there.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.40"))
TESSERACT_CMD = (
    os.getenv("TESSERACT_CMD")
    or shutil.which("tesseract")
    or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
