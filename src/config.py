from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DB_DIR = ROOT / "data" / "chroma"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_TOKENS = 400
CHUNK_OVERLAP = 60
TOP_K = 8
RERANK_KEEP = 4

SUFFICIENCY_THRESHOLD = 0.35   # tune later with eval data
SUPPORT_THRESHOLD = 0.55       # tune later
MAX_CORRECTIONS = 1