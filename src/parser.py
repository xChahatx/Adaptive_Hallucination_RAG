import fitz  # PyMuPDF
from pathlib import Path


def parse_pdf(path: Path) -> list[dict]:
    """Return [{doc_id, page, text}] for one PDF."""
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:
            continue
        pages.append({
            "doc_id": path.stem,
            "page": i,
            "text": text,
        })
    doc.close()
    return pages


def parse_folder(folder: Path) -> list[dict]:
    out = []
    for pdf in sorted(folder.glob("*.pdf")):
        out.extend(parse_pdf(pdf))
    return out


if __name__ == "__main__":
    from src.config import RAW_DIR
    pages = parse_folder(RAW_DIR)
    print(f"{len(pages)} pages parsed")
    if pages:
        print(pages[0]["doc_id"], "| page", pages[0]["page"])
        print(pages[0]["text"][:300])