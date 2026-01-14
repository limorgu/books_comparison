"""
Build a CSV table from extracted page JSONs for quick review / spot-checking.

Reads (recursively):
  ~/Documents/books/inbox_photos/**/*.json

Writes:
  ~/Documents/books/extraction_table.csv
"""

import json
import csv
import re
from pathlib import Path

ROOT_DIR = Path.home() / "Documents" / "books" / "inbox_photos"
OUT_CSV = Path.home() / "Documents" / "books" / "extraction_table.csv"


def clean_preview(text: str, max_len: int = 180) -> str:
    """Single-line preview for table display."""
    if not text:
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def _safe_int(x):
    try:
        if x is None or x == "":
            return None
        return int(x)
    except Exception:
        return None


def main():
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {ROOT_DIR}")

    rows = []

    # ✅ recursive read (handles subfolders)
    json_files = sorted(ROOT_DIR.rglob("*.json"))
    if not json_files:
        print("No JSON files found — run extraction first.")
        return

    for jf in json_files:
        # skip accidental non-sidecar jsons if you have any (optional)
        # if jf.name in {"judge_results.json", "judge_missing_visibility.json"}:
        #     continue

        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Skipping unreadable JSON: {jf} ({e})")
            continue

        book_name = (data.get("book_name") or "").strip()
        author = (data.get("author") or "").strip()
        page_number = _safe_int(data.get("page_number"))

        rows.append({
            "book_name": book_name,
            "author": author,
            "section_heading": data.get("section_heading"),
            "page_number": page_number,
            "life_stage_flag": data.get("life_stage_flag"),
            "text_length": len(data.get("text") or ""),
            "text_preview": clean_preview(data.get("text") or ""),
            "source_file": data.get("source_file"),
            "json_path": str(jf),
            "reference": data.get("reference"),
        })

    # ✅ Sort by book_name, then page_number, then filename
    def sort_key(r):
        book = (r["book_name"] or "").lower()
        pn = r["page_number"]
        pn_sort = pn if isinstance(pn, int) else 10**9  # missing pages go last
        src = (r["source_file"] or "").lower()
        return (book, pn_sort, src)

    rows.sort(key=sort_key)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "book_name",
                "author",
                "section_heading",
                "page_number",
                "life_stage_flag",
                "text_length",
                "text_preview",
                "source_file",
                "json_path",
                "reference",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved table: {OUT_CSV}")
    print(f"Rows written: {len(rows)}")


if __name__ == "__main__":
    main()
