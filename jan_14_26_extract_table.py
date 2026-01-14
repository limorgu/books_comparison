"""
Build a CSV table from extracted page JSONs for quick review / spot-checking.

Reads (recursively):
  ~/Documents/books/inbox_photos/**/*.json

Writes:
  ~/Documents/books/extraction_table.csv

Output columns are FIXED (explicit list). Missing keys become blank.
"""
from ast import main
from datetime import datetime
import shutil
import json
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path.home() / "Documents" / "books" / "inbox_photos"
OUT_CSV = Path.home() / "Documents" / "books" / "extraction_table.csv"

# If you have JSON files that are NOT page sidecars, list them here to skip
SKIP_FILENAMES = {
    # "judge_results.json",
    # "judge_missing_visibility.json",
}

# ✅ Exact columns (in the order you requested)
COLUMNS = [
    "book_name",
    "author",
    "page_number",
    "title",
    "section_heading",
    "source_file",
    "reference",
    "json_path",
    "text",
    "life_stage_flag",
    "family_type",
    "text_length",
    "text_preview",

    "regulation_A_people_pleasing",
    "regulation_B_hyper_control_perfectionism",
    "regulation_C_explosive_outbursts",
    "regulation_D_dissociation_numbing",
    "regulation_E_addictive_self_soothing",
    "regulation_F_parentification",

    "healing_G_corrective_relationships",
    "healing_H_deep_therapy",
    "healing_I_boundaries_distance",
    "healing_J_creative_narrative_work",
    "healing_K_body_based_regulation",
    "healing_L_community_meaning",

    "evidence_snippets",
    "notes",
]


def clean_preview(text: str, max_len: int = 180) -> str:
    """Single-line preview for table display."""
    if not text:
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(x)
    except Exception:
        return None


def _stringify_cell(v: Any) -> str:
    """
    Convert values to CSV-friendly strings.
    - None -> ""
    - list -> join with " | "
    - dict -> JSON string
    - everything else -> str(v)
    """
    if v is None:
        return ""
    if isinstance(v, list):
        # best for evidence_snippets
        return " | ".join("" if x is None else str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def main():
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {ROOT_DIR}")

    rows = []

    json_files = sorted(ROOT_DIR.rglob("*.json"))
    if not json_files:
        print("No JSON files found — run extraction first.")
        return

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Skipping unreadable JSON: {jf} ({e})")
            continue

        row = {}

        # --- copy all expected fields safely ---
        row["book_name"] = data.get("book_name")
        row["author"] = data.get("author")
        row["page_number"] = data.get("page_number")
        row["title"] = data.get("title")
        row["section_heading"] = data.get("section_heading")
        row["life_stage_flag"] = data.get("life_stage_flag")
        row["family_type"] = data.get("family_type")
        row["source_file"] = data.get("source_file")
        row["reference"] = data.get("reference")
        row["json_path"] = str(jf)

        text = data.get("text") or ""
        row["text"] = text
        row["text_length"] = len(text)
        row["text_preview"] = clean_preview(text)

        # regulation + healing scores
        for k in data.keys():
            if k.startswith("regulation_") or k.startswith("healing_"):
                row[k] = data.get(k)

        # lists → pipe-joined strings for CSV safety
        snippets = data.get("evidence_snippets")
        row["evidence_snippets"] = " | ".join(snippets) if isinstance(snippets, list) else None

        row["notes"] = data.get("notes")

        rows.append(row)

    if not rows:
        print("No valid rows collected.")
        return

    # --- sort ---
    rows.sort(
        key=lambda r: (
            (r.get("book_name") or "").lower(),
            r.get("page_number") if isinstance(r.get("page_number"), int) else 10**9,
            (r.get("source_file") or "").lower(),
        )
    )

    # --- write main CSV ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # --- archive copy ---
    outputs_dir = Path.home() / "Documents" / "books" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived_csv = outputs_dir / f"extraction_table_{ts}.csv"
    shutil.copy2(OUT_CSV, archived_csv)

    print(f"Saved table: {OUT_CSV}")
    print(f"Archived copy: {archived_csv}")
    print(f"Rows written: {len(rows)}")
if __name__ == "__main__":
    main()
