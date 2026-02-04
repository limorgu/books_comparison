"""
Build a CSV table from extracted page JSONs for quick review / spot-checking.

Reads:
  ~/Documents/books/inbox_photos/data_test/**/*.json

Writes (timestamped):
  ~/Documents/books/inbox_photos/data_test/jsontable_YYYYMMDD_HHMMSS.csv

Notes:
- Prefers subtitle/title found inside each JSON.
- If missing, tries common fallback keys (page_title, page_subtitle).
"""

from datetime import datetime
import json
import csv
import re
from pathlib import Path

INBOX_DIR = Path.home() / "Documents" / "books" / "inbox_photos" / "data_test"
OUT_CSV = Path.home() / "Documents" / "books" / "inbox_photos" / "data_test" / "jsontable"  # base name (no .csv)

print("INBOX_DIR:", INBOX_DIR)


def clean_preview(text: str, max_len: int = 180) -> str:
    """Single-line preview for table display."""
    if not text:
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def _clean_str(x):
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _pick_first(data: dict, keys: list[str]):
    """Return first non-empty string value among keys."""
    for k in keys:
        v = data.get(k)
        v = _clean_str(v)
        if v:
            return v
    return None


def main():
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = OUT_CSV.parent / f"{OUT_CSV.stem}_{run_ts}.csv"

    if not INBOX_DIR.exists():
        raise FileNotFoundError(f"Missing inbox folder: {INBOX_DIR}")

    rows = []

    # Search recursively
    json_files = sorted(INBOX_DIR.glob("**/*.json"))
    if not json_files:
        print("No JSON files found — run extraction first.")
        return

    print(f"Found JSON files: {len(json_files)}")

    unreadable = 0

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            unreadable += 1
            print(f"Skipping unreadable JSON: {jf.name} ({e})")
            continue

        # ---- Subtitle/Title hydration (fixes your missing subtitle issue) ----
        # Prefer your schema keys ("subtitle"/"title"), but also accept older keys.
        title = _pick_first(data, ["title", "page_title", "chapter_title"])
        subtitle = _pick_first(data, ["subtitle", "page_subtitle", "section_subtitle"])

        # If your extraction stored title only (and subtitle missing), keep subtitle None.
        # If you prefer to copy title into subtitle when subtitle is missing, uncomment:
        # if subtitle is None and title is not None:
        #     subtitle = title

        row = {
            # Core metadata
            "book_name": data.get("book_name"),
            "author": data.get("author"),
            "page_number": data.get("page_number"),
            "title": title,
            "subtitle": subtitle,
            "text": data.get("text"),
            "reference": data.get("reference"),
            "source_file": data.get("source_file"),
            "narration_voice": data.get("narration_voice"),

            # Family / environment
            "family_type": data.get("family_type"),

            # Main character dysregulation (MC ONLY)
            "dysreg_physical_pain": data.get("dysreg_physical_pain", False),
            "dysreg_dissociation_numbing": data.get("dysreg_dissociation_numbing", False),
            "dysreg_addictive_self_soothing": data.get("dysreg_addictive_self_soothing", False),
            "dysreg_explosive_outbursts": data.get("dysreg_explosive_outbursts", False),
            "dysreg_people_pleasing": data.get("dysreg_people_pleasing", False),
            "dysreg_hyper_control_perfectionism": data.get("dysreg_hyper_control_perfectionism", False),
            "dysreg_parentification_rescuer": data.get("dysreg_parentification_rescuer", False),

            # Family / environment dysregulation (if present)
            "fam_dysreg_physical_pain": data.get("fam_dysreg_physical_pain", False),
            "fam_dysreg_dissociation_numbing": data.get("fam_dysreg_dissociation_numbing", False),
            "fam_dysreg_addictive_self_soothing": data.get("fam_dysreg_addictive_self_soothing", False),
            "fam_dysreg_explosive_outbursts": data.get("fam_dysreg_explosive_outbursts", False),
            "fam_dysreg_people_pleasing": data.get("fam_dysreg_people_pleasing", False),
            "fam_dysreg_hyper_control_perfectionism": data.get("fam_dysreg_hyper_control_perfectionism", False),
            "fam_dysreg_parentification_rescuer": data.get("fam_dysreg_parentification_rescuer", False),

            # Diagnoses
            "dx_autism": data.get("dx_autism", False),
            "dx_adhd": data.get("dx_adhd", False),
            "dx_addictions": data.get("dx_addictions", False),
            "dx_depression": data.get("dx_depression", False),
            "dx_anxiety": data.get("dx_anxiety", False),
            "dx_other": data.get("dx_other", False),
            "dx_other_label": data.get("dx_other_label"),

            # Evidence + notes
            "evidence_snippets": data.get("evidence_snippets", []),
            "notes": data.get("notes"),
        }

        rows.append(row)

    # Sort primarily by book/author then page number (if numeric), otherwise by filename
    def sort_key(r):
        bn = (r.get("book_name") or "")
        au = (r.get("author") or "")
        pn = r.get("page_number")
        # normalize pn if it came as string
        try:
            pn_val = int(pn)
            pn_is_num = True
        except Exception:
            pn_val = None
            pn_is_num = False
        if pn_is_num:
            return (bn, au, 0, pn_val, r.get("source_file") or "")
        return (bn, au, 1, 10**9, r.get("source_file") or "")

    rows.sort(key=sort_key)

    fieldnames = [
        # Core metadata
        "book_name",
        "author",
        "page_number",
        "title",
        "subtitle",
        "text",
        "reference",
        "source_file",
        "narration_voice",

        # Family / environment
        "family_type",

        # Main character dysregulation
        "dysreg_physical_pain",
        "dysreg_dissociation_numbing",
        "dysreg_addictive_self_soothing",
        "dysreg_explosive_outbursts",
        "dysreg_people_pleasing",
        "dysreg_hyper_control_perfectionism",
        "dysreg_parentification_rescuer",

        # Family / environment dysregulation
        "fam_dysreg_physical_pain",
        "fam_dysreg_dissociation_numbing",
        "fam_dysreg_addictive_self_soothing",
        "fam_dysreg_explosive_outbursts",
        "fam_dysreg_people_pleasing",
        "fam_dysreg_hyper_control_perfectionism",
        "fam_dysreg_parentification_rescuer",

        # Diagnoses
        "dx_autism",
        "dx_adhd",
        "dx_addictions",
        "dx_depression",
        "dx_anxiety",
        "dx_other",
        "dx_other_label",

        # Evidence / notes
        "evidence_snippets",
        "notes",
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[{run_ts}] Saved table: {out_csv}")
    print(f"[{run_ts}] Rows written: {len(rows)}")
    if unreadable:
        print(f"[{run_ts}] Skipped unreadable JSONs: {unreadable}")


if __name__ == "__main__":
    main()
