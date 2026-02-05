#!/usr/bin/env python3
"""
Build clean, focused tables from extracted page JSON files.

Outputs:
  1) pages_clean_<timestamp>.csv          → detailed per-page data
  2) books_summary_clean_<timestamp>.csv  → concise per-book overview (recommended for analysis)

Focus of books_summary_clean:
- author, book_name
- n_pages, page_range
- family_type_primary, family_type_set
- narrator_dysreg, family_dysreg (semicolon-separated or "none")
- dx_summary (semicolon-separated or "none")
- key_snippets (short preview of evidence)
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 90)

# =========================
# CONFIG
# =========================
IN_DIR = Path.home() / "Documents" / "books" / "inbox_photos" / "data_test"
OUT_DIR = IN_DIR / "tables_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PAGES = OUT_DIR / f"pages_clean_{RUN_TS}.csv"
OUT_BOOKS = OUT_DIR / f"books_summary_clean_{RUN_TS}.csv"

# =========================
# FLAG DEFINITIONS
# =========================
NARRATOR_FLAGS = [
    "dysreg_physical_pain",
    "dysreg_dissociation_numbing",
    "dysreg_addictive_self_soothing",
    "dysreg_explosive_outbursts",
    "dysreg_people_pleasing",
    "dysreg_hyper_control_perfectionism",
    "dysreg_parentification_rescuer",
]

FAMILY_FLAGS = [
    f"fam_{flag}" for flag in NARRATOR_FLAGS
]

DX_FLAGS = [
    "dx_autism", "dx_adhd", "dx_addictions",
    "dx_depression", "dx_anxiety", "dx_other",
]

# =========================
# HELPERS
# =========================
def parse_book_author(folder_name: str) -> Tuple[str, str]:
    s = str(folder_name or "").strip()
    if not s:
        return "", ""

    m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", s)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    for sep in [" | ", "__", " — ", " - ", "—", "|"]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if len(parts) >= 2:
                return parts[0], parts[1]

    if "_" in s:
        left, right = s.split("_", 1)
        return left.strip(), right.strip()

    return s, ""


def pick_book_author_from_path(p: Path) -> Tuple[str, str, str]:
    candidates = []
    cur = p.parent
    for _ in range(6):
        if cur is None:
            break
        candidates.append(cur.name)
        cur = cur.parent

    for folder in candidates:
        b, a = parse_book_author(folder)
        if b.strip() and a.strip():
            return b, a, folder

    if candidates:
        b0, a0 = parse_book_author(candidates[0])
        return (b0 or candidates[0]), (a0 or ""), candidates[0]

    return "", "", ""


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    s = str(x).strip().lower()
    return s in ("true", "1", "yes", "y", "t")


def norm_text(x: Any) -> str:
    """Convert almost anything to clean string; return '' on anything empty/NaN"""
    if x is None:
        return ""
    if isinstance(x, (list, tuple, set)):
        # If it's a list of strings → join them
        parts = [norm_text(item) for item in x]  # recurse
        return " ".join(p for p in parts if p)
    if pd.isna(x):           # scalar NaN / None / np.nan
        return ""
    try:
        s = str(x).strip()
        return s if s else ""
    except Exception:
        return ""

def nice_feature_name(s: str) -> str:
    s = s.replace("dysreg_", "").replace("fam_dysreg_", "")
    s = s.replace("_", " ").title()
    return s


def fill_pages_inside_gaps(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["page_number"] = (
        g["page_number"]
        .interpolate(method="linear", limit_area="inside")
        .round()
        .astype("Int64")
    )
    return g


def family_type_primary_consistent(series: pd.Series, n_pages: int) -> str:
    vals = series.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if vals.empty:
        return "none"
    counts = vals.value_counts()
    threshold = max(2, int(np.ceil(0.05 * n_pages)))
    counts = counts[counts >= threshold]
    if counts.empty:
        return "none"
    top = counts.idxmax()
    return top if top else "none"


def family_type_set_consistent(series: pd.Series, n_pages: int) -> str:
    vals = series.dropna().astype(str).str.strip()
    vals = vals[(vals != "") & (vals != "unknown")]
    if vals.empty:
        return "none"
    counts = vals.value_counts()
    threshold = max(2, int(np.ceil(0.05 * n_pages)))
    keep = counts[counts >= threshold].index.tolist()
    return "|".join(sorted(keep)) if keep else "none"


def get_key_snippets(g: pd.DataFrame, max_items: int = 3, max_len: int = 140) -> str:
    if "evidence_snippets" not in g.columns:
        return "none"

    snippets = []
    # .dropna() removes NaN, but we still need to handle lists/strings uniformly
    for item in g["evidence_snippets"].dropna():
        cleaned = norm_text(item)  # now safe for scalar or list
        if cleaned:
            if len(cleaned) > max_len:
                cleaned = cleaned[:max_len-3] + "..."
            snippets.append(cleaned)
            if len(snippets) >= max_items:
                break

    return "; ".join(snippets) if snippets else "none"


# =========================
# MAIN
# =========================
def main():
    print("Scanning directory:", IN_DIR)
    json_files = sorted(IN_DIR.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    if not json_files:
        print("No JSON files found. Exiting.")
        return

    rows = []
    for p in json_files:
        d = safe_read_json(p)
        if d is None:
            continue

        book_name, author, folder_used = pick_book_author_from_path(p)

        row = {
            "author": author,
            "book_name": book_name,
            "book_folder_used": folder_used,
            "json_path": str(p),
            "page_number": d.get("page_number"),
            "title": d.get("title"),
            "subtitle": d.get("subtitle"),
            "text": d.get("text"),
            "reference": d.get("reference"),
            "source_file": d.get("source_file"),
            "evidence_snippets": d.get("evidence_snippets"),
            "notes": d.get("notes"),
        }

        for f in NARRATOR_FLAGS + FAMILY_FLAGS + DX_FLAGS:
            if f in d:
                row[f] = d[f]

        if "family_type" in d:
            row["family_type"] = norm_text(d["family_type"])

        if "dx_other_label" in d:
            row["dx_other_label"] = norm_text(d["dx_other_label"])

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} valid rows")

    # Normalize
    df["page_number"] = pd.to_numeric(df["page_number"], errors="coerce")
    for col in NARRATOR_FLAGS + FAMILY_FLAGS + DX_FLAGS:
        if col in df.columns:
            df[col] = df[col].apply(to_bool)

    # Book / author fallback
    missing = df["author"].isna() | (df["author"].str.strip() == "")
    if missing.any():
        for idx, path_str in df.loc[missing, "json_path"].items():
            b, a, _ = pick_book_author_from_path(Path(path_str))
            if a.strip():
                df.at[idx, "book_name"] = b
                df.at[idx, "author"] = a

    df["book_id"] = (df["book_name"].fillna("") + " | " + df["author"].fillna("")).str.strip()

    # Sort + fill gaps + ffill titles
    df = df.sort_values(["author", "book_name", "page_number", "source_file"]).reset_index(drop=True)
    df = df.groupby(["author", "book_name"], group_keys=False).apply(fill_pages_inside_gaps)
    df = df.sort_values(["author", "book_name", "page_number"]).reset_index(drop=True)

    if "title" in df.columns:
        df["title"] = df.groupby(["author", "book_name"])["title"].ffill()
    if "subtitle" in df.columns:
        df["subtitle"] = df.groupby(["author", "book_name"])["subtitle"].ffill()

    # Family type aggregates
    if "family_type" in df.columns:
        agg_family = (
            df.groupby("book_id", dropna=False)
            .apply(lambda g: pd.Series({
                "family_type_primary": family_type_primary_consistent(g["family_type"], len(g)),
                "family_type_set": family_type_set_consistent(g["family_type"], len(g)),
            }))
            .reset_index()
        )
        df = df.merge(agg_family, on="book_id", how="left")

    # ────────────────────────────────────────────────
    #           CLEAN BOOKS SUMMARY
    # ────────────────────────────────────────────────
    print("\nBuilding clean books summary...")

    summary_rows = []
    for (author, book_name), g in df.groupby(["author", "book_name"], dropna=False):
        # Dysregulation features
        narr_active = [
            nice_feature_name(c)
            for c in NARRATOR_FLAGS
            if c in g.columns and g[c].any()
        ]
        fam_active = [
            nice_feature_name(c)
            for c in FAMILY_FLAGS
            if c in g.columns and g[c].any()
        ]

        # Diagnoses
        dx_active = [
            c.replace("dx_", "").title()
            for c in DX_FLAGS
            if c in g.columns and g[c].any()
        ]

        row = {
            "author": norm_text(author),
            "book_name": norm_text(book_name),
            "n_pages": len(g),
            "page_range": (
                f"{int(g['page_number'].min())}–{int(g['page_number'].max())}"
                if pd.notna(g["page_number"].min()) else "unknown"
            ),
            "family_type_primary": norm_text(g["family_type_primary"].iloc[0]) or "none",
            "family_type_set": norm_text(g["family_type_set"].iloc[0]) or "none",
            "narrator_dysreg": "; ".join(narr_active) if narr_active else "none",
            "family_dysreg": "; ".join(fam_active) if fam_active else "none",
            "dx_summary": "; ".join(dx_active) if dx_active else "none",
            "key_snippets": get_key_snippets(g),
        }
        summary_rows.append(row)

    books_clean = pd.DataFrame(summary_rows)
    books_clean = books_clean.sort_values(["author", "book_name"]).reset_index(drop=True)

    books_clean.to_csv(OUT_BOOKS, index=False)

    # ─── FINAL OUTPUT ───────────────────────────────────────
    print("\n" + "═" * 75)
    print("  PIPELINE COMPLETE")
    print(f"  Pages table:   {OUT_PAGES.name}")
    print(f"  Clean summary: {OUT_BOOKS.name}")
    print("═" * 75)

    print("\nPreview of clean summary (first 10 books):")
    print("─" * 75)
    preview_cols = [
        "author", "book_name", "n_pages", "page_range",
        "family_type_primary", "narrator_dysreg", "family_dysreg", "dx_summary"
    ]
    print(books_clean[preview_cols].head(10).to_string(index=False))

    print("\nDone.\n")


if __name__ == "__main__":
    main()