#!/usr/bin/env python3
"""
Build clean, organized tables from extracted page JSON files.

Inputs:
  - Any folder containing JSON page files (recursively)

Outputs (CSV):
  1) pages_clean_<timestamp>.csv
     One row per JSON/page. Sorted by author -> book -> page_number -> source_file.
     Includes: author, book_name, book_id, page_number, title, subtitle, reference, source_file, text,
              family_type (+ aggregates), narrator/family dysreg flags, dx flags, dx_other_label, plus metadata.

  2) books_summary_<timestamp>.csv
     One row per (author, book_name) with:
       - n_rows, page_min, page_max
       - family_type_primary, family_type_set
       - narrator/family flags: <flag>_any + <flag>_refs
       - dx flags: <dx>_any + <dx>_refs
       - dx_other_label_set

Assumptions:
  - book_name + author are derived from folder names. Your current convention is typically:
      "BookName_AuthorName"
    but we also support: "Book | Author", "Book__Author", "Book - Author", "Book (Author)", etc.
  - JSONs may contain: page_number, title, subtitle, text, reference, source_file
  - Optional flags may exist:
      narrator dysreg: dysreg_*
      family dysreg: fam_dysreg_*
      dx: dx_*
      family_type, dx_other_label
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

# =========================
# CONFIG (edit paths)
# =========================
IN_DIR = Path.home() / "Documents" / "books" / "inbox_photos" / "data_test"  # folder with JSONs (recursive)
OUT_DIR = IN_DIR / "tables_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PAGES = OUT_DIR / f"pages_clean_{RUN_TS}.csv"
OUT_BOOKS = OUT_DIR / f"books_summary_{RUN_TS}.csv"

# =========================
# FLAG LISTS
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
    "fam_dysreg_physical_pain",
    "fam_dysreg_dissociation_numbing",
    "fam_dysreg_addictive_self_soothing",
    "fam_dysreg_explosive_outbursts",
    "fam_dysreg_people_pleasing",
    "fam_dysreg_hyper_control_perfectionism",
    "fam_dysreg_parentification_rescuer",
]

DX_FLAGS = [
    "dx_autism",
    "dx_adhd",
    "dx_addictions",
    "dx_depression",
    "dx_anxiety",
    "dx_other",
]

# =========================
# HELPERS
# =========================
def parse_book_author(folder_name: str) -> Tuple[str, str]:
    """
    Parse book + author from folder name.

    Supports:
      - "Book Name | Author Name"
      - "Book Name__Author Name"
      - "Book Name - Author Name" / "Book Name — Author Name"
      - "BookName_AuthorName" (your convention; we split on FIRST underscore)
      - "Family Lexicon (Natalia Ginzburg)"

    If no separator found, returns (folder_name, "").
    """
    s = str(folder_name or "").strip()
    if not s:
        return "", ""

    # Parentheses form: "Book (Author)"
    m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", s)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # Explicit separators
    for sep in [" | ", "__", " — ", " - ", "—", "|"]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if len(parts) >= 2:
                return parts[0], parts[1]

    # Your current convention: split on FIRST underscore
    if "_" in s:
        left, right = s.split("_", 1)
        book = left.strip()
        author = right.strip()
        if book and author:
            return book, author

    return s, ""


def pick_book_author_from_path(p: Path) -> Tuple[str, str, str]:
    """
    Robustly pick (book_name, author, folder_used) by trying a few folder levels up.
    Handles nesting like:
      data_test/Book_Author/json/page_001.json
      data_test/Book_Author/page_001.json
    """
    candidates: List[str] = []
    cur = p.parent
    for _ in range(5):
        if cur is None:
            break
        candidates.append(cur.name)
        cur = cur.parent

    for folder in candidates:
        b, a = parse_book_author(folder)
        if b.strip() and a.strip():
            return b, a, folder

    # fallback: use first candidate as book, blank author
    if candidates:
        b0, a0 = parse_book_author(candidates[0])
        return (b0 or candidates[0]), (a0 or ""), candidates[0]

    return "", "", ""


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed reading JSON: {path} ({e})")
        return None


def to_bool(x: Any) -> bool:
    """Normalize booleans that might be stored as strings."""
    if isinstance(x, bool):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    s = str(x).strip().lower()
    return s in ("true", "1", "yes", "y", "t")


def norm_family_type(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    return s if s else None


def fill_pages_inside_gaps(g: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN page_number only inside known gaps (no guessing at start/end)."""
    g = g.copy()
    g["page_number_original"] = g["page_number"]
    g["page_number"] = (
        g["page_number"]
        .interpolate(method="linear", limit_area="inside")
        .round()
        .astype("Int64")
    )
    return g


def refs_where_true(g: pd.DataFrame, flag_col: str) -> str:
    """
    Pipe-join references for rows where flag_col is True.
    Uses `reference` if present, otherwise falls back to `source_file`.
    """
    if flag_col not in g.columns:
        return ""

    ref_col = "reference" if "reference" in g.columns else ("source_file" if "source_file" in g.columns else None)
    if ref_col is None:
        return ""

    refs = g.loc[g[flag_col] == True, ref_col].dropna().astype(str).tolist()
    seen = set()
    out: List[str] = []
    for r in refs:
        if r not in seen:
            out.append(r)
            seen.add(r)
    return "|".join(out)


def family_type_set_consistent(series: pd.Series, n_pages: int, min_count: int = 2, min_pct: float = 0.05) -> str:
    """
    Keep only family_type labels that are consistent across the book.
    A label is kept if it appears at least max(min_count, ceil(min_pct*n_pages)) times.
    """
    vals = [v for v in series.dropna().astype(str).tolist() if v and v != "unknown"]
    if not vals:
        return ""
    counts = pd.Series(vals).value_counts()
    threshold = max(min_count, int(np.ceil(min_pct * max(n_pages, 1))))
    keep = sorted(counts[counts >= threshold].index.tolist())
    return "|".join(keep) if keep else ""


def family_type_primary_consistent(series: pd.Series, n_pages: int, min_count: int = 2, min_pct: float = 0.05) -> str:
    """Pick the most frequent *consistent* family_type label; else 'unknown'."""
    vals = [v for v in series.dropna().astype(str).tolist() if v and v != "unknown"]
    if not vals:
        return "unknown"

    counts = pd.Series(vals).value_counts()
    threshold = max(min_count, int(np.ceil(min_pct * max(n_pages, 1))))
    counts = counts[counts >= threshold]
    if counts.empty:
        return "unknown"

    top_count = counts.max()
    top_labels = counts[counts == top_count].index.tolist()
    return sorted(top_labels)[0]  # stable tie-break


def latest_matching(out_dir: Path, pattern: str) -> Optional[Path]:
    files = sorted(out_dir.glob(pattern))
    return files[-1] if files else None


# =========================
# MAIN
# =========================
def main() -> None:
    print("IN_DIR :", IN_DIR)
    print("OUT_DIR:", OUT_DIR)
    print("OUT_PAGES:", OUT_PAGES.name)
    print("OUT_BOOKS:", OUT_BOOKS.name)

    # 1) Collect JSON files
    json_files = sorted(IN_DIR.rglob("*.json"))
    print("\n[1/6] Found JSON files:", len(json_files))
    if not json_files:
        raise FileNotFoundError(f"No .json files found under: {IN_DIR}")

    # 2) Read JSONs into rows
    rows: List[Dict[str, Any]] = []
    bad = 0

    for i, p in enumerate(json_files, start=1):
        d = safe_read_json(p)
        if d is None:
            bad += 1
            continue

        book_name, author, folder_used = pick_book_author_from_path(p)

        row: Dict[str, Any] = {
            "author": author,
            "book_name": book_name,
            "book_folder": folder_used,
            "json_path": str(p),
        }

        # core fields
        for k in ["page_number", "title", "subtitle", "text", "reference", "source_file"]:
            if k in d:
                row[k] = d.get(k)

        # family_type + dx_other_label
        if "family_type" in d:
            row["family_type"] = d.get("family_type")
        if "dx_other_label" in d:
            row["dx_other_label"] = d.get("dx_other_label")

        # narrator/family flags
        for f in (NARRATOR_FLAGS + FAMILY_FLAGS):
            if f in d:
                row[f] = d.get(f)

        # dx flags
        for dx in DX_FLAGS:
            if dx in d:
                row[dx] = d.get(dx)

        # optional metadata
        for k in ["chapter", "section", "language", "narration_voice", "notes", "evidence_snippets"]:
            if k in d and k not in row:
                row[k] = d.get(k)

        rows.append(row)

        if i <= 5:
            print(f"[CHECK] {p.name} -> folder='{folder_used}' -> book='{book_name}' author='{author}'")
        if i % 500 == 0:
            print(f"  - loaded {i}/{len(json_files)} JSONs...")

    print("[2/6] Loaded rows:", len(rows), " | failed JSONs:", bad)
    df = pd.DataFrame(rows)

    # 3) Normalize
    for col in ["book_name", "author", "title", "subtitle", "reference", "source_file", "text", "family_type", "dx_other_label"]:
        if col in df.columns:
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    df["page_number"] = pd.to_numeric(df.get("page_number", np.nan), errors="coerce")

    present_narr = [c for c in NARRATOR_FLAGS if c in df.columns]
    present_fam = [c for c in FAMILY_FLAGS if c in df.columns]
    present_dx = [c for c in DX_FLAGS if c in df.columns]
    for c in (present_narr + present_fam + present_dx):
        df[c] = df[c].apply(to_bool)

    if "family_type" in df.columns:
        df["family_type"] = df["family_type"].apply(norm_family_type)

    # Backfill missing author/book from folder path (long-term fix)
    df["author"] = df.get("author", np.nan)
    df["book_name"] = df.get("book_name", np.nan)
    df["book_folder_used"] = np.nan

    missing_author = df["author"].isna() | (df["author"].astype(str).str.strip() == "")
    if missing_author.any():
        fixed = 0
        for idx, path_str in df.loc[missing_author, "json_path"].items():
            b, a, folder_used = pick_book_author_from_path(Path(path_str))
            if a.strip():
                df.at[idx, "book_name"] = b
                df.at[idx, "author"] = a
                df.at[idx, "book_folder_used"] = folder_used
                fixed += 1
        print(f"[BOOK/AUTHOR FIX] Filled missing author/book for {fixed} rows using folder names.")

    # book_id after backfill
    df["book_id"] = (
        df["book_name"].fillna("").astype(str).str.strip()
        + " | "
        + df["author"].fillna("").astype(str).str.strip()
    )

    print("[3/6] Distinct books:", df[["author", "book_name"]].drop_duplicates().shape[0])
    print("[3/6] Missing author (NA or blank):", int((df["author"].isna() | (df["author"].astype(str).str.strip() == "")).sum()))
    print("[3/6] Missing page_number:", int(df["page_number"].isna().sum()))

    # 4) Sort + fill page gaps + ffill titles/subtitles
    if "source_file" not in df.columns:
        df["source_file"] = np.nan

    sort_cols = ["author", "book_name", "page_number", "source_file"]
    df = df.sort_values(by=sort_cols, na_position="last").reset_index(drop=True)

    df = (
        df.groupby(["author", "book_name"], dropna=False, group_keys=False)
          .apply(fill_pages_inside_gaps)
    )
    df = df.sort_values(by=sort_cols, na_position="last").reset_index(drop=True)

    if "title" in df.columns:
        df["title"] = df.groupby(["author", "book_name"], dropna=False)["title"].ffill()
    if "subtitle" in df.columns:
        df["subtitle"] = df.groupby(["author", "book_name"], dropna=False)["subtitle"].ffill()

    # family_type consistency aggregates (prevents 1-off “alcoholic” pollution)
    if "family_type" in df.columns:
        # quick audit
        print("\n[FAMILY_TYPE AUDIT] Raw counts per book_id (top 50 rows):")
        tmp_raw = (
            df.dropna(subset=["family_type"])
              .query("family_type != 'unknown'")
              .groupby(["book_id", "family_type"], dropna=False)
              .size()
              .reset_index(name="count")
              .sort_values(["book_id", "count"], ascending=[True, False])
        )
        print(tmp_raw.head(50).to_string(index=False))

        family_agg = (
            df.groupby(["book_id"], dropna=False)
              .apply(lambda g: pd.Series({
                  "family_type_primary": family_type_primary_consistent(g["family_type"], n_pages=len(g)),
                  "family_type_set": family_type_set_consistent(g["family_type"], n_pages=len(g)),
              }))
              .reset_index()
        )
        df = df.merge(family_agg, on="book_id", how="left")
    else:
        df["family_type_primary"] = np.nan
        df["family_type_set"] = np.nan

    # 5) Build book summary (any + refs)
    print("[5/6] Building books summary (flags + refs)...")
    flag_groups = present_narr + present_fam + present_dx

    summary_rows: List[Dict[str, Any]] = []
    for (author, book_name), g in df.groupby(["author", "book_name"], dropna=False):
        book_id = (
            (str(book_name).strip() if book_name is not None else "")
            + " | "
            + (str(author).strip() if author is not None else "")
        )

        row: Dict[str, Any] = {
            "author": author,
            "book_name": book_name,
            "book_id": book_id,
            "n_rows": int(len(g)),
            "page_min": int(g["page_number"].min()) if pd.notna(g["page_number"].min()) else None,
            "page_max": int(g["page_number"].max()) if pd.notna(g["page_number"].max()) else None,
        }

        # family_type aggregates
        if "family_type_primary" in g.columns:
            row["family_type_primary"] = g["family_type_primary"].iloc[0]
        if "family_type_set" in g.columns:
            row["family_type_set"] = g["family_type_set"].iloc[0]

        # any + refs for flags
        for f in flag_groups:
            row[f"{f}_any"] = bool(g[f].any())
            row[f"{f}_refs"] = refs_where_true(g, f)

        # dx_other_label_set
        if "dx_other_label" in g.columns:
            labels = g["dx_other_label"].dropna().astype(str).str.strip()
            labels = [x for x in labels.tolist() if x and x.lower() != "nan"]
            row["dx_other_label_set"] = "|".join(sorted(set(labels))) if labels else ""

        summary_rows.append(row)

    books_df = pd.DataFrame(summary_rows).sort_values(by=["author", "book_name"]).reset_index(drop=True)
    print("[5/6] Books summary rows:", len(books_df))
    if len(books_df) > 0:
        preview_cols = ["author", "book_name", "n_rows", "page_min", "page_max", "family_type_primary", "family_type_set"]
        preview_cols = [c for c in preview_cols if c in books_df.columns]
        print(books_df[preview_cols].head(10).to_string(index=False))

    # 6) Save outputs + print filenames clearly + verify by reread
    print("[6/6] Saving CSV outputs...")
    df.to_csv(OUT_PAGES, index=False)
    books_df.to_csv(OUT_BOOKS, index=False)

    print("\n✅ CLEANED FILES SAVED")
    print("Pages table saved to:\n ", OUT_PAGES.resolve())
    print("Books summary saved to:\n ", OUT_BOOKS.resolve())

    # Print most recent matching files (should be exactly these)
    latest_pages = latest_matching(OUT_DIR, "pages_clean_*.csv")
    latest_books = latest_matching(OUT_DIR, "books_summary_*.csv")
    if latest_pages:
        print("\nMost recent pages file :", latest_pages.name)
    if latest_books:
        print("Most recent books file :", latest_books.name)

    # Verify by re-reading exactly what we wrote
    pages_check = pd.read_csv(OUT_PAGES)
    books_check = pd.read_csv(OUT_BOOKS)

    missing_author_cnt = int((pages_check["author"].isna() | (pages_check["author"].astype(str).str.strip() == "")).sum())
    missing_page_cnt = int(pd.to_numeric(pages_check["page_number"], errors="coerce").isna().sum())

    print("\n[CHECK] Re-read from disk:")
    print(" pages rows:", len(pages_check), "| cols:", len(pages_check.columns))
    print(" books rows:", len(books_check), "| cols:", len(books_check.columns))
    print(" pages missing author (NA or blank):", missing_author_cnt)
    print(" pages missing page_number:", missing_page_cnt)

    print("\nPIPELINE FINISHED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
