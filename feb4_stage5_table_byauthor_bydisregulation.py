#!/usr/bin/env python3
"""
Analyze memoir-extraction table and produce 2 new CSVs:

1) Book summary with ALL titles (subtitles) under each book, with features
   -> One row per (author, book_name, subtitle)

2) Book-level table:
   -> book_name, author, family_type, main dysregulation feature in childhood, main dysreg feature in adulthood

Inputs:
  - A CSV  pages_clean_*.csv 

Outputs:
  - analysis_book_titles_features_<timestamp>.csv
  - analysis_books_lifestage_main_dysreg_<timestamp>.csv
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import pandas as pd


# =========================
# CONFIG — EDIT THESE PATHS
# =========================
IN_CSV = Path.home() / "Documents" / "books" / "inbox_photos" / "data_test" / "pages_clean_latest.csv"
OUT_DIR = Path.home() / "Documents" / "books" / "inbox_photos" / "data_test" / "analysis_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================
# COLUMN DEFINITIONS
# =========================
KEY_COLS = [
    "book_name", "author", "book_id",
    "page_number", "page_number_original",
    "subtitle", "subtitle_original",
    "text", "reference", "source_file",
    "narration_voice",
    "family_type", "family_type_primary", "family_type_set",
    "evidence_snippets", "notes",
]

# Narrator dysreg features (main character)
DYSREG_COLS = [
    "dysreg_physical_pain",
    "dysreg_dissociation_numbing",
    "dysreg_addictive_self_soothing",
    "dysreg_explosive_outbursts",
    "dysreg_people_pleasing",
    "dysreg_hyper_control_perfectionism",
    "dysreg_parentification_rescuer",
]

# Family dysreg features
FAM_DYSREG_COLS = [
    "fam_dysreg_physical_pain",
    "fam_dysreg_dissociation_numbing",
    "fam_dysreg_addictive_self_soothing",
    "fam_dysreg_explosive_outbursts",
    "fam_dysreg_people_pleasing",
    "fam_dysreg_hyper_control_perfectionism",
    "fam_dysreg_parentification_rescuer",
]

DX_COLS = [
    "dx_autism",
    "dx_adhd",
    "dx_addictions",
    "dx_depression",
    "dx_anxiety",
    "dx_other",
]
DX_OTHER_LABEL_COL = "dx_other_label"


# =========================
# HELPERS
# =========================
def _to_bool_series(s: pd.Series) -> pd.Series:
    """
    Convert common encodings (1/0, True/False, "true"/"false", "yes"/"no") to boolean.
    Missing -> False.
    """
    if s is None:
        return pd.Series(dtype=bool)

    # Fast path if already bool
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    # Numeric path
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)

    # String path
    s2 = s.fillna("").astype(str).str.strip().str.lower()
    truthy = {"1", "true", "t", "yes", "y"}
    falsy = {"0", "false", "f", "no", "n", ""}
    return s2.apply(lambda x: True if x in truthy else (False if x in falsy else False))


def safe_text(x) -> str:
    return "" if pd.isna(x) else str(x)


def classify_lifestage(subtitle: str, text: str) -> str:
    """
    Heuristic classifier: childhood / adulthood / unknown.
    Uses keywords in subtitle + text. You can expand these lists anytime.
    """
    blob = (safe_text(subtitle) + " " + safe_text(text)).lower()

    childhood_patterns = [
        r"\bchildhood\b", r"\bas a child\b", r"\bwhen i was (?:a )?kid\b",
        r"\bwhen i was (?:\d{1,2})\b", r"\bat (?:age|ages) \d{1,2}\b",
        r"\bmy mother\b", r"\bmy father\b", r"\bgrade\b", r"\bschool\b",
        r"\bteen\b", r"\bteenage\b", r"\bhigh school\b", r"\belementary\b",
        r"\bkindergarten\b", r"\bmiddle school\b",
    ]
    adulthood_patterns = [
        r"\bas an adult\b", r"\bnow\b", r"\btoday\b", r"\bthese days\b",
        r"\bmarriage\b", r"\bhusband\b", r"\bwife\b", r"\bpartner\b",
        r"\bmy kids\b", r"\bmy child\b", r"\bmy children\b",
        r"\bwork\b", r"\bcareer\b", r"\btherap(y|ist)\b",
        r"\bin my (?:twenties|30s|thirties|40s|forties|50s|fifties)\b",
        r"\bafter college\b", r"\badulthood\b",
    ]

    c_hits = sum(bool(re.search(p, blob)) for p in childhood_patterns)
    a_hits = sum(bool(re.search(p, blob)) for p in adulthood_patterns)

    if c_hits == 0 and a_hits == 0:
        return "unknown"
    if c_hits > a_hits:
        return "childhood"
    if a_hits > c_hits:
        return "adulthood"
    # tie-break: keep unknown rather than guessing
    return "unknown"


def pick_top_feature(df: pd.DataFrame, feature_cols: list[str]) -> str:
    """
    Return the feature col name with the highest sum of True values.
    If all sums are 0 -> "".
    """
    if df.empty:
        return ""
    sums = {c: int(df[c].sum()) for c in feature_cols if c in df.columns}
    if not sums:
        return ""
    best_col, best_val = max(sums.items(), key=lambda kv: kv[1])
    return best_col if best_val > 0 else ""


def nice_feature_name(col: str) -> str:
    """
    Convert 'dysreg_people_pleasing' -> 'people_pleasing'
    """
    if not col:
        return ""
    return col.replace("dysreg_", "").replace("fam_dysreg_", "")


def agg_unique_semicolon(values: pd.Series, max_items: int = 60) -> str:
    items = [safe_text(x).strip() for x in values.dropna().tolist()]
    items = [x for x in items if x]
    # unique but preserve order
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
        if len(out) >= max_items:
            break
    return "; ".join(out)


# =========================
# MAIN
# =========================
def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(
            f"IN_CSV not found:\n  {IN_CSV}\n\nEdit IN_CSV at top of script to your real pages_clean CSV."
        )

    df = pd.read_csv(IN_CSV)

    # Ensure essential columns exist (don’t crash if some are missing; just proceed)
    for col in KEY_COLS + DYSREG_COLS + FAM_DYSREG_COLS + DX_COLS + [DX_OTHER_LABEL_COL]:
        if col not in df.columns:
            df[col] = pd.NA

    # Normalize booleans
    for col in DYSREG_COLS + FAM_DYSREG_COLS + DX_COLS:
        df[col] = _to_bool_series(df[col])

    # Basic cleanup
    df["subtitle"] = df["subtitle"].fillna("").astype(str).str.strip()
    df["author"] = df["author"].fillna("").astype(str).str.strip()
    df["book_name"] = df["book_name"].fillna("").astype(str).str.strip()
    df["family_type_primary"] = df["family_type_primary"].fillna("").astype(str).str.strip()

    # Create lifestage label
    df["life_stage_guess"] = df.apply(lambda r: classify_lifestage(r["subtitle"], r["text"]), axis=1)

    # =========================
    # (1) TABLE: titles under each book with features (row per subtitle)
    # =========================
    group_cols = ["author", "book_name", "book_id", "subtitle"]
    # If your subtitles are often empty, keep them but label as "(no subtitle)"
    df["subtitle_norm"] = df["subtitle"].where(df["subtitle"] != "", "(no subtitle)")
    group_cols = ["author", "book_name", "book_id", "subtitle_norm"]

    agg_dict = {
        "page_number": ["min", "max", "count"],
        "family_type_primary": "first",
        "family_type_set": "first",
        "narration_voice": lambda s: agg_unique_semicolon(s, max_items=8),
        "reference": lambda s: agg_unique_semicolon(s, max_items=15),
        "source_file": lambda s: agg_unique_semicolon(s, max_items=25),
        "evidence_snippets": lambda s: agg_unique_semicolon(s, max_items=25),
        "notes": lambda s: agg_unique_semicolon(s, max_items=25),
        "life_stage_guess": lambda s: agg_unique_semicolon(s, max_items=5),
        DX_OTHER_LABEL_COL: lambda s: agg_unique_semicolon(s, max_items=15),
    }

    # Add boolean “any” aggregations
    for col in DYSREG_COLS + FAM_DYSREG_COLS + DX_COLS:
        agg_dict[col] = "max"  # any True in the group

    titles_tbl = (
        df.groupby(group_cols, dropna=False)
          .agg(agg_dict)
          .reset_index()
    )

    # Flatten multiindex columns created by page_number aggregations
    titles_tbl.columns = [
        "_".join([c for c in col if c]).rstrip("_") if isinstance(col, tuple) else col
        for col in titles_tbl.columns
    ]

    # Rename for clarity
    titles_tbl = titles_tbl.rename(columns={
        "subtitle_norm": "subtitle",
        "page_number_min": "page_min",
        "page_number_max": "page_max",
        "page_number_count": "n_pages_rows",
    })

    # Sort nicely
    titles_tbl = titles_tbl.sort_values(["author", "book_name", "page_min", "subtitle"], kind="mergesort")

    out_titles = OUT_DIR / f"analysis_book_titles_features_{RUN_TS}.csv"
    titles_tbl.to_csv(out_titles, index=False)

    # =========================
    # (2) TABLE: book-level main dysreg feature in childhood vs adulthood
    # =========================
    book_rows = []
    for (author, book_name, book_id), g in df.groupby(["author", "book_name", "book_id"], dropna=False):
        family_type = safe_text(g["family_type_primary"].dropna().iloc[0] if g["family_type_primary"].notna().any() else "")

        g_child = g[g["life_stage_guess"] == "childhood"]
        g_adult = g[g["life_stage_guess"] == "adulthood"]

        top_child = nice_feature_name(pick_top_feature(g_child, DYSREG_COLS))
        top_adult = nice_feature_name(pick_top_feature(g_adult, DYSREG_COLS))

        # Optional: keep counts to help debug the heuristic
        child_rows = int(len(g_child))
        adult_rows = int(len(g_adult))
        unknown_rows = int((g["life_stage_guess"] == "unknown").sum())

        book_rows.append({
            "author": author,
            "book_name": book_name,
            "book_id": book_id,
            "family_type_primary": family_type,
            "main_dysreg_childhood": top_child,
            "main_dysreg_adulthood": top_adult,
            "rows_childhood": child_rows,
            "rows_adulthood": adult_rows,
            "rows_unknown": unknown_rows,
        })

    life_tbl = pd.DataFrame(book_rows).sort_values(["author", "book_name"], kind="mergesort")

    out_life = OUT_DIR / f"analysis_books_lifestage_main_dysreg_{RUN_TS}.csv"
    life_tbl.to_csv(out_life, index=False)

    print("\nDONE ✅")
    print(f"Input:      {IN_CSV}")
    print(f"Output #1:  {out_titles}")
    print(f"Output #2:  {out_life}")


if __name__ == "__main__":
    main()
