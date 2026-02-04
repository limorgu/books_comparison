🔄 Pipeline Stages
Stage 0 — Raw Inputs

Location: inbox_photos/data_raw/

Book page photos or scans

Untouched extraction outputs

No assumptions, no cleanup

This folder is never edited directly.

Stage 1 — Feature Extraction

Script: jan27_stage1_extract_features_to_json.py
Input: page-level JSONs
Output: enriched JSONs with extracted features

Extracts structured signals such as:

Dysregulation markers (physical pain, dissociation, people-pleasing, etc.)

Family dysregulation patterns

Diagnoses (autism, ADHD, depression, etc.)

Narrative voice & evidence snippets

Each page remains its own JSON for traceability.

Stage 2 — Table Construction

Script: jan27_stage2_make_json_features_table.py
Output: page-level CSV tables

Creates a flat, analyzable table where:

Each row = one page

Columns = extracted features + metadata

Sorting & consistency are enforced

This is the core “analysis-ready” dataset.

Stage 3 — Cleaning & Validation

Notebook: jan27_stage3notebook_end_end_clean_pipeline.ipynb

Used for:

Spot-checking coverage

Verifying page ranges

Fixing inconsistencies

Sanity-checking feature distributions

This is a human-in-the-loop quality gate.

Stage 4 — Thematic Analysis Tables

Script: feb4_stage5_table_byauthor_bydisregulation.py
Output location: data_test/tables_out/

Produces research-level summary tables, including:

Book-level summaries

Titles grouped under books

Main dysregulation features by life stage (childhood vs adulthood)

Author-level comparisons

These tables are meant for:

Comparative analysis

Visualization

Writing & interpretation

📊 Output Tables (tables_out/)

Typical outputs include:

Book → Titles → Features

Book-level dysregulation summary

Author × Dysregulation matrices

Family type distributions

All outputs are CSV and Git-friendly.