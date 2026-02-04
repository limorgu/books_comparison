Quick Start (5–10 minutes)

This guide gets you from nothing → analysis tables with minimal context.

1️⃣ Clone & enter the repo
git clone https://github.com/<your-username>/books_judge.git
cd books_judge


Tip: open the folder in VS Code so you can see the pipeline files side-by-side.

2️⃣ Prepare a test book folder

Place at least one book folder inside:

inbox_photos/data_test/


Folder naming convention:

BookTitle_AuthorName/


Example:

AutismInHeels_JeniferCookOtoole/


Each folder should already contain page-level JSON files
(or outputs from your image → JSON extraction step).

3️⃣ Run feature extraction (page → enriched JSON)
python inbox_photos/jan27_stage1_extract_features_to_json.py


What this does:

Reads page JSONs

Extracts dysregulation, family patterns, diagnoses, narrative voice

Writes enriched JSON files (one per page)

✅ Safe to rerun
✅ Does not overwrite raw data

4️⃣ Build the master feature table (JSON → CSV)
python inbox_photos/jan27_stage2_make_json_features_table.py


This creates a page-level CSV where:

Each row = one page

Each column = one feature or metadata field

You should now see CSVs inside:

inbox_photos/data_test/tables_out/

5️⃣ (Recommended) Validate the data

Open the notebook:

jupyter notebook inbox_photos/jan27_stage3notebook_end_end_clean_pipeline.ipynb


Use this step to:

Check page ranges

Spot missing sections

Sanity-check feature distributions

You can skip this step, but it’s strongly recommended before analysis.

6️⃣ Generate analysis tables (book-level insights)
python inbox_photos/feb4_stage5_table_byauthor_bydisregulation.py


This produces:

Book summaries

Titles grouped under books

Main dysregulation features (childhood vs adulthood)

Author-level comparison tables

Outputs are saved to:

inbox_photos/data_test/tables_out/


🎉 You’re done.

✅ What You Should See at the End

Inside tables_out/:

Multiple CSV files

Clearly named analysis tables

Ready for Excel, Pandas, or visualization

If you don’t see new files:

Check the script paths at the top of each .py

Confirm your working directory is the repo root