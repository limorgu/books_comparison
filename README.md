# Dysregulation & Family Dynamics in Memoirs

Toolset for analyzing autobiographical / memoir texts — focusing on **emotional regulation patterns**, **family system types**, **trauma indicators**, **neurodivergence markers**, and life-stage differences (childhood vs adulthood).

Currently processes page-level JSON extractions from scanned book photos → produces clean per-page tables + concise per-book summaries.

## Project Structure (as of Feb 2026)
inbox_photos/
├── data_test/
│   ├── pages_clean_.csv                  ← raw per-page extraction
│   ├── books_summary_clean_.csv          ← recommended: clean per-book summary
│   └── tables_out/                        ← all generated CSVs go here
│
├── feb4_bookregulation.py                 ← main script (pages → clean tables)
├── feb4_stage4_booksummary.py             ← (optional) further analysis steps
└── README.md


## Pipeline Steps I Followed

1. **Scan & JSON extraction**  
   Photos of book pages → OCR + LLM prompting → one JSON per page  
   (contains: text, title/subtitle, evidence_snippets, dysreg flags, family_type, etc.)

2. **Clean per-page table** (`pages_clean_YYYYMMDD_HHMMSS.csv`)  
   - Parse book/author from folder names  
   - Normalize booleans, fill page gaps inside known ranges  
   - Forward-fill titles/subtitles per book  
   - Compute consistent `family_type_primary` & `family_type_set` per book  
   - Save detailed row-per-page file

3. **Clean per-book summary** (`books_summary_clean_YYYYMMDD_HHMMSS.csv`)  
   Most useful output for analysis — one row per book with:  

   | Column               | Meaning                                                                 |
   |----------------------|-------------------------------------------------------------------------|
   | author               | Author name                                                            |
   | book_name            | Book title                                                             |
   | n_pages              | Number of processed pages                                              |
   | page_range           | e.g. 90–151                                                            |
   | family_type_primary  | Most consistent family type (or "none")                                |
   | family_type_set      | All reasonably frequent family types (pipe-separated or "none")       |
   | narrator_dysreg      | Active narrator dysregulation patterns (semicolon list or "none")     |
   | family_dysreg        | Active family dysregulation patterns                                   |
   | dx_summary           | Active diagnoses (autism, adhd, depression, … or "none")               |
   | key_snippets         | 1–3 short representative quotes/evidence                               |

   → This is the file I usually open first in Excel / pandas for filtering & adding notes.

4. **(Optional) Life-stage analysis**  
   Earlier versions tried to classify childhood vs adulthood per snippet/page → too noisy  
   → Current recommendation: do this manually or on aggregated book sections only

## How to Run

```bash
# 1. Make sure you're in the project folder
cd ~/yourpath to the project

# 2. Run the main cleaning script
python feb4_bookregulation.py
#   → creates pages_clean_*.csv  AND  books_summary_clean_*.csv
