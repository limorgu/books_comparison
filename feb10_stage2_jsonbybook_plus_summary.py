import json
import os
from pathlib import Path
from collections import defaultdict

def aggregate_book_data(results_folder: str, analysis_output_dir: Path):
    results_path = Path(results_folder)
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    
    books_library = defaultdict(list)

    # 1. Load all individual page JSONs
    json_files = list(results_path.glob("*.json"))
    if not json_files:
        print(f"⚠️ No JSON files found in {results_folder}")
        return

    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Use a safe name for the book (remove spaces/special chars if needed)
                book_title = data.get("meta_data", {}).get("book_name", "Unknown_Book")
                books_library[book_title].append(data)
        except Exception as e:
            print(f"Skipping {file.name}: {e}")

    # 2. Process each book separately
    for book_title, pages in books_library.items():
        # Sort pages numerically
        sorted_pages = sorted(
            pages, 
            key=lambda x: x["meta_data"].get("page_number") if x["meta_data"].get("page_number") is not None else 0
        )
        
        author = sorted_pages[0]["meta_data"].get("author", "Unknown")
        
        book_data = {
            "book_meta": {
                "title": book_title,
                "author": author,
                "total_pages": len(sorted_pages)
            },
            "pages": sorted_pages
        }

        # 3. Save individual book JSON
        # Replace spaces with underscores for the filename
        safe_filename = book_title.replace(" ", "_") + ".json"
        book_file_path = analysis_output_dir / safe_filename
        
        with open(book_file_path, 'w', encoding='utf-8') as f:
            json.dump(book_data, f, ensure_ascii=False, indent=2)
        
        print(f"📖 Created individual book file: {safe_filename}")

    # 4. Optional: Save the Master Library as well
    master_path = analysis_output_dir / "all_books_combined.json"
    with open(master_path, 'w', encoding='utf-8') as f:
        json.dump(books_library, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    base_results_path = Path("/Users/limorkissos/Documents/books/inbox_photos/data_test/Feb_results")
    analysis_output_dir = Path("/Users/limorkissos/Documents/books/inbox_photos/data_test/analysis_outputs")
    
    # Auto-find latest run
    run_folders = sorted([d for d in base_results_path.iterdir() if d.is_dir() and d.name.startswith("run_")])
    
    if run_folders:
        latest_folder = run_folders[-1]
        print(f"📂 Processing: {latest_folder.name}")
        aggregate_book_data(str(latest_folder), analysis_output_dir)