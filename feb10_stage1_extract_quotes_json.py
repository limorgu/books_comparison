import os
import json
import io
import base64
import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from PIL import Image
from openai import OpenAI

# ---------------------------
# config
# ---------------------------
ROOT_DIR_DEFAULT = Path("/Users/limorkissos/Documents/books/inbox_photos/data_test/Feb_books_test")
OUT_SUFFIX = ".json"
MODEL = "gpt-4o-mini" # You can use "gpt-4o" for even higher extraction accuracy
FOLDER_SEP = "_"
LIMIT_TEST = 10 

# ---------------------------
# helpers
# ---------------------------
def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "": return None
        return int(x)
    except Exception: return None

def image_to_data_url_under_5mb(path: Path, max_bytes: int = 5 * 1024 * 1024) -> str:
    img = Image.open(path).convert("RGB")
    def encode(im: Image.Image, q: int) -> bytes:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q, optimize=True)
        return buf.getvalue()
    quality = 90
    data = encode(img, quality)
    while len(data) > max_bytes and quality > 55:
        quality -= 7
        data = encode(img, quality)
    while len(data) > max_bytes:
        w, h = img.size
        img = img.resize((int(w * 0.85), int(h * 0.85)))
        data = encode(img, min(85, quality))
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def parse_book_author_from_folder(book_folder: Path) -> Tuple[str, str]:
    name = book_folder.name.strip()
    parts = [p for p in name.split(FOLDER_SEP) if p]
    if len(parts) >= 2:
        author_part = parts[-1]
        book_part = "_".join(parts[:-1])
        return book_part.replace("_", " "), author_part.replace("_", " ")
    return name.replace("_", " "), ""

# ---------------------------
# Stage 1: Extraction & Analysis
# ---------------------------
def stage_1_extraction(client: OpenAI, image_path: Path, book_name: str, author: str) -> Dict[str, Any]:
    # This schema follows your requirements exactly
    schema = {
        "type": "object",
        "properties": {
            "meta_data": {
                "type": "object",
                "properties": {
                    "book_name": {"type": "string"},
                    "page_number": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "title": {"type": "string"},
                    "subtitle": {"type": "string"}
                },
                "required": ["book_name", "page_number", "title", "subtitle"],
                "additionalProperties": False
            },
            "narrator_voice": {
                "type": "string",
                "enum": ["child", "adult", "adolescence", "unknown"]
            },
            "parent_diagnosis": {
                "type": "object",
                "properties": {
                    "exists": {"type": "boolean"},
                    "diagnosis_name": {"type": "string"},
                    "quotes": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["exists", "diagnosis_name", "quotes"],
                "additionalProperties": False
            },
            "family_type": {
                "type": "object",
                "properties": {
                    "exists": {"type": "boolean"},
                    "label": {"type": "string", "description": "e.g., divorced, mental illness, drunk father, poor"},
                    "quotes": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["exists", "label", "quotes"],
                "additionalProperties": False
            },
            "child_diagnosis": {
                "type": "object",
                "properties": {
                    "exists": {"type": "boolean"},
                    "diagnosis_name": {"type": "string"},
                    "quotes": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["exists", "diagnosis_name", "quotes"],
                "additionalProperties": False
            },
            "physical_body_sensation": {
                "type": "object",
                "properties": {
                    "exists": {"type": "boolean"},
                    "quotes": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["exists", "quotes"],
                "additionalProperties": False
            },
            "child_internal_monologue": {
                "type": "object",
                "description": "What the child tells herself about sensations, emotions, or thoughts",
                "properties": {
                    "exists": {"type": "boolean"},
                    "quotes": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["exists", "quotes"],
                "additionalProperties": False
            },
            "parent_to_child_feedback": {
                "type": "object",
                "description": "What the parent tells the child about sensations/emotions",
                "properties": {
                    "exists": {"type": "boolean"},
                    "quotes": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["exists", "quotes"],
                "additionalProperties": False
            }
        },
        "required": [
            "meta_data", "narrator_voice", "parent_diagnosis", "family_type", 
            "child_diagnosis", "physical_body_sensation", 
            "child_internal_monologue", "parent_to_child_feedback"
        ],
        "additionalProperties": False
    }

    prompt = (
        f"Analyze this page from the book '{book_name}'.\n"
        "1. Extract Meta-data (page number, title, subtitle).\n"
        "2. Identify the Narrator Voice (child, adult, adolescence, or unknown).\n"
        "3. Look for Parent or Child Diagnoses (e.g., ADHD, Depression, Anxiety).\n"
        "4. Identify Family Type labels (e.g., divorced, mental illness, drunk father, poor).\n"
        "5. Extract Physical Body Sensations (e.g., 'tummy ache', 'shaking').\n"
        "6. Capture Internal Monologue: What the child tells themselves about their feelings.\n"
        "7. Capture Parent Feedback: What the parent says to the child about the child's body or emotions.\n\n"
        "For all items, if it exists, set 'exists' to true and provide exact verbatim quotes."
    )

    data_url = image_to_data_url_under_5mb(image_path)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        response_format={"type": "json_schema", "json_schema": {"name": "book_analysis", "strict": True, "schema": schema}},
    )

    return json.loads(resp.choices[0].message.content)

# ---------------------------
# main runner
# ---------------------------
def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    root_dir = ROOT_DIR_DEFAULT.expanduser()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_base = Path("/Users/limorkissos/Documents/books/inbox_photos/data_test/Feb_results")
    output_folder = output_base / f"run_{timestamp}"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    client = OpenAI()

    images: List[Path] = []
    book_dirs = [p for p in sorted(root_dir.iterdir()) if p.is_dir()]
    for book_dir in book_dirs:
        for p in sorted(book_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                images.append(p)

    processed = 0
    for img in images:
        if LIMIT_TEST is not None and processed >= LIMIT_TEST:
            print(f"\n✋ Test limit reached ({LIMIT_TEST}).")
            break

        out_path = output_folder / (img.name + OUT_SUFFIX)
        book_name, author = parse_book_author_from_folder(img.parent)

        try:
            # Single Stage 1 call now handles everything
            result = stage_1_extraction(client, img, book_name, author)
            
            # Ensure the book name from folder is injected into metadata
            result["meta_data"]["book_name"] = book_name
            result["meta_data"]["author"] = author
            
            # Save the file
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            processed += 1
            print(f"✅ Processed: {img.name}")
        except Exception as e:
            print(f"❌ Error on {img.name}: {e}")

    print(f"\nDone. {processed} JSONs created in: {output_folder}")

if __name__ == "__main__":
    main()