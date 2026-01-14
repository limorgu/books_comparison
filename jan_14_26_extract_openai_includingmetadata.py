"""
Extract book page fields from images under ~/Documents/books/inbox_photos
(folder-per-book), using OpenAI vision + Structured Outputs, and save a sidecar
JSON per image next to the image.

Folder convention (recommended):
  ~/Documents/books/inbox_photos/<book_name>__<author_name>/*.jpg

Examples:
  inbox_photos/Its_Just_Your_Imagination__Revital_Shiri-Horowitz/PXL_....jpg
  inbox_photos/What_Looks_Like_Bravery__Laurel_Braitman/PXL_....jpg

Output sidecar:
  image.jpg.json  (stored next to the image)

Requires:
  pip install openai pillow
Env:
  export OPENAI_API_KEY="..."
"""

import os
import json
import io
import base64
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from xmlrpc import client

from PIL import Image
from openai import OpenAI

ROOT_DIR = Path.home() / "Documents" / "books" / "inbox_photos"
OUT_SUFFIX = ".json"
MODEL = "gpt-4o-mini"  # fast + cheap + supports vision + structured output

# If your folder names use a different separator, change this:
FOLDER_SEP = "__"


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def image_to_data_url_under_5mb(path: Path, max_bytes: int = 5 * 1024 * 1024) -> str:
    """Return a data URL (jpeg) resized/compressed to stay <= 5MB."""
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
        if img.size[0] < 400 or img.size[1] < 400:
            break

    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def parse_book_author_from_folder(image_path: Path) -> Dict[str, str]:
    """
    Expected structure:
      .../data/<book_folder>/<image>.jpg
    Example folder:
      It's_just_your_imagination_revital_shiri_horowitz
    """
    book_folder = image_path.parent.name

    # normalize separators
    parts = book_folder.replace("-", "_").split("_")
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) >= 4:
        author = " ".join(parts[-3:])       # last 3 tokens
        book_name = " ".join(parts[:-3])    # the rest
    else:
        author = ""
        book_name = " ".join(parts)

    # IMPORTANT: don't .title() everything (it can mess apostrophes + capitalization)
    book_name = " ".join(book_name.split()).strip()
    author = " ".join(author.split()).strip()

    return {
        "book_name": book_name,
        "author": author if author else None
    }
def extract_page_number_from_bottom_crop(client: OpenAI, image_path: Path) -> Optional[int]:
    """
    If the main extraction missed the page number, do a second pass:
    crop the bottom of the page where page numbers usually appear.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # bottom 18% of the page (tweak if needed: 0.15–0.25)
    crop = img.crop((0, int(h * 0.82), w, h))

    # Make it easier to read: upscale
    crop = crop.resize((w * 2, int(h * 0.18) * 2))

    # Encode crop as data URL
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=90, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "page_number": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        },
        "required": ["page_number"],
    }

    resp = client.responses.create(
        model=MODEL,
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Look ONLY at this cropped image (bottom of the page). "
                        "If you see a page number, return it as an integer. "
                        "If none is visible, return null. Do not guess."
                    ),
                },
                {"type": "input_image", "image_url": data_url},
            ],
        }],
        text={"format": {"type": "json_schema", "name": "page_num_only", "strict": True, "schema": schema}},
    )

    parsed = json.loads(resp.output_text)
    return _safe_int(parsed.get("page_number"))


def extract_one_image(client: OpenAI, image_path: Path, book_name: str, author: str) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "page_number": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "text": {"type": "string"},
            "life_stage_flag": {
                "type": "string",
                "enum": ["childhood", "adulthood", "both", "unclear"]
            },
        },
        "required": ["page_number", "text", "life_stage_flag"],
    }

    data_url = image_to_data_url_under_5mb(image_path)

    resp = client.responses.create(
        model=MODEL,
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Extract the visible page text accurately.\n"
                        "- page_number: only if visible; else null\n"
                        "- text: exact text with paragraph breaks as \\n\n"
                        "- life_stage_flag: childhood/adulthood/both/unclear\n"
                        "Do not guess."
                    ),
                },
                {"type": "input_image", "image_url": data_url},
            ],
        }],
        text={"format": {"type": "json_schema", "name": "page_extract", "strict": True, "schema": schema}},
    )

    parsed = json.loads(resp.output_text)

    # ✅ result MUST be inside the function
    result = {
        "book_name": book_name,
        "author": author,
        "page_number": _safe_int(parsed.get("page_number")),
        "text": parsed.get("text") or "",
        "life_stage_flag": parsed.get("life_stage_flag") or "unclear",
        "source_file": image_path.name,
        "reference": str(image_path),
    }

    # ✅ retry page number if missing
    if result["page_number"] is None:
        pn = extract_page_number_from_bottom_crop(client, image_path)
        if pn is not None:
            result["page_number"] = pn

    return result



def score_regulation_and_healing(client: OpenAI, page_text: str, life_stage_flag: str) -> Dict[str, Any]:
    """
    Text-only classifier:
    - family_type (one of 4 or null/unknown)
    - regulation ratings A-F (0-3)
    - healing ratings G-L (0-3) [mostly meaningful for adulthood; still allow 0-3]
    """

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "family_type": {
                "anyOf": [
                    {
                        "type": "string",
                        "enum": ["chaotic_distressed", "narcissistic", "autistic_socially_odd", "alcoholic_addicted", "unknown"],
                    },
                    {"type": "null"},
                ]
            },
            "regulation_A_people_pleasing": {"type": "integer", "minimum": 0, "maximum": 3},
            "regulation_B_hyper_control_perfectionism": {"type": "integer", "minimum": 0, "maximum": 3},
            "regulation_C_explosive_outbursts": {"type": "integer", "minimum": 0, "maximum": 3},
            "regulation_D_dissociation_numbing": {"type": "integer", "minimum": 0, "maximum": 3},
            "regulation_E_addictive_self_soothing": {"type": "integer", "minimum": 0, "maximum": 3},
            "regulation_F_parentification": {"type": "integer", "minimum": 0, "maximum": 3},
            "healing_G_corrective_relationships": {"type": "integer", "minimum": 0, "maximum": 3},
            "healing_H_deep_therapy": {"type": "integer", "minimum": 0, "maximum": 3},
            "healing_I_boundaries_distance": {"type": "integer", "minimum": 0, "maximum": 3},
            "healing_J_creative_narrative_work": {"type": "integer", "minimum": 0, "maximum": 3},
            "healing_K_body_based_regulation": {"type": "integer", "minimum": 0, "maximum": 3},
            "healing_L_community_meaning": {"type": "integer", "minimum": 0, "maximum": 3},
            "evidence_snippets": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 6
            },
            "notes": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": [
            "family_type",
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
        ],
    }

    prompt = (
        "You are labeling ONE scanned page from a memoir/self-help book.\n\n"
        "Task:\n"
        "1) family_type: choose the single best match IF the page contains evidence about family dynamics.\n"
        "   Options:\n"
        "   - chaotic_distressed\n"
        "   - narcissistic\n"
        "   - autistic_socially_odd\n"
        "   - alcoholic_addicted\n"
        "   - unknown (if not enough evidence)\n\n"
        "2) Rate non-functional emotion regulation styles A–F on a 0–3 scale based ONLY on evidence in this page:\n"
        "   0=not present, 1=low, 2=medium, 3=very strong\n"
        "   A people-pleasing/self-erasure\n"
        "   B hyper-control/perfectionism\n"
        "   C explosive outbursts/acting out\n"
        "   D dissociation/emotional numbing\n"
        "   E addictive self-soothing (food/screens/substances/fantasy)\n"
        "   F parentification (little adult/rescuer)\n\n"
        "3) Rate healing factors G–L on 0–3 scale if present (same scale). If not present, use 0.\n"
        "   G corrective relationships\n"
        "   H deep therapy beyond symptoms\n"
        "   I boundaries/distance from dysfunction\n"
        "   J creative/narrative work\n"
        "   K body-based regulation\n"
        "   L community/meaning\n\n"
        f"Life stage hint (may help you interpret context): {life_stage_flag}\n\n"
        "Return strict JSON only. Include up to 6 short evidence_snippets quoted from the page text.\n"
        "If unsure, prefer lower scores and 'unknown'.\n"
    )

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "user", "content": [{"type": "input_text", "text": prompt + "\n\nPAGE TEXT:\n" + page_text}]}
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "page_regulation_healing_scores",
                "strict": True,
                "schema": schema,
            }
        },
    )
    return json.loads(resp.output_text)


    data_url = image_to_data_url_under_5mb(image_path)

    resp = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Extract the visible page text accurately.\n"
                            "- Preserve paragraph breaks with \\n.\n"
                            "- page_number: return the printed page number ONLY if visible, else null.\n"
                            "- life_stage_flag: classify the page content as childhood/adulthood/both/unknown.\n"
                            "Return JSON only."
                        ),
                    },
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "book_page_extract",
                "strict": True,
                "schema": schema,
            }
        },
    )

    data = json.loads(resp.output_text)

    out = {
        "book_name": meta["book_name"],
        "author": meta["author"],
        "page_number": _safe_int(data.get("page_number")),
        "text": data.get("text", "") or "",
        "life_stage_flag": data.get("life_stage_flag", "unknown"),
        "source_file": image_path.name,
        "reference": str(image_path),  # full path
    }
    return out


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='...'")

    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {ROOT_DIR}")

    client = OpenAI()

    # Methodical: recursively walk all folders under ROOT_DIR
    images = [
        p for p in sorted(ROOT_DIR.rglob("*"))
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ]

    if not images:
        print(f"No images found under {ROOT_DIR}")
        return

    for img in images:
        out_path = img.with_suffix(img.suffix + OUT_SUFFIX)  # image.jpg.json
        if out_path.exists():
            print(f"Skip (already has JSON): {out_path}")
            continue

        # 1) Parse book + author from the folder name FIRST
        meta = parse_book_author_from_folder(img)

        print(f"Extracting: {img}")
        data = extract_one_image(
            client,
            img,
            book_name=meta["book_name"],
            author=meta["author"] or ""
        )

        # If extraction failed and returned an error dict, don't score it
        if data.get("extraction_error"):
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved (with extraction_error): {out_path}")
            continue

        # 2) Score regulation/healing based on extracted text + life stage
        scores = score_regulation_and_healing(
            client,
            data["text"],
            data.get("life_stage_flag", "unclear")
        )
        data.update(scores)

        # 3) Write final JSON once (after all fields are present)
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")

    print("Done.")
if __name__ == "__main__":
    main()
