"""
Extract + label book pages from images under:
  ~/Documents/books/inbox_photos/data_test/<bookname_authorname>/*.jpg

Creates a sidecar JSON per image: image.jpg.json

What it does (single pass):
1) Vision extraction:
   - page_number (only if visible)
   - title/subtitle (only if clearly present)
   - page text
   - narration_voice: child/adolescent/adult/mixed/unclear
2) Text-only labeling (based on extracted page text):
   - family_type (single label)
   - dysregulation symptom flags (bool) + optional evidence snippets
   - diagnoses mentioned flags (bool) + optional evidence snippets

Requires:
  pip install openai pillow
Env:
  export OPENAI_API_KEY="..."
"""

import os
import json
import io
import base64
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from PIL import Image
from openai import OpenAI

# ---------------------------
# config
# ---------------------------
ROOT_DIR_DEFAULT = Path.home() / "Documents" / "books" / "inbox_photos" / "data_test"
OUT_SUFFIX = ".json"
MODEL = "gpt-4o-mini"

# your folder naming is: bookname_authorname  (underscore)
FOLDER_SEP = "_"


# ---------------------------
# helpers
# ---------------------------
def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(x)
    except Exception:
        return None


def _clean_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _pretty_from_slug(s: str) -> str:
    return s.replace("_", " ").strip()


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
        if img.size[0] < 400 or img.size[1] < 400:
            break

    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------
# folder parsing
# ---------------------------
def parse_book_author_from_folder(book_folder: Path) -> Tuple[str, str]:
    """
    Folder expected: <bookname_authorname>
    Safe heuristic: author = last token, book = rest
    """
    name = book_folder.name.strip()
    parts = [p for p in name.split(FOLDER_SEP) if p]

    if len(parts) >= 2:
        author_part = parts[-1]
        book_part = "_".join(parts[:-1])
        return _pretty_from_slug(book_part), _pretty_from_slug(author_part)

    return _pretty_from_slug(name), ""


# ---------------------------
# Vision extraction
# ---------------------------
def extract_one_image(client: OpenAI, image_path: Path, book_name: str, author: str) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "page_number": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "text": {"type": "string"},
            "title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "subtitle": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "narration_voice": {
                "type": "string",
                "enum": ["child", "adolescent", "adult", "mixed", "unclear"],
            },
        },
        "required": ["page_number", "text", "title", "subtitle", "narration_voice"],
    }

    prompt = (
        "Extract content from this photographed book page.\n\n"
        "Return strict JSON with:\n"
        "- page_number: ONLY if printed and clearly visible, else null\n"
        "- title: ONLY if this page clearly shows a section/chapter title, else null\n"
        "- subtitle: ONLY if clearly printed under the title, else null\n"
        "- text: exact visible page text; preserve paragraph breaks as \\n\\n\n"
        "- narration_voice: the main character's voice on this page: child/adolescent/adult/mixed/unclear\n\n"
        "Rules:\n"
        "- Do NOT guess page_number/title/subtitle.\n"
        "- If unsure, use null (for title/subtitle/page_number) or 'unclear' (for narration_voice).\n"
        "- Return JSON only.\n"
    )

    data_url = image_to_data_url_under_5mb(image_path)

    resp = client.responses.create(
        model=MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
        text={"format": {"type": "json_schema", "name": "page_extract", "strict": True, "schema": schema}},
    )

    parsed = json.loads(resp.output_text)

    out: Dict[str, Any] = {
        "book_name": book_name,
        "author": author if author else None,
        "page_number": _safe_int(parsed.get("page_number")),
        "text": parsed.get("text") or "",
        "source_file": image_path.name,
        "reference": str(image_path),
        "narration_voice": parsed.get("narration_voice") or "unclear",
    }

    title = _clean_str(parsed.get("title"))
    subtitle = _clean_str(parsed.get("subtitle"))
    if title is not None:
        out["title"] = title
    if subtitle is not None:
        out["subtitle"] = subtitle

    return out


# ---------------------------
# Text-only labeling (family + dysregulation + diagnoses)
# ---------------------------
def label_page_text(client: OpenAI, page_text: str, narration_voice: str) -> Dict[str, Any]:
    """
    Adds:
    Adds:
    - family_type: single label (family-of-origin)
    - narrator dysregulation flags (ONLY narrator/main character)
    - family-members dysregulation flags (NOT narrator)
    - diagnoses mentioned flags (explicitly mentioned only)
    - evidence_snippets: up to 8 short quotes
    - notes: optional
    """
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "family_type": {
                "type": "string",
                "enum": [
                    "chaotic_distressed",
                    "narcissistic",
                    "autistic_socially_odd",
                    "alcoholic_addicted",
                    "unknown",
                ],
            },

           # Narrator dysregulation (ONLY narrator/main character)
            "dysreg_physical_pain": {"type": "boolean"},
            "dysreg_dissociation_numbing": {"type": "boolean"},
            "dysreg_addictive_self_soothing": {"type": "boolean"},
            "dysreg_explosive_outbursts": {"type": "boolean"},
            "dysreg_people_pleasing": {"type": "boolean"},
            "dysreg_hyper_control_perfectionism": {"type": "boolean"},
            "dysreg_parentification_rescuer": {"type": "boolean"},

            # Family-members dysregulation (NOT narrator)
            "fam_dysreg_physical_pain": {"type": "boolean"},
            "fam_dysreg_dissociation_numbing": {"type": "boolean"},
            "fam_dysreg_addictive_self_soothing": {"type": "boolean"},
            "fam_dysreg_explosive_outbursts": {"type": "boolean"},
            "fam_dysreg_people_pleasing": {"type": "boolean"},
            "fam_dysreg_hyper_control_perfectionism": {"type": "boolean"},
            "fam_dysreg_parentification_rescuer": {"type": "boolean"},

            # Diagnoses mentioned (explicit mention only)
            "dx_autism": {"type": "boolean"},
            "dx_adhd": {"type": "boolean"},
            "dx_addictions": {"type": "boolean"},
            "dx_depression": {"type": "boolean"},
            "dx_anxiety": {"type": "boolean"},
            "dx_other": {"type": "boolean"},
            "dx_other_label": {"anyOf": [{"type": "string"}, {"type": "null"}]},


            "evidence_snippets": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 8,
            },
            "notes": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": [
            "family_type",
            # narrator
            "dysreg_physical_pain",
            "dysreg_dissociation_numbing",
            "dysreg_addictive_self_soothing",
            "dysreg_explosive_outbursts",
            "dysreg_people_pleasing",
            "dysreg_hyper_control_perfectionism",
            "dysreg_parentification_rescuer",

            # family
            "fam_dysreg_physical_pain",
            "fam_dysreg_dissociation_numbing",
            "fam_dysreg_addictive_self_soothing",
            "fam_dysreg_explosive_outbursts",
            "fam_dysreg_people_pleasing",
            "fam_dysreg_hyper_control_perfectionism",
            "fam_dysreg_parentification_rescuer",

            # diagnoses
            "dx_autism",
            "dx_adhd",
            "dx_addictions",
            "dx_depression",
            "dx_anxiety",
            "dx_other",
            "dx_other_label",

            "evidence_snippets",
            "notes",
        ],
    }

    prompt = (
    "You are labeling ONE page of text from a memoir/self-help book.\n\n"
    f"Narration voice hint: {narration_voice}\n\n"

    "Task A — family_type (family-of-origin):\n"
    "Choose the single best match IF the page contains evidence about family dynamics the narrator grew up in.\n"
    "Options:\n"
    "- chaotic_distressed\n"
    "- narcissistic\n"
    "- autistic_socially_odd\n"
    "- alcoholic_addicted\n"
    "- unknown (if not enough evidence on this page)\n\n"

    "CRITICAL SEPARATION RULE (do NOT mix):\n"
    "1) dysreg_* flags are ONLY for the NARRATOR / MAIN CHARACTER's OWN symptoms/behaviors/internal experience on THIS page.\n"
    "   Examples that count for dysreg_*: the narrator says they yelled, exploded, dissociated, people-pleased, controlled, used addictive soothing,\n"
    "   felt physical pain, or acted as a caretaker.\n"
    "2) fam_dysreg_* flags are ONLY for FAMILY MEMBERS' behaviors/symptoms shown on THIS page (e.g., father yells, mother controls, sibling explodes).\n"
    "3) If the page only describes the FAMILY/ENVIRONMENT (rules, shouting, criticism) and does NOT show the narrator doing/feeling it,\n"
    "   then all dysreg_* MUST be FALSE and only fam_dysreg_* (and/or family_type) can be TRUE.\n\n"

    "PROOF REQUIREMENT FOR dysreg_* (STRICT):\n"
    "- If you set any dysreg_* = TRUE, you MUST include at least one quote in evidence_snippets that clearly attributes it to the narrator.\n"
    "- The quote must describe the narrator's own action/internal state (e.g., 'I...', 'my...', 'we...' where the narrator is the actor/feeler).\n"
    "- If you cannot provide such a narrator-attributed quote, set dysreg_* = FALSE.\n\n"

    "ANTI-MISLABEL RULES (STRICT):\n"
    "- Do NOT mark dysreg_explosive_outbursts TRUE for yelling done by the father/mother/someone else.\n"
    "- Do NOT mark dysreg_hyper_control_perfectionism TRUE for rules imposed by the father/mother.\n"
    "- If a parent is the one yelling/controlling, use fam_dysreg_explosive_outbursts / fam_dysreg_hyper_control_perfectionism instead.\n\n"

    "Task B — narrator dysregulation symptom flags (TRUE only if clearly present in the NARRATOR on this page, else FALSE):\n"
    "IMPORTANT DEFINITIONS (be strict):\n"
    "- dysreg_explosive_outbursts = TRUE ONLY if the narrator shows anger/rage, yelling, screaming, fighting,\n"
    "  throwing/breaking things, aggression, tantrums/meltdowns framed as anger, or explicit loss of temper.\n"
    "  Do NOT mark TRUE for vague words like 'blowups' unless narrator anger is explicit.\n"
    "- dysreg_hyper_control_perfectionism = TRUE only if the narrator's perfectionism/control/rigidity is described as a coping style (not just planning).\n"
    "- dysreg_people_pleasing = TRUE only if the narrator's appeasing/approval-seeking/self-erasure is described.\n"
    "- dysreg_dissociation_numbing = TRUE only if narrator emotional shutdown, numbness, derealization, 'checked out', or disconnection is described.\n"
    "- dysreg_addictive_self_soothing = TRUE only if narrator compulsive soothing via food/screens/substances/shopping/sex/fantasy etc. is described.\n"
    "- dysreg_physical_pain = TRUE only if narrator bodily pain/somatic symptoms are described (not just sensory overwhelm).\n"
    "- dysreg_parentification_rescuer = TRUE only if narrator/main character is cast as caretaker/rescuer/little adult.\n\n"

    "Task B2 — family-members dysregulation flags (TRUE only if a FAMILY MEMBER shows it on this page, else FALSE):\n"
    "- fam_dysreg_explosive_outbursts includes yelling/insults/threats by family members.\n"
    "- fam_dysreg_hyper_control_perfectionism includes rigid controlling rules/forbidding imposed by family members.\n\n"

    "Task C — diagnoses mentioned on this page (TRUE only if explicitly mentioned, else FALSE):\n"
    "- dx_autism, dx_adhd, dx_addictions, dx_depression, dx_anxiety\n"
    "- dx_other: TRUE only if another diagnosis is explicitly mentioned; then set dx_other_label\n\n"

    "Evidence:\n"
    "- Include up to 8 short evidence_snippets quoted verbatim from the page text.\n"
    "- Prefer quotes that support TRUE flags.\n"
    "- If any dysreg_* is TRUE, at least one quote must be narrator-attributed.\n\n"

    "Rules:\n"
    "- Base everything ONLY on the page text.\n"
    "- If there is no evidence, use family_type='unknown' and FALSE flags.\n"
    "- Return strict JSON only.\n\n"

    "PAGE TEXT:\n"
    f"{page_text}"
)


    resp = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        text={"format": {"type": "json_schema", "name": "page_labels", "strict": True, "schema": schema}},
    )

    return json.loads(resp.output_text)




# ---------------------------
# main runner
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(ROOT_DIR_DEFAULT),
                        help="Root folder that contains book folders")
    parser.add_argument("--book_folder", type=str, default="",
                        help="Run ONLY on one book folder name (e.g. itsnotyourimagination_revitalhorvitz)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional: process only first N images (0 = no limit)")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='...'")

    root_dir = Path(args.root).expanduser()
    if not root_dir.exists():
        raise FileNotFoundError(f"Missing folder: {root_dir}")

    # safety stop
    if "data_raw" in str(root_dir):
        raise RuntimeError("Safety stop: root_dir points to data_raw. Use data_test only.")

    client = OpenAI()

    # decide which book folders to process
    if args.book_folder:
        book_dirs = [root_dir / args.book_folder]
        if not book_dirs[0].exists():
            raise FileNotFoundError(f"Book folder not found: {book_dirs[0]}")
    else:
        book_dirs = [p for p in sorted(root_dir.iterdir()) if p.is_dir()]

    # collect images
    images: List[Path] = []
    for book_dir in book_dirs:
        for p in sorted(book_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                images.append(p)

    if args.limit and args.limit > 0:
        images = images[:args.limit]

    if not images:
        print(f"No images found under: {root_dir}")
        return

    processed = 0

    for img in images:
        out_path = img.with_suffix(img.suffix + OUT_SUFFIX)
        if out_path.exists():
            continue

        book_name, author = parse_book_author_from_folder(img.parent)

        try:
            extracted = extract_one_image(client, img, book_name=book_name, author=author)

            # label text only if we got some text
            if extracted.get("text", "").strip():
                labels = label_page_text(
                    client,
                    extracted["text"],
                    extracted.get("narration_voice", "unclear"),
                )
                extracted.update(labels)
            else:
                # still write a consistent empty label set (no nulls)
                extracted.update({
                    "family_type": "unknown",
                    "dysreg_physical_pain": False,
                    "dysreg_dissociation_numbing": False,
                    "dysreg_addictive_self_soothing": False,
                    "dysreg_explosive_outbursts": False,
                    "dysreg_people_pleasing": False,
                    "dysreg_hyper_control_perfectionism": False,
                    "dysreg_parentification_rescuer": False,
                    "dx_autism": False,
                    "dx_adhd": False,
                    "dx_addictions": False,
                    "dx_depression": False,
                    "dx_anxiety": False,
                    "dx_other": False,
                    "dx_other_label": None,
                    "evidence_snippets": [],
                    "notes": "No text extracted; labeling skipped.",
                })

            data = extracted

        except Exception as e:
            data = {
                "book_name": book_name,
                "author": author if author else None,
                "page_number": None,
                "text": "",
                "source_file": img.name,
                "reference": str(img),
                "extraction_error": str(e),
            }

        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        processed += 1
        print(f"Saved: {out_path}")

    print(f"Done. New JSONs created: {processed}")


if __name__ == "__main__":
    main()
