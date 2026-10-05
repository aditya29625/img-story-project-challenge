"""Improved vision stage: Florence-2-base structured visual understanding (Module 7).

Replaces BLIP's single sentence with:
  - <MORE_DETAILED_CAPTION>  → full scene description
  - <OD>                     → object detection labels
  - <DENSE_REGION_CAPTION>   → per-region descriptions

OCR is disabled (returns junk on illustrated/anime images).
"""
import os, re, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

FLORENCE_ID = "florence-community/Florence-2-base"
LOAD_MODE = "single"  # always one model in memory
LOAD_TIME_S = 0.0

_model_cache = {}

_CHARACTER_KEYWORDS = {
    "person", "man", "woman", "boy", "girl", "child", "people",
    "character", "figure", "human", "adult", "teenager", "kid",
    "hero", "villain", "warrior", "soldier", "knight", "wizard",
    "animal", "cat", "dog", "bird", "horse", "fish", "fox",
    "creature", "monster", "dragon", "spirit", "ghost",
}

_ACTION_VERBS = {
    "walking", "running", "standing", "sitting", "lying", "sleeping",
    "holding", "carrying", "wearing", "eating", "drinking", "reading",
    "writing", "looking", "watching", "gazing", "staring", "smiling",
    "laughing", "crying", "talking", "speaking", "listening",
    "playing", "working", "cooking", "painting", "drawing",
    "driving", "riding", "flying", "swimming", "jumping", "climbing",
    "opening", "closing", "pushing", "pulling", "lifting",
    "reaching", "pointing", "waving", "gesturing", "dancing", "singing",
}


def _load():
    global LOAD_TIME_S
    if "model" not in _model_cache:
        t0 = time.perf_counter()
        processor = AutoProcessor.from_pretrained(FLORENCE_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_ID, torch_dtype=torch.float32, trust_remote_code=True
        ).eval()
        LOAD_TIME_S = round(time.perf_counter() - t0, 2)
        _model_cache["model"] = model
        _model_cache["processor"] = processor
    return _model_cache["model"], _model_cache["processor"]


def _run(model, processor, image, task, max_new_tokens):
    inputs = processor(text=task, images=image, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(generated, task=task, image_size=(image.width, image.height))
    return generated, parsed


def _clean(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _dedupe(lst):
    seen, out = set(), []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _extract_entities(dense_labels):
    """Parse DENSE_REGION_CAPTION labels for characters, actions, spatial relations."""
    chars, actions, spatial, regions, objects = [], [], [], [], []

    spatial_patterns = [
        r"\bnext to\b", r"\bbeside\b", r"\bbehind\b", r"\bin front of\b",
        r"\babove\b", r"\bbelow\b", r"\bunder\b", r"\bover\b",
        r"\bbetween\b", r"\bnear\b", r"\bleft\b", r"\bright\b",
        r"\bcenter\b", r"\bcorner\b", r"\bedge\b", r"\bside\b",
        r"\binside\b", r"\boutside\b",
    ]

    for label in dense_labels:
        ll = label.lower()
        regions.append(label)
        for kw in _CHARACTER_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", ll) and kw not in chars:
                chars.append(kw)
        for verb in _ACTION_VERBS:
            if re.search(r"\b" + re.escape(verb) + r"\b", ll) and verb not in actions:
                actions.append(verb)
        for pat in spatial_patterns:
            m = re.search(pat, ll)
            if m:
                start = max(0, m.start() - 15)
                end = min(len(ll), m.end() + 15)
                phrase = ll[start:end].strip()
                if phrase not in spatial:
                    spatial.append(phrase)
                break
        has_char = any(re.search(r"\b" + re.escape(kw) + r"\b", ll) for kw in _CHARACTER_KEYWORDS)
        if not has_char:
            objects.append(label)

    return {"chars": chars, "actions": actions, "spatial": spatial,
            "regions": regions, "objects": objects}


def describe(image_path):
    """Run Florence-2 on an image and return structured vision JSON.

    Returns a dict with keys:
      image_id, scene, description, objects, od_labels,
      characters, actions, relationships, spatial_relations,
      region_descriptions, style_or_mood, ocr_text,
      runtime_s, model_load_s
    """
    t_start = time.perf_counter()
    model, processor = _load()
    t_loaded = time.perf_counter()

    image = Image.open(image_path).convert("RGB")
    out = {}

    for key, task, mnt in [
        ("detailed_caption", "<MORE_DETAILED_CAPTION>", 160),
        ("od", "<OD>", 96),
        ("dense", "<DENSE_REGION_CAPTION>", 256),
    ]:
        _, parsed = _run(model, processor, image, task, mnt)
        out[key] = parsed.get(task) if isinstance(parsed, dict) else parsed

    # Object labels from OD
    od_labels = []
    for lbl in (out["od"] or {}).get("labels", []):
        cleaned = _clean(lbl).lower()
        if cleaned and cleaned not in od_labels:
            od_labels.append(cleaned)

    # Structured entities from dense captions
    dense_labels = (out["dense"] or {}).get("labels", [])
    dense = _extract_entities(dense_labels)

    all_objects = _dedupe(od_labels + dense["objects"])

    detailed = _clean(out["detailed_caption"])

    # Scene: first sentence of detailed caption
    scene = ""
    if detailed:
        first = detailed.split(".")[0]
        scene = first[:150] if first else ""

    # Style / mood
    style_mood = ""
    mood_kws = {"whimsical", "playful", "serene", "peaceful", "cheerful", "joyful",
                "festive", "lively", "dark", "gloomy", "mysterious", "ominous",
                "cozy", "warm", "cold", "bright", "colorful", "vibrant",
                "anime", "illustration", "cartoon", "painting"}
    for kw in mood_kws:
        if kw in detailed.lower():
            style_mood = kw
            break

    # Characters and actions from detailed caption too
    detailed_chars = [kw for kw in _CHARACTER_KEYWORDS
                      if re.search(r"\b" + re.escape(kw) + r"\b", detailed.lower())]
    detailed_actions = [v for v in _ACTION_VERBS
                        if re.search(r"\b" + re.escape(v) + r"\b", detailed.lower())]

    return {
        "image_id": os.path.basename(image_path),
        "scene": scene,
        "description": detailed,
        "objects": all_objects,
        "od_labels": od_labels,
        "characters": _dedupe(dense["chars"] + detailed_chars),
        "actions": _dedupe(dense["actions"] + detailed_actions),
        "relationships": _dedupe(dense["spatial"]),
        "ocr_text": "",          # OCR disabled (junk on illustrated images)
        "spatial_relations": _dedupe(dense["spatial"]),
        "region_descriptions": dense["regions"],
        "style_or_mood": style_mood,
        "runtime_s": round(time.perf_counter() - t_loaded, 2),
        "model_load_s": round(t_loaded - t_start, 2),
    }


if __name__ == "__main__":
    import sys, json
    for p in sys.argv[1:]:
        d = describe(p)
        print(json.dumps({k: v for k, v in d.items() if k != "region_descriptions"}, indent=2))
    print(f"\nModel load: {LOAD_TIME_S}s | Mode: {LOAD_MODE}")
