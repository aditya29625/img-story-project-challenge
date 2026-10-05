"""Context builder: turns structured Florence-2 vision JSON into a story prompt context string.

This is a deterministic text assembly step (no ML, no image-specific vocabulary).
Both single-image and multi-image (sequence) contexts are supported.
"""
import re, json
from typing import List, Dict, Any, Set

_IRREGULAR_PLURALS = {"people": "person", "children": "child", "men": "man",
                      "women": "woman", "feet": "foot"}


def normalize_entity(label: str) -> str:
    """Lower-case, collapse whitespace, singularise last word."""
    words = re.sub(r"\s+", " ", str(label).strip().lower()).split()
    if not words or words == [""]:
        return ""
    w = words[-1]
    if w in _IRREGULAR_PLURALS:
        w = _IRREGULAR_PLURALS[w]
    elif w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        w = w[:-1]
    words[-1] = w
    return " ".join(words)


def _od_labels(desc: Dict[str, Any]) -> List[str]:
    od = desc.get("od_labels")
    if od is None:
        od = [o for o in desc.get("objects", []) if len(o.split()) <= 2]
    return od


def frame_entities(desc: Dict[str, Any]) -> Set[str]:
    """Normalised entity set of one frame: characters + OD labels."""
    labels = list(desc.get("characters", [])) + list(_od_labels(desc))
    return {e for e in (normalize_entity(x) for x in labels) if e}


def recurring_entities(frames: List[Set[str]]) -> List[str]:
    """Entities appearing in at least 2 different frames (sorted, deterministic)."""
    counts: Dict[str, int] = {}
    for f in frames:
        for e in f:
            counts[e] = counts.get(e, 0) + 1
    return sorted(e for e, c in counts.items() if c >= 2)


def _dedupe(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_single_image_context(desc: Dict[str, Any], max_words: int = 180) -> str:
    """Build a structured context string from a single Florence-2 vision output."""
    parts = []

    if desc.get("scene"):
        parts.append(f"Scene: {desc['scene']}")

    chars = desc.get("characters", [])
    if chars:
        parts.append(f"Characters: {', '.join(chars)}")

    od = _dedupe(_od_labels(desc))
    od_set = set(od)
    others = [o for o in _dedupe(desc.get("objects", [])) if o not in od_set]
    obj_list = od + others[:5]
    if obj_list:
        parts.append(f"Objects: {', '.join(obj_list)}")

    actions = desc.get("actions", [])
    if actions:
        parts.append(f"Actions: {', '.join(actions)}")

    spatial = desc.get("spatial_relations", [])
    if spatial:
        parts.append(f"Spatial: {', '.join(spatial[:3])}")

    for r in _dedupe(desc.get("region_descriptions", []))[:3]:
        parts.append(f"Region: {r}")

    if desc.get("style_or_mood"):
        parts.append(f"Style: {desc['style_or_mood']}")

    context = ". ".join(parts) + "."
    words = context.split()
    if len(words) > max_words:
        context = " ".join(words[:max_words]) + "..."
    return context


def build_sequence_context(descs: List[Dict[str, Any]], max_words_per_image: int = 120) -> str:
    """Build a sequence-level context for multiple images in order."""
    if not descs:
        return ""

    parts = ["SEQUENCE OF IMAGES (in order):"]
    for i, desc in enumerate(descs):
        parts.append(f"\n--- IMAGE {i + 1} ---")
        if desc.get("scene"):
            parts.append(f"Location: {desc['scene']}")
        if desc.get("characters"):
            parts.append(f"Characters: {', '.join(desc['characters'])}")
        od = _dedupe(_od_labels(desc))
        if od:
            parts.append(f"Objects: {', '.join(od[:8])}")
        if desc.get("actions"):
            parts.append(f"Actions: {', '.join(desc['actions'])}")
        for r in _dedupe(desc.get("region_descriptions", []))[:2]:
            parts.append(f"Detail: {r}")
        if desc.get("style_or_mood"):
            parts.append(f"Mood: {desc['style_or_mood']}")

    if len(descs) > 1:
        parts.append("\n--- CONTINUITY NOTES ---")
        frames = [frame_entities(d) for d in descs]
        recurring = recurring_entities(frames)
        char_entities = {normalize_entity(c) for d in descs for c in d.get("characters", [])}
        rec_chars = [e for e in recurring if e in char_entities]
        rec_objs = [e for e in recurring if e not in char_entities]
        if rec_chars:
            parts.append(f"Recurring characters (in 2+ frames): {', '.join(rec_chars)}")
        if rec_objs:
            parts.append(f"Recurring objects (in 2+ frames): {', '.join(rec_objs[:10])}")
        parts.append("Maintain character identity and location consistency across frames.")
        parts.append("Connect events naturally; do not reset the story at each image.")

    context = " ".join(parts)
    words = context.split()
    max_total = max_words_per_image * len(descs)
    if len(words) > max_total:
        context = " ".join(words[:max_total]) + "..."
    return context


def build_story_prompt(context: str, num_images: int = 1, target_words: int = 100) -> str:
    """Build the Qwen story prompt from a context string."""
    if num_images == 1:
        return (
            f"Write a short story of {target_words - 20} to {target_words + 20} words "
            f"based only on this description of an image:\n{context}\nReturn only the story."
        )
    return (
        f"Here are {num_images} consecutive images from a sequence.\n{context}\n"
        f"Write ONE continuous story of about {target_words} words that follows these images in order, "
        f"using only what the descriptions say. Maintain character identity and location consistency. "
        f"Connect events naturally between frames. Do not describe each image separately. Return only the story."
    )


def save_vision_json(descs: List[Dict[str, Any]], output_path: str) -> None:
    """Persist structured vision output to JSON (runtime metadata excluded)."""
    output = {
        d.get("image_id", "unknown"): {
            "scene": d.get("scene", ""),
            "description": d.get("description", ""),
            "objects": d.get("objects", []),
            "od_labels": d.get("od_labels", []),
            "characters": d.get("characters", []),
            "actions": d.get("actions", []),
            "relationships": d.get("relationships", []),
            "spatial_relations": d.get("spatial_relations", []),
            "region_descriptions": d.get("region_descriptions", []),
            "style_or_mood": d.get("style_or_mood", ""),
            "ocr_text": d.get("ocr_text", ""),
        }
        for d in descs
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
