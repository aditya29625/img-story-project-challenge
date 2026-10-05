"""Evaluation metrics for Image -> Story (Module 10: evaluation + runtime logging).

Per image-story pair (score):
  grounding_score = 0.4*clip_n + 0.4*(1 - nli_contra_mean) + 0.2*(0 if attribute_conflict else 1)
    clip_n          = clip((clip_image_story_mean - 0.15) / 0.15, 0, 1)
                      CLIP ViT-B/32 cosine between the IMAGE and each story sentence, averaged.
                      This is the ONLY signal that directly compares the story with the image itself.
    nli_contra_mean = mean P(contradiction | premise = supplied visual context, hypothesis = story sentence).
                      NLI measures TEXTUAL consistency between story and the visual context string.
                      If the context is wrong, a story faithful to it still scores well — this is a limitation.
    attribute_conflict = deterministic colour/material check: catches "white cabinets" vs "oak cabinet" etc.

  length_valid   = 80 <= word_count <= 120 (benchmark requirement, not quality).
  grounding_pass = grounding_score >= 0.60 AND no attribute conflict AND length_valid.

Module 10 requirement: metric detects real failures the baseline misses; runtime is logged separately.
"""
import os, re, time, csv
from contextlib import ContextDecorator
from collections import Counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModelForSequenceClassification

from context_builder import frame_entities, recurring_entities, normalize_entity
from baseline import is_length_valid

THRESHOLD = 0.60
CLIP_LOW, CLIP_SPAN = 0.15, 0.15
TIMINGS = {}   # stage -> list of seconds


class Timer(ContextDecorator):
    """Context manager that appends elapsed seconds to TIMINGS[name]."""
    def __init__(self, name): self.name = name
    def __enter__(self): self.t = time.perf_counter(); return self
    def __exit__(self, *a): TIMINGS.setdefault(self.name, []).append(time.perf_counter() - self.t)


with Timer("load_models"):
    _clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _nli_name = "cross-encoder/nli-MiniLM2-L6-H768"
    _nli_tok = AutoTokenizer.from_pretrained(_nli_name)
    _nli = AutoModelForSequenceClassification.from_pretrained(_nli_name).eval()
_CONTRA = [i for i, l in _nli.config.id2label.items() if l.lower() == "contradiction"][0]

COLORS = {"white", "black", "red", "blue", "green", "yellow", "brown", "grey", "pink", "orange", "purple"}

# Materials whose natural colour is implied by context
_MATERIAL_COLOR = {
    "oak": {"brown"}, "mahogany": {"brown", "red"}, "ebony": {"black"},
    "pine": {"yellow", "white"}, "maple": {"yellow", "white"},
    "marble": {"white", "grey"}, "steel": {"grey"}, "gold": {"yellow"},
    "silver": {"grey"}, "copper": {"brown", "orange"},
}


def attribute_conflict(caption: str, story: str):
    """Return a list of (caption_word, story_word) conflict pairs (deterministic).

    Detects:
    1. A colour mentioned in the caption but contradicted by a different colour in the story.
    2. A material in the story whose natural colour conflicts with the caption's stated colour.
    """
    conflicts = []
    cap_words = set(re.findall(r"[a-z]+", caption.lower()))
    story_words = set(re.findall(r"[a-z]+", story.lower()))
    cap_colors = cap_words & COLORS
    story_colors = story_words & COLORS

    for cc in cap_colors:
        for sc in story_colors:
            if sc != cc:
                conflicts.append(f"colour:{cc}→{sc}")

    for mat, nat_colors in _MATERIAL_COLOR.items():
        if mat in story_words:
            for cc in cap_colors:
                if cc not in nat_colors:
                    conflicts.append(f"material:{mat}≠{cc}")
    return conflicts


def split_sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _clip_sims(image: Image.Image, texts: list) -> "torch.Tensor":
    """CLIP cosine similarity between one image and a list of texts."""
    inputs = _clip_proc(text=texts, images=image, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outs = _clip(**inputs)
    img_feat = outs.image_embeds / outs.image_embeds.norm(dim=-1, keepdim=True)
    txt_feat = outs.text_embeds / outs.text_embeds.norm(dim=-1, keepdim=True)
    return (img_feat * txt_feat).sum(dim=-1)


def _contradiction_probs(premise: str, hypotheses: list) -> "torch.Tensor":
    """NLI contradiction probability for (premise, each hypothesis)."""
    pairs = [(premise, h) for h in hypotheses]
    enc = _nli_tok(pairs, padding=True, truncation=True, return_tensors="pt", max_length=256)
    with torch.no_grad():
        logits = _nli(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    return probs[:, _CONTRA]


def grounding_from_components(clip_mean: float, nli_contra_mean: float, has_conflict: bool) -> tuple:
    """Composite grounding score from CLIP, NLI and attribute check."""
    clip_n = max(0.0, min(1.0, (clip_mean - CLIP_LOW) / CLIP_SPAN))
    nli_n = 1.0 - nli_contra_mean
    conflict_n = 0.0 if has_conflict else 1.0
    score = 0.4 * clip_n + 0.4 * nli_n + 0.2 * conflict_n
    return (clip_n, round(score, 4))


def is_grounding_pass(score: float, conflicts: list, length_valid: bool) -> bool:
    return score >= THRESHOLD and not conflicts and length_valid


def repetition_score(story: str) -> dict:
    """Trigram repetition rate (lower is better)."""
    toks = re.findall(r"[a-z']+", story.lower())
    sents = split_sentences(story)
    repeated_sentences = sum(c - 1 for c in Counter(sents).values() if c > 1)
    bigrams = list(zip(toks, toks[1:])) if len(toks) > 1 else []
    repeated_bigrams = sum(c - 1 for c in Counter(bigrams).values() if c > 1)
    trigrams = list(zip(toks, toks[1:], toks[2:])) if len(toks) > 2 else []
    repeated_trigrams = sum(c - 1 for c in Counter(trigrams).values() if c > 1)
    rate = repeated_trigrams / len(trigrams) if trigrams else 0.0
    return {
        "repeated_sentences": repeated_sentences,
        "repeated_bigrams": repeated_bigrams,
        "repeated_trigrams": repeated_trigrams,
        "repetition_rate": round(rate, 4),
        "distinct3": round(1.0 - rate if trigrams else 1.0, 4),
    }


TRANSITION_WORDS = {"then", "next", "after", "afterwards", "later", "suddenly",
                    "meanwhile", "before", "finally", "soon", "eventually", "following"}


def continuity_score(descs: list, story: str) -> dict:
    """Multi-image structural continuity proxy (entity overlap + transition words).

    NOT a narrative coherence measure. Generic labels like 'person' count like any other entity.
    Read the listed recurring_entities alongside the score.
    """
    n = len(descs or [])
    result = {"num_frames": n, "recurring_entities": [], "recurring_count": 0,
              "entity_consistency": None, "transition_markers": None, "adjacent_similarity": None}
    if n < 2:
        return result

    frames = [frame_entities(d) for d in descs]
    recurring = recurring_entities(frames)
    story_norm = " " + " ".join(normalize_entity(w) for w in re.findall(r"[a-z']+", story.lower())) + " "
    mentioned = [e for e in recurring if f" {e} " in story_norm]

    overlaps = [len(a & b) / len(a | b) for a, b in zip(frames, frames[1:]) if (a | b)]
    result.update({
        "recurring_entities": recurring,
        "recurring_count": len(recurring),
        "entity_consistency": round(len(mentioned) / len(recurring), 4) if recurring else None,
        "transition_markers": sum(1 for t in re.findall(r"[a-z]+", story.lower()) if t in TRANSITION_WORDS),
        "adjacent_similarity": round(sum(overlaps) / len(overlaps), 4) if overlaps else None,
    })
    return result


def score(image_path, caption, story):
    """Score one (image, context, story) triple. Evaluation runtime reported separately from generation."""
    t0 = time.perf_counter()
    sents = split_sentences(story) or [story]
    wc = len(story.split())
    image = Image.open(image_path).convert("RGB")

    t = time.perf_counter()
    with Timer("clip"):
        s = _clip_sims(image, sents + [caption])
    clip_s = time.perf_counter() - t

    sent_sims, cap_sim = s[:-1], float(s[-1])

    t = time.perf_counter()
    with Timer("nli"):
        c = _contradiction_probs(caption, sents)
    nli_s = time.perf_counter() - t

    t = time.perf_counter()
    bad = attribute_conflict(caption, story)
    clip_mean = float(sent_sims.mean())
    _, g = grounding_from_components(clip_mean, float(c.mean()), bool(bad))
    ok_len = is_length_valid(wc)
    rep = repetition_score(story)
    rules_s = time.perf_counter() - t

    return {
        "word_count": wc, "length_valid": ok_len,
        "truncated": not story.strip().rstrip("\"'").endswith((".", "!", "?")),
        "clip_image_story_mean": clip_mean,
        "clip_image_story_min": float(sent_sims.min()),
        "clip_image_caption": cap_sim,
        "nli_contra_mean": float(c.mean()),
        "nli_contra_max": float(c.max()),
        "attribute_conflict": bad,
        "grounding_score": g,
        "grounding_pass": is_grounding_pass(g, bad, ok_len),
        **rep,
        "eval_clip_s": round(clip_s, 3),
        "eval_nli_s": round(nli_s, 3),
        "eval_rules_s": round(rules_s, 4),
        "eval_total_s": round(time.perf_counter() - t0, 3),
    }


def evaluate_run(rows, out_csv):
    """rows: list of {image, caption, story}. Writes CSV, prints means and failure counts."""
    res = [{**r, **score(r["image"], r["caption"], r["story"])} for r in rows]
    for r in res:
        r["attribute_conflict"] = " ".join(r["attribute_conflict"])
    out_rows = [{**r, "image": os.path.basename(r["image"])} for r in res]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    num = ["clip_image_story_mean", "clip_image_story_min", "nli_contra_mean",
           "nli_contra_max", "grounding_score", "repetition_rate", "eval_total_s"]
    print(f"n = {len(res)}", {k: round(sum(r[k] for r in res) / len(res), 3) for k in num})
    print("fail: length", sum(not r["length_valid"] for r in res),
          "| attr_conflict", sum(bool(r["attribute_conflict"]) for r in res),
          "| grounding", sum(not r["grounding_pass"] for r in res))
    return res
