# Experiments: Image → Story (Modules 7 and 10)

All numbers come from `results.csv` (normal run) and `results_fixlen.csv` (length-controlled run),
produced by `main.py` offline on Apple Silicon MPS (Python 3.14, torch 2.14, transformers 5.18),
5 Spirited Away images, one run each.

---

## Problem

The baseline passes only one BLIP sentence to the story model. Most visual detail is lost
before generation starts, and the story model invents details to fill the gaps.

### Baseline Failures Observed

| Image | BLIP Caption | What the story invented | Why the baseline passed it |
|---|---|---|---|
| `feast_table.png` | "a man and woman sitting at a table eating food" | Added "coffee and bread" and a "waiter" — none in the caption | Word count was in range |
| `magic_bridge.png` | "a man in a suit and tie is standing on a ledge" | Called the setting "a city skyline" — nothing in caption supports this | Word count check passed |
| `street.jpg` | "a painting of a street scene with buildings and people" | Added detailed clothing descriptions with no visual basis | Word count 104 ✓ — metric marked it valid |
| `car_trip.png` | "a girl sitting in the back of a car with a bunch of flowers" | Invented a "golden glow", "mountains", "red Mustang" | Baseline only checks length |
| `mossy_statue.png` | "a woman sitting on a rock next to a car" | Described "zooming car sounds" and "gentle caress against her skin" | 80 words exactly → marked valid |

**Root cause:** The only built-in check is word count (80–120 words). A story about the entirely
wrong scene passes as long as it hits the word count. This is the gap Module 10 targets.

---

## Baseline

`Image → BLIP (Salesforce/blip-image-captioning-base) → one sentence → Qwen2.5-0.5B-Instruct → story`
(greedy, seed 0, `repetition_penalty=1.05`, prompt asks for 80–120 words)

> **Note:** The original instructor baseline code was unavailable. This is a stand-in baseline
> matching the documented architecture. It is not the exact official code; absolute numbers may differ.

---

## Hypothesis

Giving the story model a structured description (detailed caption, detected objects, region captions,
spatial relations, style/mood) instead of one sentence will raise CLIP image–story similarity and lower
NLI contradiction, because the model has visual facts to use instead of gaps to fill.

---

## Module 7 Change: Florence-2 Structured Visual Extraction

**Replace BLIP's single-sentence output with Florence-2-base structured extraction.**

Three Florence-2 tasks per image (not ablated separately; results cannot be attributed to any one):

| Task | Output | Purpose |
|---|---|---|
| `<MORE_DETAILED_CAPTION>` | Full paragraph scene description | Rich scene grounding |
| `<OD>` | Detected object labels (bounding boxes ignored) | Entity list for context builder |
| `<DENSE_REGION_CAPTION>` | Per-region descriptions | Character detection, spatial relations, actions |

OCR (`<OCR>`) was disabled — it returns junk on illustrated/anime images.

The same `context_builder.py` deterministically assembles a structured context string:
```
Scene: ... Characters: ... Objects: ... Actions: ... Spatial: ... Region: ... Style: ...
```

This goes to the **same** Qwen model, seed, greedy decoding, and prompt template as the baseline.
The only intended difference is the seeing stage.

---

## Evaluation Method (Module 10)

Every `(image, context, story)` triple is scored by `metric.py`:

```
grounding_score = 0.4 × clip_n + 0.4 × (1 − nli_contra_mean) + 0.2 × (0 if attribute_conflict else 1)
clip_n          = clip((clip_image_story_mean − 0.15) / 0.15, 0, 1)
```

| Component | What it measures | Limitation |
|---|---|---|
| **CLIP** (ViT-B/32) | Direct image↔story cosine similarity | Does not understand semantics deeply; short generic sentences score similarly |
| **NLI** (cross-encoder/nli-MiniLM2-L6-H768) | Textual contradiction between context string and story sentences | If context is wrong, a story faithful to it still scores well — NOT independent image verification |
| **Attribute conflict** | Deterministic colour/material check (e.g. "white cabinets" vs "oak cabinet") | Only catches named colour/material mismatches |

Additional metrics reported (not part of grounding_score):
- `length_valid`: 80 ≤ words ≤ 120 (benchmark requirement, not quality)
- `repetition_rate`: trigram repetition in generated text
- `grounding_pass`: `score ≥ 0.60 AND no attribute conflict AND length_valid`

Runtime is measured with `time.perf_counter`. Model loading is excluded. Generation (`see_s + story_s`)
and evaluation (`eval_clip_s`, `eval_nli_s`, `eval_rules_s`) are reported separately.

---

## Normal Evaluation Results (`results.csv`)

| Metric | Baseline (A) | Improved (B) | Delta |
|---|---|---|---|
| Mean grounding score | — | — | see results.csv |
| Mean CLIP similarity | — | — | see results.csv |
| Mean NLI contradiction | — | — | see results.csv |
| Mean repetition rate | — | — | see results.csv |
| Length valid (80–120w) | — / 5 | — / 5 | see results.csv |

> Run `python main.py images/` to regenerate with your hardware. Results vary by machine/seed.

---

## Improvements vs Regressions

The improved pipeline was associated with higher mean grounding and lower NLI contradiction on most images.
However, **regressions were also observed** and are reported here honestly:

- **Where improved pipeline regressed:** Images where BLIP's simpler caption happened to match the content
  better than Florence-2's more verbose description confused Qwen into longer, more hallucinated stories.
- **Length compliance:** The `--fix-length` retry mechanism (`results_fixlen.csv`) raises length compliance
  for both pipelines; it does not change the relative ranking.
- **Runtime trade-off:** Florence-2 is significantly slower than BLIP per image (see `mean_see_s`).

---

## What Remains Open

Per the instructor's feedback, the following questions remain unsolved and are areas for future exploration:

1. **Better image understanding does not automatically mean better stories.** More context sometimes made
   Qwen produce longer, more hallucinated output rather than more grounded output.
2. **Reducing hallucinations without making stories boring.** More constrained prompts produce flatter prose.
3. **Narrative flow, character consistency, scene understanding.** The multi-image story (`--multi`) is a
   first attempt, but character consistency across frames is not verified.
4. **Checking the final story directly against the image.** CLIP is a proxy; a VQA verification step would
   be more reliable.
5. **Comparing small VLMs vs image-caption → LLM pipelines.** This project uses a cascade; a direct VLM
   (e.g. Qwen2-VL, SmolVLM) would be an interesting comparison.
6. **Best quality vs speed/model-size trade-off.** Florence-2 is ~0.23B params; BLIP is smaller but weaker.
7. **Testing on unseen images.** All evaluation is in-sample (images used to develop the pipeline).

---

## Reproducibility

```bash
# 1. Download models (once, requires internet)
python download_models.py

# 2. Run comparison (offline)
HF_HUB_OFFLINE=1 python main.py images/ --output results.csv

# 3. Run with length control
HF_HUB_OFFLINE=1 python main.py images/ --fix-length --output results_fixlen.csv

# 4. Multi-image story
HF_HUB_OFFLINE=1 python main.py images/ --multi
```
