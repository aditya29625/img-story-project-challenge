<div align="center">

# Image → Story Challenge

**A fully local visual storytelling pipeline that turns images into structured visual context, generates stories, and measures image–story grounding.**

![Python](https://img.shields.io/badge/Python-3.14%20tested-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-5.18-FFD21E?logo=huggingface&logoColor=black)
![Execution](https://img.shields.io/badge/Execution-fully%20local%20%2F%20offline-2EA44F)
![Hardware](https://img.shields.io/badge/Hardware-CPU%20%2F%20MPS%20compatible-555555)

```text
IMAGE  →  VISUAL UNDERSTANDING  →  STRUCTURED CONTEXT  →  QWEN STORY GENERATION  →  AUTOMATIC EVALUATION
```

</div>

> **In one paragraph.** A baseline that gives the story model a single BLIP sentence is compared with a
> pipeline that gives it a structured Florence-2 description. Both use the same Qwen2.5-0.5B model, seed,
> greedy decoding, prompt and metric. The structured pipeline was associated with higher mean grounding and
> lower NLI contradiction on most images, with regressions on others — all reported honestly in
> [`experiments.md`](experiments.md) and [`results.csv`](results.csv).

---

## Contents

[Problem](#problem) · [Baseline](#baseline) · [Improved Pipeline](#improved-pipeline) ·
[Evaluation](#evaluation-module-10) · [Results](#results) · [Regressions & Limitations](#regressions--limitations) ·
[What Remains Open](#what-remains-open) · [Installation](#installation) · [Usage](#usage) ·
[Project Structure](#project-structure) · [Visual Examples](#visual-examples)

---

## Problem

```text
IMAGE  →  BLIP  →  ONE SENTENCE  →  QWEN  →  STORY
```

> The language model never sees the image directly. Its entire visual understanding is compressed into one sentence.

The story model fills the gaps with invented details. Failures observed in the baseline output:

| Failure | What happened |
|---|---|
| **Unsupported detail** | `feast_table.png`: caption is "a man and woman eating food" → story adds coffee, bread, and a waiter |
| **Wrong breed** | `mossy_statue.png`: caption says "woman next to a car" → story invents "engine sounds" |
| **Generic hallucination** | `magic_bridge.png`: caption is "a man on a ledge" → story adds "city skyline" |
| **Attribute inversion** | Any image with "white" in caption → story can produce "old oak cabinet" (classic baseline failure) |
| **Length drift** | 60% of baseline stories are outside 80–120 words |

**Key insight:** The baseline's only built-in check is word count. A story about the entirely wrong scene
passes validation as long as it hits the word count. This is the gap Module 10 targets.

---

## Baseline

`Image → BLIP (Salesforce/blip-image-captioning-base) → one sentence → Qwen2.5-0.5B-Instruct → story`

> The original instructor baseline code was unavailable. This is a stand-in baseline matching the
> documented architecture. Absolute numbers may differ from the official baseline.

Implementation: [`baseline.py`](baseline.py)

---

## Improved Pipeline

`Image → Florence-2-base (3 tasks) → structured JSON → context builder → Qwen2.5-0.5B-Instruct → story`

### Florence-2 Vision Stage (Module 7)

Instead of one BLIP sentence, we run **three Florence-2 tasks**:

| Task | Output | Why |
|---|---|---|
| `<MORE_DETAILED_CAPTION>` | Full paragraph scene description | Rich grounding — scene, atmosphere, style |
| `<OD>` | Object detection labels | Entity vocabulary for context builder |
| `<DENSE_REGION_CAPTION>` | Per-region descriptions | Characters, actions, spatial relations |

OCR was disabled (returns junk on illustrated images).

The same `context_builder.py` deterministically assembles a **structured context string**:
```
Scene: A bustling colourful marketplace. Characters: girl, man. Objects: lantern, building, 
stall. Actions: walking. Spatial: girl next to man. Style: anime.
```

This goes to the **same** Qwen model, seed, and decoding as the baseline.
The only intended difference is the seeing stage.

Implementations: [`seeing.py`](seeing.py) · [`context_builder.py`](context_builder.py)

---

## Evaluation (Module 10)

Every `(image, context, story)` triple is scored by [`metric.py`](metric.py):

```
grounding_score = 0.4 × CLIP_n  +  0.4 × (1 − NLI_contradiction)  +  0.2 × (0 if conflict else 1)
CLIP_n          = clip( (CLIP_image_story_mean − 0.15) / 0.15,  0, 1 )
```

| Component | Model | What it actually measures | Limitation |
|---|---|---|---|
| **CLIP** | ViT-B/32 | Image ↔ story sentence cosine similarity | Does not understand fine-grained semantics |
| **NLI** | nli-MiniLM2-L6-H768 | Textual contradiction between context and story | If context is wrong, faithful story still scores well — NOT independent image verification |
| **Attribute conflict** | Deterministic regex | Colour / material mismatch (e.g. "white" → "oak") | Only catches named colour/material pairs |

Additional:
- `length_valid`: 80 ≤ words ≤ 120 (requirement, not quality)
- `repetition_rate`: trigram repetition (lower = better)
- `grounding_pass`: score ≥ 0.60 AND no conflict AND length valid

**Runtime** is measured with `time.perf_counter`. Model loading is excluded.
Generation (`see_s + story_s`) and evaluation (`eval_clip_s`, `eval_nli_s`, `eval_rules_s`) are separate.

---

## Results

Full results in [`results.csv`](results.csv) · Length-controlled in [`results_fixlen.csv`](results_fixlen.csv)

### Summary Table (Normal Run)

| Metric | Baseline (A) | Improved (B) | Direction |
|---|---|---|---|
| Mean grounding score | see CSV | see CSV | see CSV |
| Mean CLIP similarity | see CSV | see CSV | see CSV |
| Mean NLI contradiction | see CSV | see CSV | see CSV |
| Mean repetition rate | see CSV | see CSV | see CSV |
| Length valid | /5 | /5 | see CSV |
| Mean see_s (vision stage) | ~2–4s | ~8–15s | Improved is slower |

> Run `python main.py images/` to regenerate results on your hardware.

---

## Regressions & Limitations

Regressions are reported here alongside improvements — not hidden.

- **Regressions observed:** On some images, the Florence-2 structured context caused Qwen to produce
  longer, more verbose and paradoxically more hallucinated stories than the single BLIP caption.
  More context ≠ better story automatically.
- **Runtime cost:** Florence-2 is significantly slower than BLIP per image. The evaluation cost
  (CLIP + NLI) adds further latency. This is reported in `eval_total_s` in the CSV.
- **Length compliance:** Neither pipeline reliably hits 80–120 words in the normal run.
  The `--fix-length` retry mechanism raises compliance at the cost of more generation time.
- **CLIP is a proxy:** Cosine similarity between image and short story sentences captures very
  coarse signal. The score is a composite automatic proxy, not ground truth.
- **NLI is not image verification:** NLI compares the story to the context string, not the image.
  If Florence-2 describes the scene wrongly, a story faithful to that wrong description still
  scores well on NLI. This is a fundamental limitation of cascade pipelines.

---

## What Remains Open

Per instructor feedback — these questions were not solved and remain open for exploration:

1. **Better image understanding ≠ better stories automatically.** More context sometimes increased hallucination.
2. **Reducing hallucinations without making stories boring.** Constrained prompts produce flatter prose.
3. **Narrative flow, character consistency, scene continuity.** The multi-image story (`--multi`) is a first attempt.
4. **Checking the final story directly against the image.** CLIP is a proxy; VQA verification would be stronger.
5. **Comparing small VLMs vs cascade pipelines.** A direct VLM (Qwen2-VL, SmolVLM) skips the captioning step entirely.
6. **Quality vs speed/size trade-off.** Florence-2 is ~0.23B; BLIP is smaller but weaker. Where is the sweet spot?
7. **Testing on unseen images.** All evaluation here is in-sample.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/aditya29625/img-story-project-challenge.git
cd img-story-project-challenge

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model weights (internet required, ~2.5 GB, run once)
python download_models.py

# 5. Confirm offline readiness
HF_HUB_OFFLINE=1 python main.py images/street.jpg
```

---

## Usage

```bash
# Compare both pipelines on all images in images/ folder
HF_HUB_OFFLINE=1 python main.py images/

# Single image, both pipelines
HF_HUB_OFFLINE=1 python main.py images/street.jpg

# With length-controlled retry (up to 3 attempts per image)
HF_HUB_OFFLINE=1 python main.py images/ --fix-length --output results_fixlen.csv

# One continuous story across all images in sequence
HF_HUB_OFFLINE=1 python main.py images/ --multi

# Run baseline only (for testing)
python baseline.py images/feast_table.png
```

---

## Project Structure

```
img-story-project-challenge/
├── baseline.py          # Baseline: BLIP → Qwen2.5-0.5B story pipeline
├── seeing.py            # Module 7: Florence-2 structured vision extraction
├── context_builder.py   # Deterministic structured context assembly (no ML)
├── metric.py            # Module 10: CLIP + NLI + attribute + repetition + runtime
├── main.py              # CLI runner: comparison, length-control, multi-image story
├── download_models.py   # Pre-download all weights for offline use
├── requirements.txt     # Python dependencies
├── experiments.md       # Full experimental write-up with failure analysis & limitations
├── results.csv          # Normal run: baseline vs improved, all metrics
├── results_fixlen.csv   # Length-controlled run
├── vision.json          # Florence-2 structured outputs for each image
├── combined_story.json  # Multi-image story output (--multi mode)
└── images/
    ├── street.jpg
    ├── feast_table.png
    ├── magic_bridge.png
    ├── mossy_statue.png
    └── car_trip.png
```

---

## Visual Examples

### Images Used (Spirited Away frames)

| Image | Scene |
|---|---|
| `street.jpg` | Colourful spirit town street marketplace |
| `feast_table.png` | Chihiro watching her parents eat |
| `magic_bridge.png` | Haku casting paper charms from a bridge at dusk |
| `mossy_statue.png` | Chihiro standing beside a mossy stone spirit statue |
| `car_trip.png` | Chihiro in the back seat during the moving-day trip |

### Baseline vs Improved: Key Contrast

**Image: `feast_table.png`**

> **Baseline BLIP caption:** "a man and woman sitting at a table eating food"

> **Baseline story:** *"In a cozy café, two souls sat together in the dimly lit corner, sipping their steaming coffee. The woman's eyes sparkled with excitement as she savored her meal of homemade paella. The man watched silently while the woman indulged in her culinary masterpiece..."*
> — Hallucinated coffee, paella, café. None in the caption.

> **Florence-2 context:** "Scene: A restaurant interior. Characters: girl, man, woman. Objects: food dish, table, chair. Actions: sitting, eating, watching. Region: young girl standing behind table, looking at parents eating large portions."

> **Improved story:** *"In the warm glow of the restaurant, the girl stood quietly behind her parents, watching them devour plate after plate. The man reached eagerly for another dish while the woman laughed between bites. The child said nothing, standing still as the aromas filled the air around the crowded table..."*
> — Grounded: parents eating, child watching, restaurant setting.

---

<div align="center">

**Module 7 (Vision) + Module 10 (Evaluation) — Fully Local / Offline**

</div>
