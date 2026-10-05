"""Image -> Story Challenge: Main Runner (Modules 7 & 10).

Usage examples:
  # Compare both pipelines on all images in the images/ folder:
  python main.py images/

  # Single image, both pipelines:
  python main.py images/street.jpg --mode both

  # Length-controlled retry mode:
  python main.py images/ --fix-length

  # Multi-image story (one story across all images in sequence order):
  python main.py images/ --multi

  # Offline confirmation:
  HF_HUB_OFFLINE=1 python main.py images/
"""
import os, sys, time, json, csv, argparse
from pathlib import Path

import torch

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MULTI_TARGET_WORDS = 250
MULTI_MAX_NEW_TOKENS = 350

import baseline, seeing, context_builder, metric


def find_images(paths):
    """Expand file/directory arguments to sorted image paths."""
    found = {}
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files = [path]
        else:
            files = []
        for f in files:
            found[str(f)] = f
    return [str(f) for f in sorted(found.values(), key=lambda f: f.name.lower())]


def _ok_row(name, variant, path, context, story, see_s, story_s):
    """One successful pipeline run. Evaluation runtime is reported separately."""
    m = metric.score(path, context, story)
    m["attribute_conflict"] = " ".join(m["attribute_conflict"])
    return {"image": name, "variant": variant, "status": "ok", "error": "",
            "context": context, "story": story,
            "see_s": round(see_s, 2), "story_s": round(story_s, 2),
            "generation_s": round(see_s + story_s, 2), **m}


def _error_row(name, variant, err):
    """A failed run is recorded honestly; no metric values are invented."""
    return {"image": name, "variant": variant, "status": "error", "error": str(err)}


def write_rows_csv(rows, path):
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)


def summarize(rows, variants=("A_baseline", "B_improved")):
    """Per-variant means over SUCCESSFUL runs only."""
    out = {}
    for v in variants:
        mine = [r for r in rows if r["variant"] == v]
        ok = [r for r in mine if r.get("status") == "ok"]
        n = len(ok)
        s = {"successful": n, "failed": len(mine) - n, "total": len(mine)}
        if n:
            mean = lambda k: sum(float(x[k]) for x in ok) / n
            s.update({
                "mean_grounding": round(mean("grounding_score"), 4),
                "mean_clip": round(mean("clip_image_story_mean"), 4),
                "mean_nli_contradiction": round(mean("nli_contra_mean"), 4),
                "mean_repetition_rate": round(mean("repetition_rate"), 4),
                "length_valid": sum(bool(x["length_valid"]) for x in ok),
                "grounding_pass": sum(bool(x["grounding_pass"]) for x in ok),
                "mean_see_s": round(mean("see_s"), 2),
                "mean_story_s": round(mean("story_s"), 2),
                "mean_generation_s": round(mean("generation_s"), 2),
                "mean_eval_total_s": round(mean("eval_total_s"), 2),
            })
        out[v] = s
    return out


def run_comparison(image_paths, fix_length=False, output_csv="results.csv", vision_json="vision.json"):
    """Run baseline and improved pipeline on all images. One failing image never stops the others."""
    rows, vision = [], {}
    baseline.warm_up()  # pre-load BLIP + Qwen before timing begins

    for p in image_paths:
        name = os.path.basename(p)
        print(f"\n[{image_paths.index(p)+1}/{len(image_paths)}] Processing {name}...")

        # A: Baseline  (BLIP -> Qwen)
        try:
            a = baseline.run(p, fix_length)
            rows.append(_ok_row(name, "A_baseline", p, a["caption"], a["story"], a["caption_s"], a["story_s"]))
            print(f"  A_baseline | cap: \"{a['caption'][:60]}\" | words: {a['word_count']} | valid: {a['length_valid']}")
        except Exception as e:
            rows.append(_error_row(name, "A_baseline", e))
            print(f"  A_baseline error: {e}")

        # B: Improved  (Florence-2 -> structured context -> Qwen)
        try:
            desc = seeing.describe(p)
            vision[name] = {k: v for k, v in desc.items() if k not in ("runtime_s", "model_load_s")}
            ctx = context_builder.build_single_image_context(desc)
            baseline.warm_up(blip=False)       # Qwen pre-loaded before story timer
            t = time.perf_counter()
            story = baseline.write_story(ctx, fix_length)
            story_s = time.perf_counter() - t
            rows.append(_ok_row(name, "B_improved", p, ctx, story, desc["runtime_s"], story_s))
            wc = len(story.split())
            print(f"  B_improved | see: {desc['runtime_s']}s | words: {wc} | valid: {baseline.is_length_valid(wc)}")
        except Exception as e:
            rows.append(_error_row(name, "B_improved", e))
            print(f"  B_improved error: {e}")

    write_rows_csv(rows, output_csv)
    print(f"\nResults saved to {output_csv}")

    with open(vision_json, "w", encoding="utf-8") as f:
        json.dump(vision, f, indent=2, ensure_ascii=False)
    print(f"Vision JSON saved to {vision_json}")

    summary = summarize(rows)
    print("\n=== Summary ===")
    for v, s in summary.items():
        print(f"\n{v}:")
        for k, val in s.items():
            print(f"  {k}: {val}")
    return rows, summary


def run_multi_image_story(image_paths):
    """One continuous story across all images (in filename order) + continuity proxy."""
    try:
        descs, failed, total_see_s = [], [], 0.0
        for p in image_paths:
            try:
                t0 = time.perf_counter()
                descs.append(seeing.describe(p))
                total_see_s += time.perf_counter() - t0
            except Exception as e:
                failed.append({"image": os.path.basename(p), "error": str(e)})

        if not descs:
            return {"images": [os.path.basename(p) for p in image_paths],
                    "frames_failed": failed, "error": "multi failed: no frame could be described"}

        seq_ctx = context_builder.build_sequence_context(descs)
        prompt = context_builder.build_story_prompt(seq_ctx, len(descs), target_words=MULTI_TARGET_WORDS)
        tok, model = baseline._qwen()
        t1 = time.perf_counter()
        msgs = [{"role": "system", "content": "You are a creative storyteller."},
                {"role": "user", "content": prompt}]
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt_text, return_tensors="pt")
        torch.manual_seed(baseline.SEED)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MULTI_MAX_NEW_TOKENS,
                                  do_sample=False, repetition_penalty=1.05)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        story = baseline._fit_length(raw, hi=10**6)
        story_s = time.perf_counter() - t1

        return {
            "images": [d["image_id"] for d in descs],
            "frames_failed": failed,
            "target_words": MULTI_TARGET_WORDS,
            "story": story,
            "cut_off_tail_removed": story != raw,
            "word_count": len(story.split()),
            "see_s": round(total_see_s, 2),
            "story_s": round(story_s, 2),
            "continuity": metric.continuity_score(descs, story),
            "vision_json": {d["image_id"]: {k: v for k, v in d.items()
                            if k not in ("runtime_s", "model_load_s")} for d in descs},
        }
    except Exception as e:
        return {"images": [os.path.basename(p) for p in image_paths], "error": f"multi failed: {e}"}


def main():
    parser = argparse.ArgumentParser(description="Image -> Story Challenge pipeline runner")
    parser.add_argument("targets", nargs="+", help="Image file(s) or directory")
    parser.add_argument("--fix-length", action="store_true",
                        help="Use up to 3 prompt-retry attempts for 80-120 word stories")
    parser.add_argument("--multi", action="store_true",
                        help="Generate ONE story across all images in sequence order")
    parser.add_argument("--output", default="results.csv", help="Output CSV path (default: results.csv)")
    parser.add_argument("--vision-json", default="vision.json", help="Vision JSON output path")
    args = parser.parse_args()

    image_paths = find_images(args.targets)
    if not image_paths:
        print("No images found.")
        sys.exit(1)
    print(f"Found {len(image_paths)} image(s): {[os.path.basename(p) for p in image_paths]}")

    if args.multi:
        result = run_multi_image_story(image_paths)
        print("\n=== Multi-Image Story ===")
        print(f"Images: {result.get('images', [])}")
        print(f"Word count: {result.get('word_count', '?')}")
        print(f"\nStory:\n{result.get('story', result.get('error', 'Failed'))}")
        print(f"\nContinuity: {result.get('continuity', {})}")
        with open("combined_story.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("\nSaved to combined_story.json")
    else:
        run_comparison(image_paths, fix_length=args.fix_length,
                       output_csv=args.output, vision_json=args.vision_json)


if __name__ == "__main__":
    main()
