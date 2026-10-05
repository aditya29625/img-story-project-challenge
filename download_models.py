"""Pre-download all required model weights for fully offline execution.

Models downloaded (~2.5 GB total):
  - Salesforce/blip-image-captioning-base  (Baseline vision)
  - Qwen/Qwen2.5-0.5B-Instruct            (Story generation, both pipelines)
  - florence-community/Florence-2-base     (Improved vision, Module 7)
  - openai/clip-vit-base-patch32           (Evaluation, Module 10)
  - cross-encoder/nli-MiniLM2-L6-H768     (Evaluation, Module 10)

Run this ONCE before the exam (requires internet). After this, set:
  HF_HUB_OFFLINE=1  python main.py images/
"""
import sys, time
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM,
    AutoProcessor, AutoModelForCausalLM as AutoVLM,
    CLIPModel, CLIPProcessor,
    AutoModelForSequenceClassification,
)

MODELS = [
    ("BLIP (baseline vision)",
     lambda: (BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
               BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base"))),
    ("Qwen2.5-0.5B-Instruct (story generation)",
     lambda: (AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"),
               AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"))),
    ("Florence-2-base (improved vision, Module 7)",
     lambda: (AutoProcessor.from_pretrained("florence-community/Florence-2-base", trust_remote_code=True),
               AutoVLM.from_pretrained("florence-community/Florence-2-base", trust_remote_code=True))),
    ("CLIP ViT-B/32 (evaluation, Module 10)",
     lambda: (CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
               CLIPModel.from_pretrained("openai/clip-vit-base-patch32"))),
    ("NLI MiniLM (evaluation, Module 10)",
     lambda: (AutoTokenizer.from_pretrained("cross-encoder/nli-MiniLM2-L6-H768"),
               AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-MiniLM2-L6-H768"))),
]


def main():
    print("=" * 60)
    print("Image -> Story Challenge: Model Pre-downloader")
    print("=" * 60)
    errors = []
    for i, (name, loader) in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Downloading: {name}...")
        t = time.perf_counter()
        try:
            loader()
            elapsed = round(time.perf_counter() - t, 1)
            print(f"  ✓ Done in {elapsed}s")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            errors.append(name)

    print("\n" + "=" * 60)
    if errors:
        print(f"✗ {len(errors)} model(s) failed to download:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("✓ All models cached. Offline inference is ready.")
        print("  Run: HF_HUB_OFFLINE=1 python main.py images/")


if __name__ == "__main__":
    main()
