"""Step 2: measure accuracy on the Labeled Faces in the Wild (LFW) pairs, and set the decision threshold.

Run it from the project root with the virtual environment active (after src/download_lfw.py):
    python src/02_evaluate_lfw.py

The first run turns about 7,700 photos into embeddings (roughly 5 to 15 minutes) and saves them in
data/processed (ignored by Git), so later runs take seconds. Only anonymous totals and charts are written
to reports/, which is safe to publish. No LFW photo or per-person result is ever saved there.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")                      # draw charts to files, with no window
import matplotlib.pyplot as plt
import numpy as np

from evaluation import (cross_validate, equal_error_rate, error_rates, parse_pairs, summarise,
                        threshold_for_far, wilson_interval)
from face_engine import FaceEngine
from matching import l2_normalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LFW_DIR = PROJECT_ROOT / "data" / "raw" / "lfw"
PAIRS_FILE = PROJECT_ROOT / "data" / "raw" / "pairs.txt"
CACHE_FILE = PROJECT_ROOT / "data" / "processed" / "lfw_embeddings.npz"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET_FAR = 0.001            # the deployed threshold lets through at most 0.1% of different-person pairs


def load_cache():
    if CACHE_FILE.exists():
        data = np.load(CACHE_FILE, allow_pickle=False)
        embeddings = {str(k): v for k, v in zip(data["keys"], data["vectors"])}
        return embeddings, {str(k) for k in data["failed"]}
    return {}, set()


def save_cache(embeddings, failed):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    keys = np.array(list(embeddings), dtype=str)
    vectors = np.stack(list(embeddings.values())) if embeddings else np.zeros((0, 128), dtype=np.float32)
    np.savez_compressed(CACHE_FILE, keys=keys, vectors=vectors, failed=np.array(sorted(failed), dtype=str))


def embed_all(engine, relpaths):
    """Embedding for every photo. Photos where no face is found are counted as failures."""
    embeddings, failed = load_cache()
    todo = [p for p in relpaths if p not in embeddings and p not in failed]
    if todo:
        print(f"Computing embeddings for {len(todo):,} photos (first run only)...")
    started = time.time()
    for done, relpath in enumerate(todo, start=1):
        image = cv2.imread(str(LFW_DIR / relpath))
        observation = engine.observe(image, prefer_centre=True) if image is not None else None
        if observation is None:
            failed.add(relpath)
        else:
            embeddings[relpath] = observation.embedding
        if done % 500 == 0:
            rate = done / (time.time() - started)
            print(f"   {done:,} / {len(todo):,}   about {int((len(todo) - done) / rate / 60) + 1} min left", flush=True)
            save_cache(embeddings, failed)                       # save progress, so an interruption loses little
    if todo:
        save_cache(embeddings, failed)
    return embeddings, failed


def score_pairs(pairs, embeddings):
    """Cosine similarity for every pair where both photos had a detectable face."""
    unit = {path: l2_normalize(vector) for path, vector in embeddings.items()}
    folds = [([], []) for _ in range(10)]
    skipped = {"same": 0, "different": 0}
    for pair in pairs:
        if pair.first not in unit or pair.second not in unit:
            skipped["same" if pair.same else "different"] += 1
            continue
        score = float(np.dot(unit[pair.first], unit[pair.second]))
        folds[pair.fold][0 if pair.same else 1].append(score)
    return [(np.array(s), np.array(d)) for s, d in folds], skipped


def make_charts(same, different, threshold):
    REPORTS_DIR.mkdir(exist_ok=True)
    # Chart 1: how the scores of same-person and different-person pairs are spread out
    low = min(same.min(), different.min())
    high = max(same.max(), different.max())
    if high - low < 1e-6:                                   # guard: all scores identical would give zero-width bins
        high = low + 1e-3
    bins = np.linspace(low, high, 61)                       # the same bins for both groups, so they can be compared
    plt.figure(figsize=(8, 4.5))
    plt.hist(different, bins=bins, alpha=0.7, label="Different people", color="#d95f02")
    plt.hist(same, bins=bins, alpha=0.7, label="Same person", color="#1b9e77")
    plt.axvline(threshold, color="black", linestyle="--", label=f"Chosen threshold ({threshold:.2f})")
    plt.xlabel("Cosine similarity score")
    plt.ylabel("Number of photo pairs")
    plt.title("Scores on the LFW benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "score_distributions.png", dpi=150)
    plt.close()

    # Chart 2: the ROC (Receiver Operating Characteristic) curve
    thresholds = np.unique(np.concatenate([same, different]))
    far, frr = error_rates(same, different, thresholds)
    plt.figure(figsize=(6, 5))
    plt.plot(np.maximum(far, 1e-4), 1 - frr, color="#1b9e77")
    plt.xscale("log")
    plt.xlabel("False accept rate (log scale)")
    plt.ylabel("True accept rate")
    plt.title("ROC curve on the LFW benchmark")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=150)
    plt.close()


def main() -> None:
    for path in (PAIRS_FILE, LFW_DIR):
        if not path.exists():
            sys.exit(f"Missing {path}. Run first: python src/download_lfw.py")

    pairs = parse_pairs(PAIRS_FILE.read_text(encoding="utf-8"))
    relpaths = sorted({p for pair in pairs for p in (pair.first, pair.second)})
    print(f"{len(pairs):,} pairs, {len(relpaths):,} distinct photos")

    engine = FaceEngine(MODELS_DIR)
    embeddings, failed = embed_all(engine, relpaths)
    folds, skipped_by_kind = score_pairs(pairs, embeddings)
    skipped = skipped_by_kind["same"] + skipped_by_kind["different"]

    same = np.concatenate([s for s, _ in folds])
    different = np.concatenate([d for _, d in folds])
    evaluated = len(same) + len(different)
    print(f"\nPairs scored: {evaluated:,}   skipped (no face found): {skipped:,}   "
          f"photos without a detectable face: {len(failed):,} of {len(relpaths):,}")

    eer, eer_threshold = equal_error_rate(same, different)
    threshold = threshold_for_far(different, TARGET_FAR)
    far_pooled, frr_pooled = error_rates(same, different, threshold)
    false_accepts = int(round(far_pooled[0] * len(different)))
    far_low, far_high = wilson_interval(false_accepts, len(different))
    false_rejects = int(round(frr_pooled[0] * len(same)))
    frr_low, frr_high = wilson_interval(false_rejects, len(same))

    # A same-person pair where no face was found can never be verified, so it is a rejection in practice.
    # (A different-person pair with no face found cannot be falsely accepted, so FAR is unaffected.)
    total_same = len(same) + skipped_by_kind["same"]
    end_to_end_false_rejects = int(round(frr_pooled[0] * len(same))) + skipped_by_kind["same"]
    end_to_end_frr = end_to_end_false_rejects / total_same

    print(f"\nEqual error rate (EER): {eer:.2%}  at threshold {eer_threshold:.3f}")
    print(f"Chosen threshold for a false accept rate of at most {TARGET_FAR:.1%}: {threshold:.3f}")
    print(f"   false accept rate: {far_pooled[0]:.3%}  ({false_accepts} of {len(different):,}; 95% range {far_low:.3%} to {far_high:.3%})")
    print(f"   false reject rate: {frr_pooled[0]:.2%}  ({false_rejects} of {len(same):,}; 95% range {frr_low:.2%} to {frr_high:.2%})")
    print(f"   end-to-end false reject rate, counting same-person pairs where no face was found: "
          f"{end_to_end_frr:.2%}  ({end_to_end_false_rejects} of {total_same:,})")

    report = {"target_far": TARGET_FAR}
    print("\nHeld-out results (threshold chosen on 9 folds, measured on the 10th):")
    for target in (0.01, 0.001):
        result = cross_validate(folds, target)
        far_mean, far_sd = summarise(result["far"])
        frr_mean, frr_sd = summarise(result["frr"])
        print(f"   target FAR {target:.1%}:  FAR {far_mean:.3%} +/- {far_sd:.3%}   FRR {frr_mean:.2%} +/- {frr_sd:.2%}")
        report[f"cross_validated_far_{target}"] = {"far_mean": far_mean, "far_sd": far_sd, "frr_mean": frr_mean, "frr_sd": frr_sd}
    accuracy_mean, accuracy_sd = summarise(cross_validate(folds, TARGET_FAR)["accuracy"])
    print(f"   accuracy (standard LFW protocol): {accuracy_mean:.2%} +/- {accuracy_sd:.2%}")

    smallest_far = 1 / len(different)
    print(f"\nWith {len(different):,} different-person pairs, the smallest measurable false accept rate is "
          f"about {smallest_far:.3%}. Claims of 'one in a million' cannot be made from this test.")

    REPORTS_DIR.mkdir(exist_ok=True)
    report.update({
        "metric": "cosine similarity",
        "model": "face_recognition_sface_2021dec",
        "dataset": "LFW, standard 10-fold pairs",
        "pairs_scored": evaluated, "pairs_skipped": skipped,
        "pairs_skipped_same": skipped_by_kind["same"], "pairs_skipped_different": skipped_by_kind["different"],
        "end_to_end_frr": end_to_end_frr,
        "photos_without_face": len(failed), "photos_total": len(relpaths),
        "equal_error_rate": eer, "eer_threshold": eer_threshold,
        "threshold": threshold,
        "pooled_far": far_pooled[0], "pooled_far_95_range": [far_low, far_high],
        "pooled_frr": frr_pooled[0], "pooled_frr_95_range": [frr_low, frr_high],
        "accuracy_cross_validated": {"mean": accuracy_mean, "sd": accuracy_sd},
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    (REPORTS_DIR / "threshold.json").write_text(json.dumps({
        "threshold": threshold, "target_far": TARGET_FAR, "metric": "cosine similarity",
        "model": report["model"], "dataset": report["dataset"], "generated": report["generated"],
    }, indent=2), encoding="utf-8")
    (REPORTS_DIR / "lfw_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_charts(same, different, threshold)
    print(f"\nSaved to {REPORTS_DIR}: threshold.json, lfw_evaluation.json, roc_curve.png, score_distributions.png")
    print("These files hold anonymous totals only, so they are safe to commit.")


if __name__ == "__main__":
    main()
