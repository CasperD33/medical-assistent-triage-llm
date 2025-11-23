"""
Evaluation script for the triage LLM.

Runs different modes on the synthetic eval split and prints:
- number of evaluated cases
- accuracy
- confusion matrix (rows = true label, columns = predicted label)
- a few example errors
- sklearn confusion matrix + classification report

Modes:
- 'baseline' : instruction-only zero-shot
- 'fewshot'  : baseline + few-shot examples from train split
"""

import json
from pathlib import Path

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from triage_model import (
    load_cases,
    split_cases,
    triage_baseline,
    triage_fewshot,
)

# All possible NTS labels
LABELS = ["U0", "U1", "U2", "U3", "U4", "U5"]

# Where to save detailed results
RESULTS_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
    / "3_results"
)
RESULTS_DIR.mkdir(exist_ok=True)


def save_per_case(stats: dict, filename: str) -> None:
    """
    Save the per-case results of one mode to 3_results/filename.

    Each entry has:
      - id
      - question
      - true
      - pred
      - correct
    """
    path = RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats["per_case"], f, ensure_ascii=False, indent=2)
    print(f"Saved per-case results to {path}")


def print_confusion_from_cm(cm, labels: list[str]) -> None:
    """
    Nicely print a confusion matrix (2D array) with label headers.
    Rows = true labels, columns = predicted labels.
    """
    print("\nConfusion matrix (rows = true, columns = predicted):")
    col_width = 6
    header = ["true\\pred"] + labels
    print("".join(f"{h:>{col_width}}" for h in header))
    for i, true in enumerate(labels):
        row_vals = [true] + [str(cm[i, j]) for j in range(len(labels))]
        print("".join(f"{v:>{col_width}}" for v in row_vals))


def plot_confusion_matrix_sklearn(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    title: str,
    filename: str,
) -> None:
    """
    Plot a confusion matrix using sklearn's confusion_matrix and save it as PNG.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    # annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)

    plt.tight_layout()

    path = RESULTS_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to {path}")


def evaluate_mode(
    mode: str,
    train_cases: list[dict],
    eval_cases: list[dict],
    max_cases: int | None = None,
) -> dict:
    """
    Evaluate one setup (baseline or fewshot) on the eval split.

    Args:
        mode: one of {"baseline", "fewshot"}
        train_cases: list of train-split cases (needed for few-shot mode)
        eval_cases: list of eval-split cases (never used as examples)
        max_cases: if not None, limit evaluation to at most this many cases

    Returns:
        dict with summary statistics:
        {
            "mode": str,
            "n_eval": int,
            "n_correct": int,
            "failed": int,
            "accuracy": float,
            "y_true": list[str],
            "y_pred": list[str],
            "per_case": list[dict]
        }
    """
    if mode not in {"baseline", "fewshot"}:
        raise ValueError("mode must be 'baseline' or 'fewshot'")

    if not eval_cases:
        print(f"No eval cases found (no cases with split == 'eval') for mode={mode}.")
        return {
            "mode": mode,
            "n_eval": 0,
            "n_correct": 0,
            "failed": 0,
            "accuracy": 0.0,
            "y_true": [],
            "y_pred": [],
            "per_case": [],
        }

    # Optionally limit number of cases for performance reasons
    if max_cases is not None:
        eval_cases = eval_cases[:max_cases]

    n_correct = 0
    n_total = 0
    failed = 0

    y_true: list[str] = []
    y_pred: list[str] = []
    per_case: list[dict] = []

    print(f"\n--- Evaluating mode = {mode} on {len(eval_cases)} cases ---")

    for i, case in enumerate(eval_cases, start=1):
        true_label = case["nts_code"]
        question = case["question_en"]
        case_id = case.get("id", f"case_{i:03d}")

        try:
            if mode == "baseline":
                result = triage_baseline(question)
            elif mode == "fewshot":
                result = triage_fewshot(question, train_cases, max_examples=12)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        except Exception as e:
            print(f"[WARN] Case {case_id}: model call failed: {e}")
            failed += 1
            continue

        pred_label = str(result.get("label", "")).strip()

        if pred_label not in LABELS:
            print(
                f"[WARN] Case {case_id}: invalid predicted label {pred_label!r}, "
                f"skipping."
            )
            failed += 1
            continue

        if true_label not in LABELS:
            print(
                f"[WARN] Case {case_id}: invalid true label {true_label!r}, "
                f"skipping."
            )
            failed += 1
            continue

        y_true.append(true_label)
        y_pred.append(pred_label)

        n_total += 1
        correct = (pred_label == true_label)
        if correct:
            n_correct += 1

        per_case.append(
            {
                "id": case_id,
                "question": question,
                "true": true_label,
                "pred": pred_label,
                "correct": correct,
            }
        )

    accuracy = n_correct / n_total if n_total > 0 else 0.0

    # Print summary
    print(f"\nMode: {mode}")
    print(f"Evaluated cases: {n_total} (skipped/failed: {failed})")
    print(f"Correct: {n_correct}")
    print(f"Accuracy: {accuracy:.3f}")

    # sklearn confusion matrix + pretty print
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print_confusion_from_cm(cm, LABELS)

    # Print a few example errors
    errors = [c for c in per_case if not c["correct"]]
    if errors:
        print(f"\nExample errors for mode={mode}:")
        for ex in errors[:5]:
            print(
                f"- {ex['id']}: true={ex['true']}, pred={ex['pred']} | "
                f"question={ex['question']}"
            )

    return {
        "mode": mode,
        "n_eval": n_total,
        "n_correct": n_correct,
        "failed": failed,
        "accuracy": accuracy,
        "y_true": y_true,
        "y_pred": y_pred,
        "per_case": per_case,
    }


if __name__ == "__main__":
    # IMPORTANT: keep this small if your machine is slow.
    # You can start with 5 or 10 to test, then increase if it runs okay.
    MAX_CASES = 88

    # Load once and reuse
    all_cases = load_cases()
    train_cases, eval_cases = split_cases(all_cases)

    print(
        f"Loaded {len(all_cases)} cases "
        f"({len(train_cases)} train / {len(eval_cases)} eval)"
    )

    # Baseline zero-shot
    baseline_stats = evaluate_mode(
        "baseline",
        train_cases=train_cases,
        eval_cases=eval_cases,
        max_cases=MAX_CASES,
    )

    # Few-shot prompting
    fewshot_stats = evaluate_mode(
        "fewshot",
        train_cases=train_cases,
        eval_cases=eval_cases,
        max_cases=MAX_CASES,
    )

    # Save detailed per-case JSON
    save_per_case(baseline_stats, "baseline_per_case.json")
    save_per_case(fewshot_stats, "fewshot_per_case.json")

    # --- sklearn metrics: classification reports ---
    print("\n=== sklearn metrics: baseline ===")
    report_base = classification_report(
        baseline_stats["y_true"],
        baseline_stats["y_pred"],
        labels=LABELS,
        target_names=LABELS,
        digits=3,
        zero_division=0,
    )
    print(report_base)

    print("\n=== sklearn metrics: few-shot ===")
    report_fs = classification_report(
        fewshot_stats["y_true"],
        fewshot_stats["y_pred"],
        labels=LABELS,
        target_names=LABELS,
        digits=3,
        zero_division=0,
    )
    print(report_fs)

    # Save classification reports to files
    (RESULTS_DIR / "baseline_classification_report.txt").write_text(
        report_base, encoding="utf-8"
    )
    (RESULTS_DIR / "fewshot_classification_report.txt").write_text(
        report_fs, encoding="utf-8"
    )

    # Save confusion matrix plots
    plot_confusion_matrix_sklearn(
        baseline_stats["y_true"],
        baseline_stats["y_pred"],
        LABELS,
        title="Confusion Matrix — Baseline",
        filename="confusion_baseline.png",
    )

    plot_confusion_matrix_sklearn(
        fewshot_stats["y_true"],
        fewshot_stats["y_pred"],
        LABELS,
        title="Confusion Matrix — Few-shot",
        filename="confusion_fewshot.png",
    )

    print("\n=== Summary ===")
    print(
        f"Baseline accuracy: {baseline_stats['accuracy']:.3f} "
        f"on {baseline_stats['n_eval']} cases "
        f"(failed: {baseline_stats['failed']})"
    )
    print(
        f"Few-shot accuracy: {fewshot_stats['accuracy']:.3f} "
        f"on {fewshot_stats['n_eval']} cases "
        f"(failed: {fewshot_stats['failed']})"
    )
