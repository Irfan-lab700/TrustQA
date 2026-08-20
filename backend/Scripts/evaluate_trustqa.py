import json
import os
import re
import string
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

RESULTS_FILE = "results/trustqa_test1.json"
OUTPUT_FILE = "results/trustqa_tier1_tier2_nq_evaluation.json"

THRESHOLD = 0.55


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    SQuAD-style normalization.
    Used for EM/F1 comparison.
    """

    if text is None:
        return ""

    text = str(text).lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove articles
    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text
    )

    # Normalize whitespace
    text = " ".join(text.split())

    return text


# ============================================================
# EXACT MATCH
# ============================================================

def exact_match(prediction, gold_answers):

    prediction = normalize_text(prediction)

    for gold in gold_answers:
        if prediction == normalize_text(gold):
            return 1

    return 0


# ============================================================
# F1
# ============================================================

def f1_score(prediction, gold_answers):

    prediction_tokens = normalize_text(
        prediction
    ).split()

    if not prediction_tokens:
        return 0.0

    best_f1 = 0.0

    for gold in gold_answers:

        gold_tokens = normalize_text(
            gold
        ).split()

        if not gold_tokens:
            continue

        common = Counter(
            prediction_tokens
        ) & Counter(
            gold_tokens
        )

        num_same = sum(common.values())

        if num_same == 0:
            continue

        precision = (
            num_same /
            len(prediction_tokens)
        )

        recall = (
            num_same /
            len(gold_tokens)
        )

        f1 = (
            2 * precision * recall /
            (precision + recall)
        )

        best_f1 = max(
            best_f1,
            f1
        )

    return best_f1


# ============================================================
# LOAD RESULTS
# ============================================================

print("=" * 70)
print("TRUSTQA NOVEL APPROACH EVALUATION")
print("=" * 70)

print("\nLoading results...")

with open(
    RESULTS_FILE,
    "r",
    encoding="utf-8"
) as f:

    results = json.load(f)

print(
    f"Loaded {len(results)} results"
)


# ============================================================
# METRIC STORAGE
# ============================================================

total = len(results)

final_em_scores = []
final_f1_scores = []

tier1_em_scores = []
tier1_f1_scores = []

tier2_em_scores = []
tier2_f1_scores = []

entropies = []
correct_entropies = []
wrong_entropies = []

tier1_latencies = []
retrieval_latencies = []
tier2_generation_latencies = []
total_latencies = []

tier1_accepted = 0
tier2_triggered = 0

tier2_corrected = 0
tier2_regressions = 0

tier2_evaluated = 0

final_not_found = 0


# ============================================================
# PER-QUESTION EVALUATION
# ============================================================

for item in results:

    # --------------------------------------------------------
    # GOLD
    # --------------------------------------------------------

    gold_data = item.get(
        "gold_answers",
        {}
    )

    gold_answers = gold_data.get(
        "text",
        []
    )

    if isinstance(
        gold_answers,
        str
    ):
        gold_answers = [
            gold_answers
        ]


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    tier1_prediction = item.get(
        "tier1_prediction",
        ""
    )

    tier2_prediction = item.get(
        "tier2_prediction"
    )

    final_prediction = item.get(
        "final_prediction",
        ""
    )


    # --------------------------------------------------------
    # TIER DECISION
    # --------------------------------------------------------

    tier2_was_triggered = bool(
        item.get(
            "tier2_triggered",
            False
        )
    )


    # --------------------------------------------------------
    # EM / F1
    # --------------------------------------------------------

    tier1_em = exact_match(
        tier1_prediction,
        gold_answers
    )

    tier1_f1 = f1_score(
        tier1_prediction,
        gold_answers
    )

    final_em = exact_match(
        final_prediction,
        gold_answers
    )

    final_f1 = f1_score(
        final_prediction,
        gold_answers
    )


    tier1_em_scores.append(
        tier1_em
    )

    tier1_f1_scores.append(
        tier1_f1
    )

    final_em_scores.append(
        final_em
    )

    final_f1_scores.append(
        final_f1
    )


    # --------------------------------------------------------
    # TIER COUNTS
    # --------------------------------------------------------

    if tier2_was_triggered:

        tier2_triggered += 1

        tier2_evaluated += 1

        tier2_em = exact_match(
            tier2_prediction or "",
            gold_answers
        )

        tier2_f1 = f1_score(
            tier2_prediction or "",
            gold_answers
        )

        tier2_em_scores.append(
            tier2_em
        )

        tier2_f1_scores.append(
            tier2_f1
        )

        # Wrong Tier-1 -> Correct Tier-2
        if tier1_em == 0 and tier2_em == 1:

            tier2_corrected += 1

        # Correct Tier-1 -> Wrong Tier-2
        if tier1_em == 1 and tier2_em == 0:

            tier2_regressions += 1

    else:

        tier1_accepted += 1


    # --------------------------------------------------------
    # ENTROPY
    # --------------------------------------------------------

    entropy = item.get(
        "tier1_entropy"
    )

    if entropy is not None:

        entropy = float(
            entropy
        )

        entropies.append(
            entropy
        )

        if final_em == 1:

            correct_entropies.append(
                entropy
            )

        else:

            wrong_entropies.append(
                entropy
            )


    # --------------------------------------------------------
    # LATENCY
    # --------------------------------------------------------

    tier1_latency = item.get(
        "tier1_latency_sec",
        0.0
    )

    retrieval_latency = item.get(
        "retrieval_time_sec",
        0.0
    )

    tier2_generation_latency = item.get(
        "tier2_generation_time_sec",
        0.0
    )

    total_latency = item.get(
        "total_time_sec",
        0.0
    )


    tier1_latencies.append(
        float(tier1_latency)
    )

    retrieval_latencies.append(
        float(retrieval_latency)
    )

    tier2_generation_latencies.append(
        float(tier2_generation_latency)
    )

    total_latencies.append(
        float(total_latency)
    )


    # --------------------------------------------------------
    # NOT FOUND
    # --------------------------------------------------------

    if (
        str(final_prediction)
        .strip()
        .upper()
        == "NOT_FOUND"
    ):

        final_not_found += 1


# ============================================================
# HELPERS
# ============================================================

def average(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


# ============================================================
# FINAL METRICS
# ============================================================

final_em = average(
    final_em_scores
)

final_f1 = average(
    final_f1_scores
)

final_accuracy = final_em


tier1_avg_em = average(
    tier1_em_scores
)

tier1_avg_f1 = average(
    tier1_f1_scores
)

tier2_avg_em = average(
    tier2_em_scores
)

tier2_avg_f1 = average(
    tier2_f1_scores
)


# ============================================================
# TIER RATES
# ============================================================

tier1_accept_rate = (
    tier1_accepted / total
    if total else 0
)

tier2_trigger_rate = (
    tier2_triggered / total
    if total else 0
)


# ============================================================
# TIER-2 EFFECTIVENESS
# ============================================================

correction_rate = (
    tier2_corrected /
    tier2_triggered
    if tier2_triggered
    else 0
)

regression_rate = (
    tier2_regressions /
    tier2_triggered
    if tier2_triggered
    else 0
)


# ============================================================
# RETRIEVAL
# ============================================================

retrieval_calls = tier2_triggered

calls_avoided = tier1_accepted

calls_avoided_rate = (
    calls_avoided / total
    if total else 0
)


# ============================================================
# NOT FOUND
# ============================================================

not_found_rate = (
    final_not_found / total
    if total else 0
)


# ============================================================
# LATENCY
# ============================================================

avg_tier1_latency = average(
    tier1_latencies
)

avg_retrieval_latency = average(
    retrieval_latencies
)

avg_tier2_generation_latency = average(
    tier2_generation_latencies
)

avg_total_latency = average(
    total_latencies
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TRUSTQA RESULTS")
print("=" * 70)

print(
    f"\nTotal Questions       : {total}"
)

print(
    f"Final EM              : {final_em:.4f}"
)

print(
    f"Final F1              : {final_f1:.4f}"
)

print(
    f"Final Accuracy        : {final_accuracy:.4f}"
)


print("\n" + "-" * 70)
print("TIER DISTRIBUTION")
print("-" * 70)

print(
    f"Tier-1 Accepted       : {tier1_accepted}"
)

print(
    f"Tier-2 Triggered      : {tier2_triggered}"
)

print(
    f"Tier-1 Accept Rate    : "
    f"{tier1_accept_rate * 100:.2f}%"
)

print(
    f"Tier-2 Trigger Rate   : "
    f"{tier2_trigger_rate * 100:.2f}%"
)


print("\n" + "-" * 70)
print("TIER PERFORMANCE")
print("-" * 70)

print(
    f"Tier-1 Avg EM         : "
    f"{tier1_avg_em:.4f}"
)

print(
    f"Tier-1 Avg F1         : "
    f"{tier1_avg_f1:.4f}"
)

print(
    f"Tier-2 Avg EM         : "
    f"{tier2_avg_em:.4f}"
)

print(
    f"Tier-2 Avg F1         : "
    f"{tier2_avg_f1:.4f}"
)


print("\n" + "-" * 70)
print("TIER-2 EFFECTIVENESS")
print("-" * 70)

print(
    f"Wrong Tier-1 -> Correct Tier-2 "
    f": {tier2_corrected}"
)

print(
    f"Correction Rate       : "
    f"{correction_rate * 100:.2f}%"
)

print(
    f"Correct Tier-1 -> Wrong Tier-2 "
    f": {tier2_regressions}"
)

print(
    f"Regression Rate       : "
    f"{regression_rate * 100:.2f}%"
)


print("\n" + "-" * 70)
print("ENTROPY ANALYSIS")
print("-" * 70)

print(
    f"Avg Entropy           : "
    f"{average(entropies):.4f}"
)

print(
    f"Avg Entropy Correct   : "
    f"{average(correct_entropies):.4f}"
)

print(
    f"Avg Entropy Wrong     : "
    f"{average(wrong_entropies):.4f}"
)

print(
    f"Threshold             : "
    f"{THRESHOLD}"
)


print("\n" + "-" * 70)
print("RETRIEVAL")
print("-" * 70)

print(
    f"FAISS Retrieval Calls : "
    f"{retrieval_calls}"
)

print(
    f"Retrieval Call Rate   : "
    f"{tier2_trigger_rate * 100:.2f}%"
)

print(
    f"Calls Avoided         : "
    f"{calls_avoided}"
)

print(
    f"Calls Avoided Rate    : "
    f"{calls_avoided_rate * 100:.2f}%"
)


print("\n" + "-" * 70)
print("NOT_FOUND")
print("-" * 70)

print(
    f"Final NOT_FOUND       : "
    f"{final_not_found}"
)

print(
    f"Final NOT_FOUND Rate  : "
    f"{not_found_rate * 100:.2f}%"
)


print("\n" + "-" * 70)
print("LATENCY")
print("-" * 70)

print(
    f"Avg Tier-1 Latency    : "
    f"{avg_tier1_latency:.3f} sec"
)

print(
    f"Avg FAISS Latency     : "
    f"{avg_retrieval_latency:.3f} sec"
)

print(
    f"Avg Tier-2 Generation : "
    f"{avg_tier2_generation_latency:.3f} sec"
)

print(
    f"Avg Total Latency     : "
    f"{avg_total_latency:.3f} sec"
)


# ============================================================
# SAVE EVALUATION
# ============================================================

evaluation = {

    "total_questions": total,

    "final_metrics": {
        "exact_match": final_em,
        "f1": final_f1,
        "accuracy": final_accuracy
    },

    "tier_distribution": {
        "tier1_accepted": tier1_accepted,
        "tier2_triggered": tier2_triggered,
        "tier1_accept_rate": tier1_accept_rate,
        "tier2_trigger_rate": tier2_trigger_rate
    },

    "tier_performance": {
        "tier1_avg_em": tier1_avg_em,
        "tier1_avg_f1": tier1_avg_f1,
        "tier2_avg_em": tier2_avg_em,
        "tier2_avg_f1": tier2_avg_f1
    },

    "tier2_effectiveness": {
        "wrong_tier1_correct_tier2": tier2_corrected,
        "correction_rate": correction_rate,
        "correct_tier1_wrong_tier2": tier2_regressions,
        "regression_rate": regression_rate
    },

    "entropy": {
        "average": average(entropies),
        "average_correct": average(correct_entropies),
        "average_wrong": average(wrong_entropies),
        "threshold": THRESHOLD
    },

    "retrieval": {
        "faiss_calls": retrieval_calls,
        "retrieval_call_rate": tier2_trigger_rate,
        "calls_avoided": calls_avoided,
        "calls_avoided_rate": calls_avoided_rate
    },

    "not_found": {
        "count": final_not_found,
        "rate": not_found_rate
    },

    "latency": {
        "avg_tier1_sec": avg_tier1_latency,
        "avg_retrieval_sec": avg_retrieval_latency,
        "avg_tier2_generation_sec": avg_tier2_generation_latency,
        "avg_total_sec": avg_total_latency
    }
}


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        evaluation,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\n")
print("=" * 70)
print("EVALUATION SAVED")
print("=" * 70)

print(
    OUTPUT_FILE
)