import json
import re
import string
from collections import Counter


# ============================================================
# FILE
# ============================================================

TIER2_FILE = "results/trustqa_tier2.json"

F1_THRESHOLD = 0.40
ENTROPY_THRESHOLD = 0.55


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    """
    Normalize answer for token-level F1.

    - lowercase
    - remove punctuation
    - normalize whitespace
    """
    text = str(text).lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Normalize whitespace
    text = " ".join(text.split())

    return text


# ============================================================
# TOKEN F1
# ============================================================

def f1_score(prediction, gold):
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)

    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


# ============================================================
# MAX F1 AGAINST MULTIPLE GOLD ANSWERS
# ============================================================

def max_f1(prediction, gold_answers):

    scores = []

    for gold in gold_answers:

        score = f1_score(
            prediction,
            gold
        )

        scores.append(score)

    if not scores:
        return 0.0

    return max(scores)


# ============================================================
# LOAD TIER-2 RESULTS
# ============================================================

print("Loading Tier-2 results...")

with open(
    TIER2_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


print(
    "Loaded:",
    len(data),
    "results"
)


# ============================================================
# METRICS
# ============================================================

baseline_correct = 0
baseline_wrong = 0

rag_correct = 0
rag_wrong = 0

baseline_entropies_correct = []
baseline_entropies_wrong = []

rag_triggered = 0

baseline_wrong_to_rag_correct = 0
baseline_correct_to_rag_wrong = 0

both_correct = 0
both_wrong = 0

triggered_correct = 0
triggered_wrong = 0

fixed_cases = []
broken_cases = []
still_wrong_cases = []


# ============================================================
# EVALUATE
# ============================================================

for item in data:

    question = item["question"]

    gold_answers = item["gold_answers"]

    baseline_prediction = item.get(
        "baseline_prediction",
        ""
    )

    final_prediction = item.get(
        "final_prediction",
        ""
    )

    entropy = item.get(
        "baseline_entropy",
        0.0
    )

    rag_trigger = item.get(
        "rag_trigger",
        False
    )


    # --------------------------------------------------------
    # BASELINE F1
    # --------------------------------------------------------

    baseline_f1 = max_f1(
        baseline_prediction,
        gold_answers
    )

    baseline_is_correct = (
        baseline_f1 >= F1_THRESHOLD
    )


    # --------------------------------------------------------
    # RAG / FINAL F1
    # --------------------------------------------------------

    rag_f1 = max_f1(
        final_prediction,
        gold_answers
    )

    rag_is_correct = (
        rag_f1 >= F1_THRESHOLD
    )


    # --------------------------------------------------------
    # BASELINE METRICS
    # --------------------------------------------------------

    if baseline_is_correct:

        baseline_correct += 1

        baseline_entropies_correct.append(
            entropy
        )

    else:

        baseline_wrong += 1

        baseline_entropies_wrong.append(
            entropy
        )


    # --------------------------------------------------------
    # RAG METRICS
    # --------------------------------------------------------

    if rag_is_correct:

        rag_correct += 1

    else:

        rag_wrong += 1


    # --------------------------------------------------------
    # RAG ACTIVATION
    # --------------------------------------------------------

    if rag_trigger:

        rag_triggered += 1


        if rag_is_correct:

            triggered_correct += 1

        else:

            triggered_wrong += 1


    # --------------------------------------------------------
    # TRANSITIONS
    # --------------------------------------------------------

    if not baseline_is_correct and rag_is_correct:

        baseline_wrong_to_rag_correct += 1

        fixed_cases.append(
            {
                "question": question,
                "baseline": baseline_prediction,
                "rag": final_prediction,
                "gold": gold_answers,
                "baseline_f1": round(
                    baseline_f1,
                    3
                ),
                "rag_f1": round(
                    rag_f1,
                    3
                ),
                "entropy": entropy
            }
        )


    elif baseline_is_correct and not rag_is_correct:

        baseline_correct_to_rag_wrong += 1

        broken_cases.append(
            {
                "question": question,
                "baseline": baseline_prediction,
                "rag": final_prediction,
                "gold": gold_answers,
                "baseline_f1": round(
                    baseline_f1,
                    3
                ),
                "rag_f1": round(
                    rag_f1,
                    3
                ),
                "entropy": entropy
            }
        )


    elif baseline_is_correct and rag_is_correct:

        both_correct += 1


    else:

        both_wrong += 1

        still_wrong_cases.append(
            {
                "question": question,
                "baseline": baseline_prediction,
                "rag": final_prediction,
                "gold": gold_answers,
                "baseline_f1": round(
                    baseline_f1,
                    3
                ),
                "rag_f1": round(
                    rag_f1,
                    3
                ),
                "entropy": entropy
            }
        )


# ============================================================
# CALCULATIONS
# ============================================================

total = len(data)

baseline_accuracy = (
    baseline_correct / total
    if total
    else 0
)

rag_accuracy = (
    rag_correct / total
    if total
    else 0
)

rag_activation = (
    rag_triggered / total
    if total
    else 0
)

wrong_reduction = (
    baseline_wrong - rag_wrong
)

wrong_to_correct_rate = (
    baseline_wrong_to_rag_correct
    / baseline_wrong
    if baseline_wrong
    else 0
)

correct_to_wrong_rate = (
    baseline_correct_to_rag_wrong
    / baseline_correct
    if baseline_correct
    else 0
)

avg_entropy_correct = (
    sum(baseline_entropies_correct)
    / len(baseline_entropies_correct)
    if baseline_entropies_correct
    else 0
)

avg_entropy_wrong = (
    sum(baseline_entropies_wrong)
    / len(baseline_entropies_wrong)
    if baseline_entropies_wrong
    else 0
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 70)
print("TRUSTQA TIER-2 EVALUATION")
print("=" * 70)

print(
    f"Total questions: {total}"
)

print()

print("========== BASELINE ==========")

print(
    f"Correct: {baseline_correct}"
)

print(
    f"Wrong: {baseline_wrong}"
)

print(
    f"Accuracy: {baseline_accuracy * 100:.2f}%"
)

print(
    f"Avg entropy (Correct): "
    f"{avg_entropy_correct:.5f}"
)

print(
    f"Avg entropy (Wrong): "
    f"{avg_entropy_wrong:.5f}"
)


print()
print("========== TIER-2 / RAG ==========")

print(
    f"Correct: {rag_correct}"
)

print(
    f"Wrong: {rag_wrong}"
)

print(
    f"Accuracy: {rag_accuracy * 100:.2f}%"
)


print()
print("========== RAG ACTIVATION ==========")

print(
    f"RAG Triggered: {rag_triggered}"
)

print(
    f"RAG Activation: "
    f"{rag_activation * 100:.2f}%"
)

print(
    f"Entropy Threshold: "
    f"{ENTROPY_THRESHOLD}"
)


print()
print("========== TRANSITIONS ==========")

print(
    "Baseline Wrong -> RAG Correct:",
    baseline_wrong_to_rag_correct
)

print(
    "Baseline Correct -> RAG Wrong:",
    baseline_correct_to_rag_wrong
)

print(
    "Both Correct:",
    both_correct
)

print(
    "Both Wrong:",
    both_wrong
)


print()
print("========== HALLUCINATION / ERROR EFFECT ==========")

print(
    "Wrong answers before RAG:",
    baseline_wrong
)

print(
    "Wrong answers after RAG:",
    rag_wrong
)

print(
    "Errors reduced:",
    wrong_reduction
)

print(
    f"Wrong -> Correct rate: "
    f"{wrong_to_correct_rate * 100:.2f}%"
)

print(
    f"Correct -> Wrong rate: "
    f"{correct_to_wrong_rate * 100:.2f}%"
)


print()
print("========== TRIGGERED CASES ==========")

print(
    "Triggered + Correct:",
    triggered_correct
)

print(
    "Triggered + Wrong:",
    triggered_wrong
)

print(
    "Triggered + Fixed:",
    len([
        x for x in fixed_cases
        if x["entropy"] < ENTROPY_THRESHOLD
    ])
)

print(
    "Triggered + Broken:",
    len([
        x for x in broken_cases
        if x["entropy"] < ENTROPY_THRESHOLD
    ])
)


# ============================================================
# FIXED
# ============================================================

print()
print("========== BASELINE WRONG -> RAG CORRECT ==========")

if not fixed_cases:

    print("None")

else:

    for case in fixed_cases:

        print()

        print("Q:", case["question"])

        print(
            "Baseline:",
            case["baseline"]
        )

        print(
            "RAG:",
            case["rag"]
        )

        print(
            "Gold:",
            case["gold"]
        )

        print(
            "Baseline F1:",
            case["baseline_f1"]
        )

        print(
            "RAG F1:",
            case["rag_f1"]
        )

        print(
            "Entropy:",
            case["entropy"]
        )

        print("-" * 65)


# ============================================================
# BROKEN
# ============================================================

print()
print("========== BASELINE CORRECT -> RAG WRONG ==========")

if not broken_cases:

    print("None")

else:

    for case in broken_cases:

        print()

        print("Q:", case["question"])

        print(
            "Baseline:",
            case["baseline"]
        )

        print(
            "RAG:",
            case["rag"]
        )

        print(
            "Gold:",
            case["gold"]
        )

        print(
            "Baseline F1:",
            case["baseline_f1"]
        )

        print(
            "RAG F1:",
            case["rag_f1"]
        )

        print(
            "Entropy:",
            case["entropy"]
        )

        print("-" * 65)


# ============================================================
# STILL WRONG
# ============================================================

print()
print("========== STILL WRONG AFTER RAG ==========")

if not still_wrong_cases:

    print("None")

else:

    for case in still_wrong_cases:

        print()

        print("Q:", case["question"])

        print(
            "Baseline:",
            case["baseline"]
        )

        print(
            "RAG:",
            case["rag"]
        )

        print(
            "Gold:",
            case["gold"]
        )

        print(
            "Baseline F1:",
            case["baseline_f1"]
        )

        print(
            "RAG F1:",
            case["rag_f1"]
        )

        print(
            "Entropy:",
            case["entropy"]
        )

        print("-" * 65)


print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)