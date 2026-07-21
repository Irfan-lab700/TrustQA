import json
import re
import string
from collections import Counter

# Normalization

def normalize(text):

    text = text.lower()

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text
    )

    text = "".join(
        ch for ch in text
        if ch not in string.punctuation
    )

    text = " ".join(text.split())

    return text

# Exact Match

def exact_match(prediction, ground_truth):

    return int(
        normalize(prediction)
        ==
        normalize(ground_truth)
    )


# Token F1

def f1_score(prediction, ground_truth):

    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(ground_truth).split()

    common = (
        Counter(pred_tokens)
        &
        Counter(gold_tokens)
    )

    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)

    recall = num_same / len(gold_tokens)

    return (
        2 * precision * recall
        /
        (precision + recall)
    )

# Load Results

with open(
    "results/baseline_llm_nq_new.json",
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)

# Evaluate

total = len(data)

em_total = 0
f1_total = 0

for sample in data:

    prediction = sample["prediction"]

    gold_answers = sample["gold_answers"]["text"]

    best_em = 0
    best_f1 = 0

    for gold in gold_answers:

        best_em = max(
            best_em,
            exact_match(prediction, gold)
        )

        best_f1 = max(
            best_f1,
            f1_score(prediction, gold)
        )

    em_total += best_em
    f1_total += best_f1

# Final Metrics

em = em_total / total
f1 = f1_total / total

print("\nNQ BASELINE RESULTS\n")

print(f"Questions : {total}")
print(f"EM         : {em:.4f}")
print(f"F1         : {f1:.4f}")