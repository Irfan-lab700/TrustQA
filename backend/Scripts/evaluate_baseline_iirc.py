import json
import re
import string
from collections import Counter


def normalize_answer(text):
    text = str(text).lower()

    text = re.sub(r"\b(a|an|the)\b", " ", text)

    text = "".join(
        ch for ch in text
        if ch not in string.punctuation
    )

    text = " ".join(text.split())

    return text


def exact_match_score(prediction, ground_truth):
    return int(
        normalize_answer(prediction)
        ==
        normalize_answer(ground_truth)
    )


def f1_score(prediction, ground_truth):

    pred_tokens = normalize_answer(
        prediction
    ).split()

    gold_tokens = normalize_answer(
        ground_truth
    ).split()

    common = (
        Counter(pred_tokens)
        &
        Counter(gold_tokens)
    )

    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return int(pred_tokens == gold_tokens)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return (
        2 * precision * recall
        /
        (precision + recall)
    )


RESULT_FILE = "results/baseline_llm_iirc.json"

with open(
    RESULT_FILE,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

total_em = 0.0
total_f1 = 0.0

for item in data:

    prediction = item["prediction"]

    gold_answers = item["gold_answers"]

    em = max(
        exact_match_score(
            prediction,
            gold
        )
        for gold in gold_answers
    )

    f1 = max(
        f1_score(
            prediction,
            gold
        )
        for gold in gold_answers
    )

    total_em += em
    total_f1 += f1

n = len(data)

print("\nIIRC BASELINE RESULTS\n")
print(f"Questions : {n}")
print(f"EM         : {total_em/n:.4f}")
print(f"F1         : {total_f1/n:.4f}")