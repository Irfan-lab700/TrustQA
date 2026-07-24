import json
import re
import string
from collections import Counter


def normalize_answer(s):
    s = s.lower()

    s = ''.join(
        ch for ch in s
        if ch not in string.punctuation
    )

    s = re.sub(
        r'\b(a|an|the)\b',
        ' ',
        s
    )

    s = ' '.join(s.split())

    return s


def exact_match_score(
    prediction,
    ground_truth
):
    return (
        normalize_answer(prediction)
        ==
        normalize_answer(ground_truth)
    )


def f1_score(
    prediction,
    ground_truth
):

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

    num_same = sum(
        common.values()
    )

    if num_same == 0:
        return 0

    precision = (
        num_same /
        len(pred_tokens)
    )

    recall = (
        num_same /
        len(gold_tokens)
    )

    return (
        2 * precision * recall
    ) / (
        precision + recall
    )


with open(
    "results/rrf_rag_nq.json",
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


total_em = 0
total_f1 = 0
not_found = 0

for item in data:

    prediction = item[
        "prediction"
    ]

    if (
        prediction.strip()
        ==
        "NOT_FOUND"
    ):
        not_found += 1

    answers = item[
        "gold_answers"
    ]["text"]

    em = max(
        exact_match_score(
            prediction,
            ans
        )
        for ans in answers
    )

    f1 = max(
        f1_score(
            prediction,
            ans
        )
        for ans in answers
    )

    total_em += em
    total_f1 += f1


n = len(data)

print()
print("RRF RAG NQ RESULTS")
print()
print("Questions :", n)
print(
    "EM         :",
    round(total_em / n, 4)
)
print(
    "F1         :",
    round(total_f1 / n, 4)
)
print(
    "NOT_FOUND  :",
    not_found
)