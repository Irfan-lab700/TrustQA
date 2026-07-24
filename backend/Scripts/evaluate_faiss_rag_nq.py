import json
import re
import string
from collections import Counter


RESULT_FILE = "results/faiss_rag_nq.json"


def normalize_answer(s):
    s = s.lower()

    s = "".join(
        ch for ch in s
        if ch not in string.punctuation
    )

    s = " ".join(s.split())

    return s


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
        2 *
        precision *
        recall /
        (precision + recall)
    )


def exact_match_score(
    prediction,
    ground_truth
):

    return (
        normalize_answer(
            prediction
        )
        ==
        normalize_answer(
            ground_truth
        )
    )


with open(
    RESULT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


total_em = 0
total_f1 = 0

not_found = 0


for sample in data:

    prediction = sample[
        "prediction"
    ]

    if (
        prediction.strip()
        ==
        "NOT_FOUND"
    ):
        not_found += 1

    golds = sample[
        "gold_answers"
    ]["text"]

    em = max(
        exact_match_score(
            prediction,
            g
        )
        for g in golds
    )

    f1 = max(
        f1_score(
            prediction,
            g
        )
        for g in golds
    )

    total_em += em
    total_f1 += f1


n = len(data)

print()
print("FAISS RAG NQ RESULTS")
print()
print("Questions :", n)
print(
    "EM         : %.4f"
    % (total_em / n)
)
print(
    "F1         : %.4f"
    % (total_f1 / n)
)
print(
    "NOT_FOUND  :",
    not_found
)