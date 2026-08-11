import json
import re
import string
from collections import Counter


RESULT_FILE = "results/bm25_rag_iirc.json"


def normalize_answer(s):

    s = str(s).lower()

    s = re.sub(r"\b(a|an|the)\b", " ", s)

    s = "".join(
        ch
        for ch in s
        if ch not in string.punctuation
    )

    s = " ".join(s.split())

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

    pred_tokens = (
        normalize_answer(prediction)
        .split()
    )

    gold_tokens = (
        normalize_answer(ground_truth)
        .split()
    )

    common = (
        Counter(pred_tokens)
        &
        Counter(gold_tokens)
    )

    num_same = sum(
        common.values()
    )

    if (
        len(pred_tokens) == 0
        or
        len(gold_tokens) == 0
    ):

        return float(
            pred_tokens
            ==
            gold_tokens
        )

    if num_same == 0:

        return 0.0

    precision = (
        num_same
        /
        len(pred_tokens)
    )

    recall = (
        num_same
        /
        len(gold_tokens)
    )

    return (
        2
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        )
    )


with open(
    RESULT_FILE,
    "r",
    encoding="utf-8"
) as f:

    results = json.load(f)


total = len(results)

em_sum = 0.0
f1_sum = 0.0

not_found = 0


for item in results:

    prediction = (
        item["prediction"]
        .strip()
    )

    gold_answers = (
        item["gold_answers"]
    )

    if (
        prediction.upper()
        ==
        "NOT_FOUND"
    ):

        not_found += 1

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

    em_sum += em
    f1_sum += f1


em = em_sum / total
f1 = f1_sum / total


print()
print("BM25 RAG IIRC RESULTS")
print()
print(
    "Questions :",
    total
)
print(
    "EM         :",
    round(em, 4)
)
print(
    "F1         :",
    round(f1, 4)
)
print(
    "NOT_FOUND  :",
    not_found
)