import json
import re
from collections import Counter

RESULT_FILE = "results/faiss_rag_iirc.json"


def normalize(text):
    text = str(text).lower().strip()

    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())

    return text


def compute_em(prediction, gold_answers):

    pred = normalize(prediction)

    for gold in gold_answers:

        if pred == normalize(gold):
            return 1

    return 0


def compute_f1(prediction, gold_answers):

    pred_tokens = normalize(
        prediction
    ).split()

    best_f1 = 0

    for gold in gold_answers:

        gold_tokens = normalize(
            gold
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
            continue

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

        f1 = (
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

        best_f1 = max(
            best_f1,
            f1
        )

    return best_f1


with open(
    RESULT_FILE,
    "r",
    encoding="utf-8"
) as f:

    results = json.load(f)


total = len(results)

em_total = 0
f1_total = 0

not_found = 0

for item in results:

    prediction = item[
        "prediction"
    ]

    gold_answers = item[
        "gold_answers"
    ]

    if (
        prediction
        .strip()
        .upper()
        ==
        "NOT_FOUND"
    ):
        not_found += 1

    em_total += compute_em(
        prediction,
        gold_answers
    )

    f1_total += compute_f1(
        prediction,
        gold_answers
    )


print()
print(
    "FAISS RAG IIRC RESULTS"
)
print()

print(
    "Questions :",
    total
)

print(
    "EM         :",
    round(
        em_total / total,
        4
    )
)

print(
    "F1         :",
    round(
        f1_total / total,
        4
    )
)

print(
    "NOT_FOUND  :",
    not_found
)