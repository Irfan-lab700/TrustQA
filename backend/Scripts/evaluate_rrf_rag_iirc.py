import json
import re

RESULT_FILE = "results/rrf_rag_iirc.json"


def normalize(text):

    text = str(text).lower()

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    text = " ".join(
        text.split()
    )

    return text


def exact_match(
    prediction,
    gold_answers
):

    prediction = normalize(
        prediction
    )

    for gold in gold_answers:

        if prediction == normalize(
            gold
        ):
            return 1

    return 0


def f1_score(
    prediction,
    gold
):

    pred_tokens = normalize(
        prediction
    ).split()

    gold_tokens = normalize(
        gold
    ).split()

    if (
        len(pred_tokens) == 0
        or
        len(gold_tokens) == 0
    ):
        return 0

    common = set(
        pred_tokens
    ) & set(
        gold_tokens
    )

    num_same = sum(
        min(
            pred_tokens.count(token),
            gold_tokens.count(token)
        )
        for token in common
    )

    if num_same == 0:
        return 0

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


em_total = 0

f1_total = 0

not_found = 0


for sample in results:

    prediction = sample[
        "prediction"
    ]

    gold_answers = sample[
        "gold_answers"
    ]

    if (
        prediction.strip()
        ==
        "NOT_FOUND"
    ):
        not_found += 1

    em_total += exact_match(
        prediction,
        gold_answers
    )

    best_f1 = max(
        f1_score(
            prediction,
            gold
        )
        for gold in gold_answers
    )

    f1_total += best_f1


n = len(results)

print(
    "\nRRF RAG IIRC RESULTS\n"
)

print(
    "Questions :",
    n
)

print(
    "EM         :",
    round(
        em_total / n,
        4
    )
)

print(
    "F1         :",
    round(
        f1_total / n,
        4
    )
)

print(
    "NOT_FOUND  :",
    not_found
)