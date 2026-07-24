import json
import re
import string
from collections import Counter


def normalize_answer(s):
    s = s.lower()

    s = "".join(
        ch for ch in s
        if ch not in string.punctuation
    )

    s = " ".join(s.split())

    return s


def compute_exact(a_gold, a_pred):
    return int(
        normalize_answer(a_gold)
        == normalize_answer(a_pred)
    )


def compute_f1(a_gold, a_pred):

    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()

    common = Counter(gold_toks) & Counter(pred_toks)

    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)

    return (
        2 * precision * recall
        / (precision + recall)
    )


with open(
    "results/bm25_rag_nq.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

total = len(data)

em = 0
f1 = 0

for sample in data:

    prediction = sample["prediction"]

    golds = sample["gold_answers"]["text"]

    em_scores = [
        compute_exact(g, prediction)
        for g in golds
    ]

    f1_scores = [
        compute_f1(g, prediction)
        for g in golds
    ]

    em += max(em_scores)
    f1 += max(f1_scores)

print("\nBM25 RAG NQ RESULTS\n")
print(f"Questions : {total}")
print(f"EM         : {em/total:.4f}")
print(f"F1         : {f1/total:.4f}")