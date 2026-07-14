#!/usr/bin/env python3
"""统计 annotations.json 中所有 case 的 G/S/B 数量，并计算 GSB=(G+S)/(B+S)。

用法:
    python compute_gsb.py /path/to/annotations.json
"""
import argparse
import collections
import json
import sys


def compute_gsb(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations", {})
    counter = collections.Counter()
    n_cases = 0
    n_choices = 0

    for _cid, value in annotations.items():
        n_cases += 1
        for _k, label in value.get("choices", {}).items():
            n_choices += 1
            counter[label] += 1

    G = counter.get("G", 0)
    S = counter.get("S", 0)
    B = counter.get("B", 0)
    other = {k: n for k, n in counter.items() if k not in ("G", "S", "B")}

    den = B + S
    gsb = (G + S) / den if den else None

    return {
        "n_cases": n_cases,
        "n_choices": n_choices,
        "G": G,
        "S": S,
        "B": B,
        "other": other,
        "gsb": gsb,
    }


def main():
    parser = argparse.ArgumentParser(
        description="统计 annotations 的 G/S/B 并计算 GSB=(G+S)/(B+S)"
    )
    parser.add_argument("annotations", help="annotations.json 文件路径")
    args = parser.parse_args()

    res = compute_gsb(args.annotations)

    print(f"cases: {res['n_cases']} | labeled choices: {res['n_choices']}")
    print(f"G: {res['G']}  S: {res['S']}  B: {res['B']}  | other: {res['other']}")
    if res["gsb"] is None:
        print("GSB=(G+S)/(B+S)= N/A (B+S=0)")
    else:
        G, S, B = res["G"], res["S"], res["B"]
        print(
            f"GSB=(G+S)/(B+S)= ({G}+{S})/({B}+{S}) = {G + S}/{B + S} = {res['gsb']:.4f}"
        )


if __name__ == "__main__":
    sys.exit(main())
