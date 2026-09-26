#!/usr/bin/env python3
"""Summarize observational behaviour logs from a neural-selection run."""

import argparse
import csv
from pathlib import Path
from statistics import mean

AGES = ("early", "middle", "late")
FOODS = ("a", "b", "c")


def read_csv(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    return [row for row in rows if row and all(value is not None for value in row.values())]


def value(row, field):
    return float(row[field])


def delta(start, end, field):
    return value(end, field) - value(start, field)


def rate(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def describe_window(label, start, end):
    print(f"{label}: steps {int(value(start, 'step')):,}–{int(value(end, 'step')):,}")
    for hunger in ("hungry", "sated"):
        parts = []
        for food in FOODS:
            opportunities = sum(delta(start, end, f"{age}_{hunger}_op_{food}")
                                for age in AGES)
            accepted = sum(delta(start, end, f"{age}_{hunger}_eat_{food}")
                           for age in AGES)
            parts.append(f"{food.upper()} {accepted:,.0f}/{opportunities:,.0f} "
                         f"({rate(accepted, opportunities):.1%})")
        print(f"  {hunger} food encounters accepted: " + ", ".join(parts))
    approaches = []
    for food in FOODS:
        progress = sum(delta(start, end, f"{age}_approach_{food}") for age in AGES)
        travel = sum(delta(start, end, f"{age}_travel_{food}") for age in AGES)
        approaches.append(f"{food.upper()} {rate(progress, travel):+.3f}")
    print("  hungry approach alignment (−1 away, +1 toward): " + ", ".join(approaches))


def lifetime_comparison(lives, last_step, window):
    recent = [row for row in lives if value(row, "death_step") >= last_step - window]
    print(f"Sampled completed lives in recent window: {len(recent):,}")
    for food in FOODS:
        paired = []
        for row in recent:
            early_op = value(row, f"early_hungry_op_{food}")
            late_op = value(row, f"late_hungry_op_{food}")
            if early_op < 3 or late_op < 3:
                continue
            early_rate = value(row, f"early_hungry_eat_{food}") / early_op
            late_rate = value(row, f"late_hungry_eat_{food}") / late_op
            paired.append(late_rate - early_rate)
        if paired:
            print(f"  same-animal hungry {food.upper()} acceptance change, "
                  f"age 300+ versus 0–99: {mean(paired):+.1%} (n={len(paired)})")
        else:
            print(f"  same-animal hungry {food.upper()} acceptance: insufficient pairs")

    if len(recent) >= 8:
        ordered = sorted(recent, key=lambda row: value(row, "active_neurons"))
        size = max(1, len(ordered) // 4)
        for label, group in (("smaller", ordered[:size]), ("larger", ordered[-size:])):
            print(f"  {label} brain quartile: {mean(value(r, 'active_neurons') for r in group):.1f} "
                  f"active neurons, {mean(value(r, 'age') for r in group):.0f} "
                  f"steps lived, {mean(value(r, 'offspring') for r in group):.2f} offspring")
        changed = sum(value(row, "mean_abs_weight_change") > 0.05 for row in recent)
        print(f"  measurable synaptic weight change (>0.05 nS): "
              f"{changed:,}/{len(recent):,} sampled lives")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="simulation --output path")
    parser.add_argument("--window", type=int, default=10000,
                        help="world steps per comparison window (default: 10000)")
    args = parser.parse_args()
    behavior = read_csv(Path(str(args.output) + ".behavior.csv"))
    lives = read_csv(Path(str(args.output) + ".lives.csv"))
    if len(behavior) < 3:
        raise SystemExit("Need at least three behaviour snapshots")
    last = int(value(behavior[-1], "step"))
    width = min(args.window, max(1, last // 3))
    early_end = next((row for row in behavior if value(row, "step") >= width), behavior[-1])
    recent_start = next((row for row in behavior if value(row, "step") >= last - width), behavior[0])
    describe_window("Early", behavior[0], early_end)
    describe_window("Recent", recent_start, behavior[-1])
    lifetime_comparison(lives, last, width)
    print("These are observational associations. Age, experience, genotype, and "
          "survival selection can all affect the rates; they do not by themselves prove learning.")
    print("C has no distant sensory channel, so its approach score is a reference, "
          "not evidence that animals can see C.")


if __name__ == "__main__":
    main()
