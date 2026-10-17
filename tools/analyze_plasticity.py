#!/usr/bin/env python3
"""Fit and hold out symbolic models for plasticity-continuation cohorts."""

import argparse
import csv
import json
import math
from pathlib import Path
import tempfile

FEATURES = (
    "frozen",
    "initial_age",
    "initial_energy",
    "initial_health",
    "neurons",
    "synapses",
    "generation",
)
TARGETS = ("survival", "offspring")


def read_trials(results_path):
    results_path = Path(results_path).resolve(strict=True)
    with results_path.open(encoding="utf-8") as source:
        results = json.load(source)
    status_path = results_path.parent / "status.json"
    if not status_path.is_file():
        raise ValueError(f"Missing completed-run status file: {status_path}")
    with status_path.open(encoding="utf-8") as source:
        status = json.load(source)
    trials = results.get("trials")
    if status.get("state") != "complete":
        raise ValueError(f"Trial batch is not complete (state: {status.get('state')!r})")
    if not isinstance(trials, list) or not trials:
        raise ValueError("results.json must contain a non-empty trials list")
    if results.get("completed_trials") != len(trials):
        raise ValueError("results.json does not contain all completed trials")
    return results_path.parent, trials


def cohort_path(root, trial):
    recorded_path = trial.get("cohort_file")
    if recorded_path:
        path = Path(recorded_path)
        if path.is_file():
            return path
    world = int(trial["world"])
    index = int(trial["trial"])
    fallback = root / f"world-{world:03d}-trial-{index:03d}" / "world.csv.plasticity.csv"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Cannot locate cohort CSV for world {world}, trial {index}")


def number(row, key, path):
    try:
        result = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {key!r} value in {path}: {row}") from error
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {key!r} value in {path}: {row}")
    return result


def integer(row, key, path):
    result = number(row, key, path)
    if not result.is_integer():
        raise ValueError(f"Expected an integer {key!r} value in {path}: {row}")
    return int(result)


def load_samples(results_path, held_out_worlds, target):
    root, trials = read_trials(results_path)
    if target not in TARGETS:
        raise ValueError(f"Unsupported target {target!r}; choose from {TARGETS}")
    held_out_worlds = set(held_out_worlds)
    if not held_out_worlds:
        raise ValueError("Specify at least one --held-out-world")

    # Repeated assignments revisit the same original cohort. Pool them within
    # each world and animal-treatment pair instead of treating repetitions as
    # independent animals.
    pooled = {}
    baseline_by_animal = {}
    cohort_ids_by_world = {}
    worlds = set()
    trial_counts = {}
    seen_trials = set()
    for trial in trials:
        world = int(trial["world"])
        trial_index = int(trial["trial"])
        trial_key = (world, trial_index)
        if trial_key in seen_trials:
            raise ValueError(f"Duplicate trial {trial_index} for world {world}")
        seen_trials.add(trial_key)
        worlds.add(world)
        trial_counts[world] = trial_counts.get(world, 0) + 1
        path = cohort_path(root, trial)
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            required = {"id", "frozen", "alive", "initial_age", "initial_energy",
                        "initial_health", "neurons", "synapses", "generation",
                        "initial_offspring", "final_offspring"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
            seen_ids = set()
            has_control = has_frozen = False
            for row in reader:
                animal_id = integer(row, "id", path)
                if animal_id in seen_ids:
                    raise ValueError(f"Duplicate animal id {animal_id} in {path}")
                seen_ids.add(animal_id)
                frozen_value = integer(row, "frozen", path)
                alive = integer(row, "alive", path)
                if frozen_value not in (0, 1) or alive not in (0, 1):
                    raise ValueError(f"Expected binary frozen/alive values in {path}")
                has_frozen |= frozen_value == 1
                has_control |= frozen_value == 0
                baseline = tuple(number(row, key, path) for key in FEATURES[1:])
                animal_key = (world, animal_id)
                previous = baseline_by_animal.setdefault(animal_key, baseline)
                if baseline != previous:
                    raise ValueError(
                        f"Baseline covariates changed for world {world}, animal {animal_id}"
                    )
                offspring = (number(row, "final_offspring", path)
                             - number(row, "initial_offspring", path))
                if offspring < 0:
                    raise ValueError(f"Post-assignment offspring count decreased in {path}")
                value = alive if target == "survival" else offspring
                key = (world, animal_id, frozen_value)
                total, count = pooled.get(key, (0.0, 0))
                pooled[key] = (total + value, count + 1)
            previous_ids = cohort_ids_by_world.setdefault(world, seen_ids)
            if seen_ids != previous_ids:
                raise ValueError(f"Original cohort IDs changed between trials in world {world}")
            if not has_control or not has_frozen:
                raise ValueError(
                    f"World {world}, trial {trial_index} must contain both treatment groups"
                )

    missing_worlds = held_out_worlds - worlds
    training_worlds = worlds - held_out_worlds
    if missing_worlds:
        raise ValueError(f"Unknown held-out world(s): {sorted(missing_worlds)}")
    if len(training_worlds) < 2:
        raise ValueError("Need at least two training worlds and one held-out world")

    samples = []
    for (world, animal_id, frozen), (total, count) in sorted(pooled.items()):
        if count == 0:
            continue
        baseline = baseline_by_animal[(world, animal_id)]
        features = (float(frozen), *baseline)
        samples.append({
            "world": world,
            "features": features,
            "target": total / count,
            "assignment_repeats": count,
        })
    training = [row for row in samples if row["world"] not in held_out_worlds]
    testing = [row for row in samples if row["world"] in held_out_worlds]
    if not training or not testing:
        raise ValueError("Training and held-out sets must both contain cohort rows")
    return training, testing, sorted(worlds), trial_counts


def regression_metrics(actual, predicted):
    errors = [prediction - value for value, prediction in zip(actual, predicted)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    mean_actual = sum(actual) / len(actual)
    total = sum((value - mean_actual) ** 2 for value in actual)
    r_squared = None if total == 0 else 1 - sum(error * error for error in errors) / total
    return {"mae": mae, "rmse": rmse, "r_squared": r_squared}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="completed plasticity_continuation results.json")
    parser.add_argument("--held-out-world", type=int, action="append", required=True,
                        help="world index to reserve entirely for evaluation; repeat as needed")
    parser.add_argument("--target", choices=TARGETS, default="survival")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--max-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="JSON report destination")
    args = parser.parse_args()
    if args.iterations < 1 or args.max_size < 7 or args.seed < 0:
        parser.error("--iterations must be positive, --max-size at least 7, and --seed nonnegative")

    training, testing, worlds, trial_counts = load_samples(
        args.results, args.held_out_world, args.target
    )
    try:
        import numpy as np
        from pysr import PySRRegressor
    except ImportError as error:
        raise SystemExit(
            "This optional analysis requires NumPy and PySR. Install PySR with "
            "`python -m pip install pysr` and rerun."
        ) from error

    x_train = np.asarray([row["features"] for row in training], dtype=float)
    y_train = np.asarray([row["target"] for row in training], dtype=float)
    x_test = np.asarray([row["features"] for row in testing], dtype=float)
    y_test = np.asarray([row["target"] for row in testing], dtype=float)
    with tempfile.TemporaryDirectory(prefix="neural-selection-pysr-") as output_directory:
        model = PySRRegressor(
            binary_operators=["+", "-", "*", "/"],
            niterations=args.iterations,
            maxsize=args.max_size,
            output_directory=output_directory,
            random_state=args.seed,
            deterministic=True,
            parallelism="serial",
        )
        model.fit(x_train, y_train)
        predictions = np.asarray(model.predict(x_test), dtype=float).reshape(-1)
        best = model.get_best()
    test_metrics = regression_metrics(y_test.tolist(), predictions.tolist())
    by_world = {}
    held_out_worlds = sorted(set(args.held_out_world))
    for world in held_out_worlds:
        indexes = [index for index, row in enumerate(testing) if row["world"] == world]
        by_world[str(world)] = {
            "rows": len(indexes),
            **regression_metrics(y_test[indexes].tolist(), predictions[indexes].tolist()),
        }
    report = {
        "target": args.target,
        "feature_order": list(FEATURES),
        "equation": str(best["sympy_format"]),
        "complexity": int(best["complexity"]),
        "training_worlds": sorted(set(worlds) - set(args.held_out_world)),
        "held_out_worlds": held_out_worlds,
        "trial_counts_by_world": {str(key): value for key, value in sorted(trial_counts.items())},
        "training_animal_treatment_rows": len(training),
        "held_out_animal_treatment_rows": len(testing),
        "held_out_metrics": test_metrics,
        "held_out_metrics_by_world": by_world,
        "baseline_held_out_metrics": regression_metrics(
            y_test.tolist(), [float(y_train.mean())] * len(y_test)
        ),
        "interpretation": (
            "Predictive symbolic regression only. Held-out worlds test transfer of this "
            "association; the equation does not establish causation, validate a learning "
            "mechanism, or replace randomized treatment comparisons. Animals and repeated "
            "assignments within a world are not independent replicates."
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2)
        destination.write("\n")
    print(f"Saved held-out symbolic-regression report to {output}")
    print(f"Equation: {report['equation']}")
    print(f"Held-out metrics: {json.dumps(test_metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
