"""Unit tests for H2 cohort loading and held-out evaluation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.analyze_plasticity import load_samples, regression_metrics


class AnalyzePlasticityTests(unittest.TestCase):
    def write_batch(self, root, world_count=3, trial_count=2):
        trials = []
        for world in range(world_count):
            for trial in range(trial_count):
                directory = root / f"world-{world:03d}-trial-{trial:03d}"
                directory.mkdir()
                cohort = directory / "world.csv.plasticity.csv"
                with cohort.open("w", newline="", encoding="utf-8") as destination:
                    writer = csv.DictWriter(destination, fieldnames=[
                        "id", "frozen", "alive", "initial_age", "initial_energy",
                        "initial_health", "neurons", "synapses", "generation",
                        "initial_offspring", "final_offspring",
                    ])
                    writer.writeheader()
                    for animal_id in range(4):
                        writer.writerow({
                            "id": animal_id,
                            "frozen": (animal_id + trial) % 2,
                            "alive": int((animal_id + world + trial) % 3 != 0),
                            "initial_age": animal_id * 5,
                            "initial_energy": 12 + animal_id,
                            "initial_health": 1.0,
                            "neurons": 52,
                            "synapses": 180,
                            "generation": world,
                            "initial_offspring": 0,
                            "final_offspring": int((animal_id + trial) % 2),
                        })
                trials.append({
                    "world": world,
                    "trial": trial,
                    "cohort_file": str(cohort),
                })
        (root / "results.json").write_text(json.dumps({
            "completed_trials": len(trials),
            "trials": trials,
        }), encoding="utf-8")
        (root / "status.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")

    def test_pools_repeated_assignments_and_holds_out_whole_worlds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_batch(root)
            training, testing, worlds, trial_counts = load_samples(
                root / "results.json", [2], "survival"
            )
            self.assertEqual(worlds, [0, 1, 2])
            self.assertEqual(trial_counts, {0: 2, 1: 2, 2: 2})
            self.assertEqual(len(training), 16)
            self.assertEqual(len(testing), 8)
            self.assertEqual({row["world"] for row in training}, {0, 1})
            self.assertEqual({row["world"] for row in testing}, {2})
            self.assertTrue(all(row["assignment_repeats"] == 1 for row in testing))

    def test_requires_two_training_worlds_and_one_held_out_world(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_batch(root, world_count=2)
            with self.assertRaisesRegex(ValueError, "two training worlds"):
                load_samples(root / "results.json", [1], "survival")

    def test_refuses_incomplete_trial_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_batch(root)
            (root / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not complete"):
                load_samples(root / "results.json", [2], "survival")

    def test_constant_outcome_has_no_r_squared(self):
        metrics = regression_metrics([1.0, 1.0], [0.5, 1.5])
        self.assertEqual(metrics["mae"], 0.5)
        self.assertEqual(metrics["rmse"], 0.5)
        self.assertIsNone(metrics["r_squared"])


if __name__ == "__main__":
    unittest.main()
