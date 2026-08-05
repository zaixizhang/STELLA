from __future__ import annotations

from pathlib import Path
import unittest

from summarize_eval import Obs, select_scores


def observation(task: str, model: str, score: float | None, repeat: int) -> Obs:
    return Obs(
        task_name=task,
        task_id=task,
        model=model,
        score=score,
        returncode=0 if score is not None else 1,
        source_file=Path(f"repeat_{repeat}.csv"),
        source_mtime=float(repeat),
    )


class SelectScoresTest(unittest.TestCase):
    def test_uniform_top3_of_six_rule_is_applied_to_every_model(self) -> None:
        observations = []
        for model, scores in {
            "STELLA": [0.1, 0.3, 0.2, 0.6, 0.5, 0.4],
            "Baseline": [0.7, 0.2, 0.3, 0.1, 0.6, 0.5],
        }.items():
            observations.extend(
                observation("higher_task", model, score, repeat)
                for repeat, score in enumerate(scores, start=1)
            )

        rows = select_scores(
            observations,
            {"higher_task": "higher"},
            top_k=3,
            expected_observations=6,
        )

        self.assertEqual({row["selected_from_file"] for row in rows}, {"uniform_top3_valid_mean"})
        self.assertEqual({row["agg_rule"] for row in rows}, {"uniform_top3_valid_mean"})
        self.assertEqual({row["total_observations"] for row in rows}, {6})
        self.assertEqual({row["selected_k"] for row in rows}, {3})
        scores = {row["model"]: row["selected_score"] for row in rows}
        self.assertEqual(scores, {"Baseline": 0.6, "STELLA": 0.5})

    def test_lower_is_better_selects_best_three_of_six(self) -> None:
        observations = [
            observation("lower_task", "STELLA", score, repeat)
            for repeat, score in enumerate([3.0, 1.0, 2.0, 6.0, 5.0, 4.0], start=1)
        ]
        rows = select_scores(
            observations,
            {"lower_task": "lower"},
            top_k=3,
            expected_observations=6,
        )
        self.assertEqual(rows[0]["selected_score"], 2.0)
        self.assertEqual(rows[0]["total_observations"], 6)
        self.assertEqual(rows[0]["selected_from_file"], "uniform_top3_valid_mean")

    def test_mismatched_six_run_budget_is_rejected(self) -> None:
        observations = [
            observation("task", "STELLA", score, repeat)
            for repeat, score in enumerate([0.1, 0.2, 0.3, 0.4, 0.5], start=1)
        ]
        observations.extend(
            observation("task", "Baseline", score, repeat)
            for repeat, score in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], start=1)
        )
        with self.assertRaisesRegex(ValueError, "Fairness check failed"):
            select_scores(
                observations,
                {"task": "higher"},
                top_k=3,
                expected_observations=6,
            )


if __name__ == "__main__":
    unittest.main()
