"""Regression checks for the physics-first repository layout."""

from __future__ import annotations

import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "neutron_star_eos"
CAMPAIGN_ROOT = REPOSITORY_ROOT / "experiments" / "compose_comparison"
WORKFLOW_ROOT = REPOSITORY_ROOT / "workflows"


class PhysicsFirstStructureTests(unittest.TestCase):
    def test_scientific_modules_remain_focused(self) -> None:
        oversized = {
            path.relative_to(REPOSITORY_ROOT).as_posix(): len(
                path.read_text(encoding="utf-8").splitlines()
            )
            for root in (SOURCE_ROOT, CAMPAIGN_ROOT)
            for path in root.rglob("*.py")
            if len(path.read_text(encoding="utf-8").splitlines()) > 500
        }
        self.assertEqual(oversized, {})
        self.assertLessEqual(
            len((CAMPAIGN_ROOT / "run.py").read_text(encoding="utf-8").splitlines()),
            100,
        )

    def test_beginner_workflows_are_short_and_task_oriented(self) -> None:
        expected = {
            "analytical_eos.py",
            "csv_eos.py",
            "compose_eos.py",
            "calculate_one_star.py",
            "calculate_mass_radius.py",
            "compare_equations_of_state.py",
        }
        actual = {path.name for path in WORKFLOW_ROOT.glob("*.py")}
        self.assertEqual(actual, expected)
        for name in sorted(expected):
            with self.subTest(workflow=name):
                source = (WORKFLOW_ROOT / name).read_text(encoding="utf-8")
                self.assertLessEqual(len(source.splitlines()), 100)
                self.assertIn("parameters to edit", source)
                compile(source, str(WORKFLOW_ROOT / name), "exec")

    def test_physical_calculations_have_obvious_locations(self) -> None:
        expected_files = (
            SOURCE_ROOT / "eos" / "validation.py",
            SOURCE_ROOT / "compose" / "reader.py",
            SOURCE_ROOT / "compose" / "barotrope.py",
            SOURCE_ROOT / "stellar" / "tov.py",
            SOURCE_ROOT / "stellar" / "sequence.py",
            SOURCE_ROOT / "plotting" / "stellar.py",
            SOURCE_ROOT / "output" / "bundles.py",
        )
        self.assertTrue(all(path.is_file() for path in expected_files))


if __name__ == "__main__":
    unittest.main()
