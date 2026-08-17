from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar
from unittest import mock

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments" / "compose_comparison"
ACQUIRE_PATH = EXPERIMENT_ROOT / "acquire.py"
RUN_PATH = EXPERIMENT_ROOT / "run.py"
CONFIG_PATH = EXPERIMENT_ROOT / "config" / "models.json"


def load_acquisition_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "compose_comparison_acquire_tested", ACQUIRE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load CompOSE acquisition module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


acquire = load_acquisition_module()


def load_runner_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "compose_comparison_runner_tested", RUN_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load CompOSE campaign runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(EXPERIMENT_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(EXPERIMENT_ROOT))
    return module


runner = load_runner_module()


def deterministic_zip(member_names: tuple[str, ...]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, name in enumerate(member_names):
            info = zipfile.ZipInfo(name, date_time=(2024, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, f"synthetic member {index}\n".encode("ascii"))
    return output.getvalue()


class ComposeExperimentScaffoldTests(unittest.TestCase):
    config: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = acquire.load_config(CONFIG_PATH)

    def synthetic_model(self, slug: str, payload: bytes) -> dict[str, Any]:
        source = next(model for model in self.config["models"] if model["slug"] == slug)
        model = copy.deepcopy(source)
        model["archive"].update(
            {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "url": "https://example.invalid/synthetic.zip",
                "zenodo_record_url": "https://example.invalid/record",
            }
        )
        return model

    def required_members(self) -> tuple[str, ...]:
        return tuple(self.config["campaign"]["required_archive_members"])

    def apr_payload(self) -> bytes:
        return deterministic_zip((*self.required_members(), "eos.mr"))

    def test_registry_is_complete_pinned_and_convention_aware(self) -> None:
        models = self.config["models"]
        self.assertEqual(len(models), 9)
        self.assertEqual(
            [model["slug"] for model in models],
            [
                "bsk26",
                "sly4",
                "dd2",
                "fsu2h",
                "tw",
                "ddme_x",
                "gm1y6",
                "apr",
                "qhc19_c",
            ],
        )
        self.assertEqual(sum(model["role"] == "core" for model in models), 8)
        self.assertEqual(sum(model["role"] == "stress" for model in models), 1)
        for model in models:
            archive = model["archive"]
            self.assertTrue(archive["url"].startswith("https://zenodo.org/api/"))
            self.assertRegex(archive["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(archive["bytes"], 0)
            self.assertEqual(
                model["compose_benchmark"]["source_url"],
                model["compose_page_url"],
            )
            self.assertTrue(model["primary_citations"])
            self.assertTrue(
                all(
                    citation["url"].startswith("https://")
                    for citation in model["primary_citations"]
                )
            )
            literature = model["literature_assessment"]
            self.assertTrue(literature["comparability"])
            self.assertTrue(literature["source_url"].startswith("https://"))
            self.assertIn("maximum_mass_msun", literature)
            self.assertIn("radius_at_maximum_mass_km", literature)
            self.assertIn("radius_at_1_4_msun_km", literature)
        by_slug = {model["slug"]: model for model in models}
        self.assertEqual(by_slug["bsk26"]["compose_eos_id"], 258)
        self.assertEqual(
            by_slug["bsk26"]["archive"]["sha256"],
            "21f8cb9ea5b3ba12538b672fb69f2ccdd8624f4b46ec174e3d70b05cc8a2266b",
        )
        self.assertEqual(by_slug["qhc19_c"]["compose_eos_id"], 151)
        self.assertEqual(
            by_slug["qhc19_c"]["compose_benchmark"]["maximum_mass_msun"],
            2.18,
        )
        self.assertFalse(by_slug["gm1y6"]["expected_optional_files"]["eos.mr"])
        ordering_models = {
            slug: model["ordering_analysis"]
            for slug, model in by_slug.items()
            if "ordering_analysis" in model
        }
        self.assertEqual(set(ordering_models), {"bsk26", "gm1y6"})
        for ordering in ordering_models.values():
            self.assertEqual(
                ordering["acceptance_policy"], acquire.ORDERING_ACCEPTANCE_POLICY
            )
            self.assertTrue(ordering["analysis_policy_rationale"].strip())
        self.assertEqual(
            ordering_models["gm1y6"]["analysis_policy"],
            "diagnostic_keep_later_monotone_subsequence",
        )
        self.assertIn(
            "core-side", ordering_models["gm1y6"]["analysis_policy_rationale"]
        )
        self.assertEqual(
            {
                model["slug"]
                for model in models
                if model["literature_assessment"]["comparability"] == "like_for_like"
            },
            {"bsk26", "qhc19_c"},
        )

    def test_offline_reuse_is_deterministic_and_never_calls_network(self) -> None:
        payload = self.apr_payload()
        model = self.synthetic_model("apr", payload)
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary)
            target = acquire.archive_path(raw_root, model)
            target.parent.mkdir(parents=True)
            target.write_bytes(payload)

            def blocked_opener(*_args: object, **_kwargs: object) -> Any:
                raise AssertionError("offline reuse attempted network access")

            first = acquire.acquire_model(
                model,
                raw_root=raw_root,
                required_members=self.required_members(),
                offline=True,
                opener=blocked_opener,
            )
            sidecar = target.parent / "download.json"
            first_sidecar = sidecar.read_bytes()
            second = acquire.acquire_model(
                model,
                raw_root=raw_root,
                required_members=self.required_members(),
                offline=True,
                opener=blocked_opener,
            )
            self.assertEqual(first["status"], "verified_existing")
            self.assertEqual(second["status"], "verified_existing")
            self.assertEqual(sidecar.read_bytes(), first_sidecar)
            manifest = json.loads(first_sidecar)
            self.assertFalse(manifest["determinism"]["retrieval_timestamp_recorded"])
            self.assertEqual(
                manifest["verification"]["sha256"],
                hashlib.sha256(payload).hexdigest(),
            )

    def test_download_is_staged_verified_and_then_offline_reusable(self) -> None:
        payload = self.apr_payload()
        model = self.synthetic_model("apr", payload)
        calls: list[tuple[str, float]] = []

        def fake_opener(request: Any, *, timeout: float) -> io.BytesIO:
            calls.append((request.full_url, timeout))
            return io.BytesIO(payload)

        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary)
            result = acquire.acquire_model(
                model,
                raw_root=raw_root,
                required_members=self.required_members(),
                offline=False,
                opener=fake_opener,
            )
            target = acquire.archive_path(raw_root, model)
            self.assertEqual(result["status"], "downloaded")
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(calls[0][0], model["archive"]["url"])
            self.assertFalse(any(target.parent.glob("*.part")))

            def blocked_opener(*_args: object, **_kwargs: object) -> Any:
                raise AssertionError("verified offline reuse attempted network access")

            reused = acquire.acquire_model(
                model,
                raw_root=raw_root,
                required_members=self.required_members(),
                offline=True,
                opener=blocked_opener,
            )
            self.assertEqual(reused["status"], "verified_existing")

    def test_offline_missing_and_existing_mismatch_fail_without_overwrite(self) -> None:
        payload = self.apr_payload()
        model = self.synthetic_model("apr", payload)
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary)
            with self.assertRaisesRegex(
                acquire.AcquisitionError, "offline mode cannot acquire missing"
            ):
                acquire.acquire_model(
                    model,
                    raw_root=raw_root,
                    required_members=self.required_members(),
                    offline=True,
                )

            target = acquire.archive_path(raw_root, model)
            target.parent.mkdir(parents=True, exist_ok=True)
            invalid = b"not the pinned archive"
            target.write_bytes(invalid)
            calls = 0

            def unexpected_opener(*_args: object, **_kwargs: object) -> Any:
                nonlocal calls
                calls += 1
                return io.BytesIO(payload)

            with self.assertRaisesRegex(
                acquire.AcquisitionError, "byte count mismatch"
            ):
                acquire.acquire_model(
                    model,
                    raw_root=raw_root,
                    required_members=self.required_members(),
                    offline=False,
                    opener=unexpected_opener,
                )
            self.assertEqual(calls, 0)
            self.assertEqual(target.read_bytes(), invalid)
            self.assertFalse((target.parent / "download.json").exists())

    def test_member_contract_and_selection_errors_are_explicit(self) -> None:
        incomplete = deterministic_zip(self.required_members()[:-1])
        model = self.synthetic_model("apr", incomplete)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "archive.zip"
            path.write_bytes(incomplete)
            with self.assertRaisesRegex(
                acquire.AcquisitionError, "exactly one eos.thermo; found 0"
            ):
                acquire.verify_archive(
                    path, model, required_members=self.required_members()
                )
        with self.assertRaisesRegex(acquire.AcquisitionError, "unknown model slug"):
            acquire.selected_models(self.config, ("does_not_exist",))
        with self.assertRaisesRegex(acquire.AcquisitionError, "duplicate slugs"):
            acquire.selected_models(self.config, ("apr", "apr"))

    def test_ordering_registry_requires_declared_acceptance_and_rationale(
        self,
    ) -> None:
        for field, replacement, message in (
            ("acceptance_policy", "post_hoc_radius_tolerance", "acceptance_policy"),
            ("analysis_policy_rationale", "", "analysis_policy_rationale"),
        ):
            with self.subTest(field=field):
                config = copy.deepcopy(self.config)
                bsk = next(
                    model for model in config["models"] if model["slug"] == "bsk26"
                )
                bsk["ordering_analysis"][field] = replacement
                with self.assertRaisesRegex(acquire.AcquisitionError, message):
                    acquire.validate_config(config)


class ComposeCampaignHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = acquire.load_config(CONFIG_PATH)

    def test_causal_threshold_uses_an_explicit_numeric_tolerance(self) -> None:
        tolerance = runner.CAUSALITY_THRESHOLD_TOLERANCE
        self.assertTrue(runner._cs2_within_causal_threshold(1.0))
        self.assertTrue(runner._cs2_within_causal_threshold(1.0 + tolerance / 2.0))
        self.assertFalse(runner._cs2_within_causal_threshold(1.0 + 2.0 * tolerance))
        self.assertFalse(runner._cs2_within_causal_threshold(0.0))
        self.assertFalse(runner._cs2_within_causal_threshold(float("nan")))

    def test_sequence_merge_deduplicates_machine_precision_grid_overlap(
        self,
    ) -> None:
        pressure = 100.0
        epsilon = np.finfo(float).eps
        near = pressure * (1.0 + 32.0 * epsilon)
        distinct = pressure * (1.0 + 256.0 * epsilon)

        def sequence_at(value: float) -> Any:
            attempt = runner.SequenceAttempt(
                central_pressure_mev_fm3=value,
                status="failed",
                star=None,
                reason="synthetic",
                reason_code="synthetic",
            )
            return runner.SequenceResult(
                model_name="synthetic",
                attempts=(attempt,),
                status="partial",
                boundary_status="synthetic",
            )

        baseline = sequence_at(pressure)
        merged = runner._merge_sequences(
            baseline,
            (baseline, sequence_at(near), sequence_at(distinct)),
        )
        self.assertEqual(len(merged.attempts), 2)
        self.assertEqual(
            merged.attempts[0].central_pressure_mev_fm3,
            pressure,
        )
        self.assertEqual(
            merged.attempts[1].central_pressure_mev_fm3,
            distinct,
        )

    def test_material_ordering_span_is_reported_but_catalogue_gates_acceptance(
        self,
    ) -> None:
        gm = next(model for model in self.config["models"] if model["slug"] == "gm1y6")

        def metrics(
            *, radius_1_4: float, pre_peak_decreases: int = 0
        ) -> dict[str, Any]:
            return {
                "sampled_peak": {"mass_msun": 2.2922, "radius_km": 12.13},
                "at_1_4_msun": {"mass_msun": 1.4, "radius_km": radius_1_4},
                "peak_bracketed_by_sampled_central_densities": True,
                "pre_peak_mass_decrease_count": pre_peak_decreases,
            }

        baseline = metrics(radius_1_4=13.7570763)
        alternative = metrics(radius_1_4=13.8650275)
        assessment = runner._ordering_analysis_assessment(
            gm,
            baseline,
            alternative,
            baseline_remaining_failures=0,
            alternative_remaining_failures=0,
        )
        self.assertTrue(assessment["acceptance_gate_passed"])
        self.assertTrue(assessment["catalogue_consistency"]["both_reductions_passed"])
        self.assertFalse(assessment["delta_within_nominal_tolerance"])
        self.assertEqual(
            assessment["classification"], "material_source_seam_systematic"
        )
        span = assessment["conditional_radius_span_at_1_4_msun_km"]
        self.assertAlmostEqual(span["span"], 0.1079512)
        self.assertFalse(assessment["physical_transition_resolved"])
        self.assertFalse(assessment["conditional_span_is_statistical_uncertainty"])

        summary = {
            "model_id": gm["model_id"],
            "metrics": baseline,
            "ordering": {
                "sensitivity": {
                    "baseline_policy": gm["ordering_analysis"]["analysis_policy"],
                    "alternative_policy": "diagnostic_monotone_subsequence",
                    "alternative_metrics": alternative,
                    **assessment,
                }
            },
        }
        report_line = runner._ordering_systematic_report_line(summary)
        self.assertIn("conditionally reported, not seam-resolved", report_line)
        self.assertIn("0.1080 km conditional span", report_line)
        self.assertIn("not TOV error, statistical uncertainty", report_line)
        self.assertIn("or a Maxwell construction", report_line)

    def test_ordering_acceptance_rejects_failed_alternative_evidence(self) -> None:
        gm = next(model for model in self.config["models"] if model["slug"] == "gm1y6")
        baseline = {
            "sampled_peak": {"mass_msun": 2.2922, "radius_km": 12.13},
            "at_1_4_msun": {"mass_msun": 1.4, "radius_km": 13.757},
            "peak_bracketed_by_sampled_central_densities": True,
            "pre_peak_mass_decrease_count": 0,
        }
        for name, alternative, remaining_failures in (
            (
                "catalogue",
                {
                    **baseline,
                    "at_1_4_msun": {"mass_msun": 1.4, "radius_km": 14.2},
                },
                0,
            ),
            ("sequence_failure", baseline, 1),
            (
                "pre_peak_decrease",
                {**baseline, "pre_peak_mass_decrease_count": 1},
                0,
            ),
        ):
            with self.subTest(name=name):
                assessment = runner._ordering_analysis_assessment(
                    gm,
                    baseline,
                    alternative,
                    baseline_remaining_failures=0,
                    alternative_remaining_failures=remaining_failures,
                )
                self.assertFalse(assessment["acceptance_gate_passed"])

    def test_branch_metrics_distinguish_bracketing_and_pre_peak_decreases(
        self,
    ) -> None:
        branch = runner.BranchData(
            pressure_mev_fm3=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
            baryon_density_fm3=np.asarray([0.2, 0.3, 0.4, 0.5, 0.6]),
            mass_msun=np.asarray([0.5, 1.0, 0.99, 1.2, 1.1]),
            radius_km=np.asarray([14.0, 13.0, 12.9, 12.0, 11.5]),
            peak_index=3,
            pre_peak_mass_decrease_count=1,
        )
        metrics = runner._branch_metrics(branch)
        self.assertTrue(metrics["peak_bracketed_by_sampled_central_densities"])
        self.assertFalse(metrics["peak_censored_at_upper_density_boundary"])
        self.assertEqual(metrics["post_peak_points"], 1)
        self.assertEqual(metrics["pre_peak_mass_decrease_count"], 1)
        self.assertNotIn("stable", metrics["algorithm"])

    def test_reference_crosscheck_uses_fixed_masses_below_turning_points(
        self,
    ) -> None:
        branch = runner.BranchData(
            pressure_mev_fm3=np.arange(1.0, 7.0),
            baryon_density_fm3=np.linspace(0.2, 0.8, 6),
            mass_msun=np.asarray([0.8, 1.0, 1.4, 1.8, 2.1, 2.05]),
            radius_km=np.asarray([14.0, 13.5, 12.8, 12.0, 11.0, 10.8]),
            peak_index=4,
            pre_peak_mass_decrease_count=0,
        )
        reference = runner.ComposeMassRadiusReference(
            model_id="synthetic",
            source_url="https://example.invalid/eos",
            radius_km=np.asarray([14.1, 13.6, 12.9, 12.1, 11.1, 10.9]),
            mass_msun=np.asarray([0.8, 1.0, 1.4, 1.8, 2.08, 2.0]),
            additional_columns=(),
            header_lines=(),
            source_bytes=1,
            source_sha256="0" * 64,
        )
        comparison = runner._comparison_to_reference(branch, reference)
        fixed = comparison["fixed_mass_comparisons"]
        self.assertEqual(
            [row["mass_msun"] for row in fixed], [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        )
        self.assertTrue(
            all(
                row["mass_msun"] <= comparison["safe_fixed_mass_ceiling_msun"]
                for row in fixed
            )
        )
        self.assertIn("sampled_peak_coordinate_comparison", comparison)
        self.assertFalse(comparison["physical_stability_inferred_for_eos_mr"])

    def test_literature_acceptance_is_gated_only_when_like_for_like(self) -> None:
        by_slug = {model["slug"]: model for model in self.config["models"]}
        deliberately_different = {
            "sampled_peak": {"mass_msun": 9.0, "radius_km": 30.0},
            "at_1_4_msun": {"radius_km": 30.0},
        }
        bsk = runner._literature_check(by_slug["bsk26"], deliberately_different)
        sly = runner._literature_check(by_slug["sly4"], deliberately_different)
        self.assertTrue(bsk["acceptance_required"])
        self.assertFalse(bsk["acceptance_gate_passed"])
        self.assertFalse(sly["acceptance_required"])
        self.assertFalse(sly["numeric_comparison_passed"])
        self.assertTrue(sly["acceptance_gate_passed"])
        self.assertEqual(sly["acceptance_status"], "not_gated_contextual")

    def test_selected_output_cleanup_preserves_unselected_and_unknown_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            derived = root / "derived"
            figures = root / "figures"
            results = root / "results"
            manifest = root / "manifest.json"
            (derived / "bsk26").mkdir(parents=True)
            (derived / "bsk26" / "summary.json").write_text("{}")
            (derived / "sly4").mkdir(parents=True)
            (derived / "sly4" / "keep.json").write_text("{}")
            (figures / "bsk26").mkdir(parents=True)
            (figures / "bsk26" / "old.png").write_bytes(b"png")
            (figures / "comparison").mkdir(parents=True)
            (figures / "comparison" / "old.png").write_bytes(b"png")
            results.mkdir()
            (results / "failure.json").write_text("{}")
            (results / "user-note.txt").write_text("keep")
            manifest.write_text("{}")
            with mock.patch.multiple(
                runner,
                DERIVED_ROOT=derived,
                FIGURE_ROOT=figures,
                RESULTS_ROOT=results,
                MANIFEST_PATH=manifest,
            ):
                runner._prepare_selected_outputs(({"slug": "bsk26"},))
            self.assertFalse((derived / "bsk26").exists())
            self.assertFalse((figures / "bsk26").exists())
            self.assertFalse((figures / "comparison").exists())
            self.assertTrue((derived / "sly4" / "keep.json").is_file())
            self.assertTrue((results / "user-note.txt").is_file())
            self.assertFalse((results / "failure.json").exists())
            self.assertFalse(manifest.exists())

    def test_cleanup_refuses_unexpected_selected_file_types(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = root / "derived" / "bsk26"
            selected.mkdir(parents=True)
            (selected / "do-not-delete.txt").write_text("user data")
            with mock.patch.multiple(
                runner,
                DERIVED_ROOT=root / "derived",
                FIGURE_ROOT=root / "figures",
                RESULTS_ROOT=root / "results",
                MANIFEST_PATH=root / "manifest.json",
            ):
                with self.assertRaisesRegex(RuntimeError, "unexpected file type"):
                    runner._prepare_selected_outputs(({"slug": "bsk26"},))
            self.assertTrue((selected / "do-not-delete.txt").is_file())

    def test_raw_preflight_validates_sidecar_before_output_cleanup(self) -> None:
        required = tuple(self.config["campaign"]["required_archive_members"])
        payload = deterministic_zip((*required, "eos.mr"))
        source = next(
            model for model in self.config["models"] if model["slug"] == "apr"
        )
        model = copy.deepcopy(source)
        digest = hashlib.sha256(payload).hexdigest()
        model["archive"].update({"bytes": len(payload), "sha256": digest})
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            directory = raw / "apr"
            directory.mkdir()
            archive = directory / "archive.zip"
            archive.write_bytes(payload)
            sidecar = {
                "archive": {"local_filename": "archive.zip"},
                "verification": {"bytes": len(payload), "sha256": digest},
            }
            (directory / "download.json").write_text(json.dumps(sidecar))
            runner._preflight_raw_inputs(
                (model,), raw_root=raw, required_members=required
            )
            sidecar["verification"]["sha256"] = "0" * 64
            (directory / "download.json").write_text(json.dumps(sidecar))
            with self.assertRaisesRegex(
                runner.AcquisitionError, "does not describe the pinned archive"
            ):
                runner._preflight_raw_inputs(
                    (model,), raw_root=raw, required_members=required
                )

    def test_manifest_enumerates_only_selected_raw_and_generated_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            derived = root / "derived"
            figures = root / "figures"
            results = root / "results"
            manifest_path = root / "manifest.json"
            selected_raw = raw / "bsk26"
            selected_raw.mkdir(parents=True)
            (selected_raw / "archive.zip").write_bytes(b"canonical")
            (selected_raw / "download.json").write_text("{}")
            legacy = raw / "sly4"
            legacy.mkdir(parents=True)
            (legacy / "eos.zip").write_bytes(b"legacy")
            (derived / "bsk26").mkdir(parents=True)
            (derived / "bsk26" / "summary.json").write_text("{}")
            (figures / "bsk26").mkdir(parents=True)
            (figures / "bsk26" / "plot.png").write_bytes(b"png")
            (figures / "comparison").mkdir(parents=True)
            (figures / "comparison" / "all_calculated_mass_radius.png").write_bytes(
                b"png"
            )
            results.mkdir()
            (results / "all_models_summary.csv").write_text("slug\nbsk26\n")
            (results / "acceptance.json").write_text("{}")
            (results / "report.md").write_text("# report\n")
            summaries = (
                {"slug": "bsk26", "archive": {"archive_filename": "archive.zip"}},
            )
            with mock.patch.multiple(
                runner,
                DERIVED_ROOT=derived,
                FIGURE_ROOT=figures,
                RESULTS_ROOT=results,
                MANIFEST_PATH=manifest_path,
            ):
                runner._manifest(summaries, CONFIG_PATH, raw_root=raw)
            manifest = json.loads(manifest_path.read_text())
            raw_roles = {item["role"] for item in manifest["canonical_raw_files"]}
            self.assertEqual(
                raw_roles,
                {"pinned_raw_archive", "acquisition_verification_sidecar"},
            )
            all_paths = [item["path"] for item in manifest["files"]]
            self.assertFalse(any(path.endswith("eos.zip") for path in all_paths))
            code_paths = [item["path"] for item in manifest["exact_code_inputs"]]
            self.assertIn("experiments/compose_comparison/run.py", code_paths)
            self.assertIn("experiments/compose_comparison/acquire.py", code_paths)
            self.assertTrue(any(path.endswith("stellar.py") for path in code_paths))
            software = manifest["software"]
            self.assertEqual(
                software["neutron_star_eos_toolkit"], runner.toolkit_version
            )
            self.assertEqual(software["scipy"], runner.scipy_version)
            constants = software["stellar_physical_constants"]
            self.assertEqual(constants["authority"], runner.STELLAR_CONSTANT_AUTHORITY)
            self.assertEqual(
                constants["gravity_conversion_Msun_per_km3_per_MeV_fm3"],
                runner.GRAVITY_CONVERSION,
            )


if __name__ == "__main__":
    unittest.main()
