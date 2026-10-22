"""Standard-library checks for the portable SegPaged suite and report boundary.

No torch, GPU, checkpoint, download or subprocess execution is required. Child
reports below are deliberately small fixtures, not experimental measurements.
"""

from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def _load(filename, name):
    specification = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


suite = _load("benchmark_SegPagedAttention.py", "_segpaged_suite_stdlib_test")
reporter = _load("report_segpaged_benchmark.py", "_segpaged_report_stdlib_test")


class SuiteArgumentsTest(unittest.TestCase):
    def assert_rejected(self, arguments, message=None):
        output = io.StringIO()
        with redirect_stderr(output), self.assertRaises(SystemExit) as error:
            suite.arguments(arguments)
        self.assertEqual(error.exception.code, 2)
        if message:
            self.assertIn(message, output.getvalue())

    def test_default_components_never_implicitly_run_model(self):
        arguments = suite.arguments([])
        self.assertEqual(arguments.components, ["operator", "public", "head-sharing"])
        self.assertNotIn("cache", arguments.components)
        self.assertIsNone(arguments.model_path)
        jobs = suite.commands(arguments, Path("/unused/suite"))
        self.assertEqual([name for name, _, _ in jobs], arguments.components)
        for _, command, _ in jobs:
            self.assertNotIn("--model-path", command)
            self.assertFalse(any("benchmark_segpaged_workflow.py" in part for part in command))

    def test_model_path_alone_does_not_enable_workflow(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            arguments = suite.arguments(["--model-path", checkpoint])
            jobs = suite.commands(arguments, Path("/unused/suite"))
        self.assertNotIn("workflow", [name for name, _, _ in jobs])

    def test_cpu_allows_only_cache_and_head_sharing_components(self):
        for components in (["cache"], ["head-sharing"], ["cache", "head-sharing"],
                           ["head-sharing", "cache"]):
            with self.subTest(components=components):
                arguments = suite.arguments(["--device", "cpu", "--components", *components])
                jobs = suite.commands(arguments, Path("/unused/suite"))
                self.assertEqual([name for name, _, _ in jobs], components)
                for name, command, result in jobs:
                    self.assertEqual(command[command.index("--device") + 1], "cpu")
                    self.assertEqual(result, Path("/unused/suite") / (name + ".json"))
        for components in (["operator"], ["public"], ["cache", "operator"], ["workflow"],
                           ["head-sharing", "public"], ["head-sharing", "workflow"]):
            with self.subTest(components=components):
                self.assert_rejected(["--device", "cpu", "--components", *components], "CPU")

    def test_head_sharing_command_passes_explicit_lengths_ratios_and_budget(self):
        arguments = suite.arguments([
            "--components", "head-sharing", "--device", "cpu",
            "--head-sharing-lengths", "8K", "16K", "--local-head-ratios", "0", "0.75", "1",
            "--budget-requests", "8", "--cache-lengths", "32K", "--operator-lengths", "64K",
        ])
        jobs = suite.commands(arguments, Path("/unused/suite"))
        self.assertEqual(len(jobs), 1)
        name, command, result = jobs[0]
        self.assertEqual(name, "head-sharing")
        self.assertEqual(Path(command[1]).name, "benchmark_segpaged_head_sharing.py")
        self.assertEqual(command[command.index("--lengths") + 1:command.index("--output")],
                         ["8K", "16K"])
        ratio_start = command.index("--local-head-ratios") + 1
        self.assertEqual(command[ratio_start:command.index("--budget-requests")],
                         ["0.0", "0.75", "1.0"])
        self.assertEqual(command[command.index("--budget-requests") + 1], "8")
        self.assertEqual(command[-2:], ["--output", str(result)])
        self.assertNotIn("32K", command)
        self.assertNotIn("64K", command)
        self.assertNotIn("--model-path", command)

    def test_head_sharing_full_preset_lengths_are_explicit_but_overridable(self):
        arguments = suite.arguments(["--components", "head-sharing", "--preset", "full"])
        command = suite.commands(arguments, Path("/unused/suite"))[0][1]
        self.assertEqual(command[command.index("--lengths") + 1:command.index("--output")],
                         ["64K", "128K", "256K", "440K"])
        arguments = suite.arguments([
            "--components", "head-sharing", "--preset", "full", "--head-sharing-lengths", "4K",
        ])
        command = suite.commands(arguments, Path("/unused/suite"))[0][1]
        self.assertEqual(command[command.index("--lengths") + 1:command.index("--output")], ["4K"])
        smoke = suite.commands(suite.arguments(["--components", "head-sharing"]),
                               Path("/unused/suite"))[0][1]
        self.assertNotIn("--lengths", smoke)

    def test_head_sharing_invalid_ratios_and_budget_are_rejected(self):
        for option in (["--budget-requests", "0"], ["--budget-requests", "-1"],
                       ["--local-head-ratios", "-0.1"], ["--local-head-ratios", "1.1"],
                       ["--local-head-ratios", "nan"], ["--local-head-ratios", "inf"]):
            with self.subTest(option=option):
                self.assert_rejected(["--components", "head-sharing", *option], "required")

    def test_head_sharing_document_size_and_geometry_reach_child_command(self):
        arguments = suite.arguments([
            "--components", "head-sharing", "--document-kv-gib", "3",
            "--head-sharing-heads", "8", "--head-sharing-head-dim", "128",
            "--head-sharing-page-size", "256", "--budget-requests", "4",
        ])
        command = suite.commands(arguments, Path("/unused/suite"))[0][1]
        expected = {"--document-kv-gib": "3.0", "--heads": "8", "--head-dim": "128",
                    "--page-size": "256", "--budget-requests": "4"}
        for option, value in expected.items():
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertNotIn("--lengths", command)
        self.assertNotIn("--model-path", command)

    def test_head_sharing_document_size_overrides_full_preset_length_default(self):
        arguments = suite.arguments([
            "--components", "head-sharing", "--preset", "full", "--document-kv-gib", "3",
        ])
        command = suite.commands(arguments, Path("/unused/suite"))[0][1]
        self.assertIn("--document-kv-gib", command)
        self.assertNotIn("--lengths", command)

    def test_head_sharing_document_size_must_be_finite_positive_and_unambiguous(self):
        for value in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(value=value):
                self.assert_rejected([
                    "--components", "head-sharing", "--document-kv-gib=" + value,
                ], "finite and positive")
        self.assert_rejected([
            "--components", "head-sharing", "--document-kv-gib", "3",
            "--head-sharing-lengths", "64K",
        ], "OR")

    def test_head_sharing_dimensions_require_positive_integers(self):
        for option in ("--head-sharing-heads", "--head-sharing-head-dim", "--head-sharing-page-size"):
            for value in ("0", "-1", "1.5"):
                with self.subTest(option=option, value=value):
                    self.assert_rejected(["--components", "head-sharing", option, value])

    def test_workflow_requires_explicit_existing_directory(self):
        self.assert_rejected(["--components", "workflow"], "model-path")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            file = root / "not-a-model-directory"
            file.write_text("fixture")
            for invalid in (file, root / "absent"):
                with self.subTest(path=invalid):
                    self.assert_rejected(
                        ["--components", "workflow", "--model-path", str(invalid)],
                        "model-path",
                    )

    def test_explicit_workflow_command_preserves_choices_and_recomputed_control(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            arguments = suite.arguments([
                "--components", "workflow", "--model-path", checkpoint,
                "--model-context-tokens", "4096", "--model-chunks", "8",
                "--datasets", "hotpotqa", "2wikimqa", "--samples", "2",
                "--model-warmup", "2", "--model-repeats", "4",
                "--max-new-tokens", "50", "--dataset-dir", checkpoint,
            ])
            jobs = suite.commands(arguments, Path("/unused/suite"))
        self.assertEqual(len(jobs), 1)
        name, command, result = jobs[0]
        self.assertEqual(name, "workflow")
        self.assertIn("--recomputed", command)
        self.assertEqual(command[command.index("--context-tokens") + 1], "4096")
        self.assertEqual(command[command.index("--chunks") + 1], "8")
        self.assertEqual(command[command.index("--max-new-tokens") + 1], "50")
        self.assertEqual(command[command.index("--warmup") + 1], "2")
        self.assertEqual(command[command.index("--repeats") + 1], "4")
        self.assertEqual(command[-2:], ["--output", str(result)])

    def test_workflow_rejects_nonzero_logical_device(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            self.assert_rejected([
                "--components", "workflow", "--model-path", checkpoint,
                "--device", "cuda:1",
            ], "CUDA_VISIBLE_DEVICES")

    def test_duplicate_and_unknown_components_are_rejected(self):
        self.assert_rejected(["--components", "cache", "cache"], "unique")
        self.assert_rejected(["--components", "unrecognized"])

    def test_invalid_repetition_and_page_parameters_are_rejected(self):
        for arguments in (
            ["--warmup", "0"], ["--warmup", "-1"], ["--repeats", "1"],
            ["--requests", "1", "0"], ["--requests", "-1"],
            ["--cache-page-size", "0"], ["--cache-page-size", "-2"],
        ):
            with self.subTest(arguments=arguments):
                self.assert_rejected(arguments)

    def test_invalid_model_shapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            for option in (
                "--samples", "--model-context-tokens", "--model-chunks",
                "--max-new-tokens", "--model-warmup", "--model-repeats",
            ):
                with self.subTest(option=option):
                    self.assert_rejected([
                        "--components", "workflow", "--model-path", checkpoint,
                        option, "0",
                    ], "positive")

    def test_invalid_device_is_not_silently_treated_as_cpu(self):
        for device in ("banana", "cuda:not-a-number", "cuda:-1"):
            with self.subTest(device=device):
                self.assert_rejected(["--components", "cache", "--device", device])

    def test_operator_public_and_cache_commands_remain_separate(self):
        arguments = suite.arguments([
            "--components", "public", "cache", "operator",
            "--operator-lengths", "64K", "128K", "--cache-lengths", "16K",
            "--cache-page-size", "128", "--requests", "1", "4",
        ])
        jobs = suite.commands(arguments, Path("/unused/suite"))
        self.assertEqual([name for name, _, _ in jobs], ["public", "cache", "operator"])
        for name, command, result in jobs:
            self.assertEqual(command[-2:], ["--output", str(result)])
            if name == "cache":
                self.assertEqual(command[command.index("--page-size") + 1], "128")
                self.assertIn("16K", command)
                self.assertNotIn("64K", command)
            else:
                self.assertEqual(command[command.index("--entry") + 1],
                                 "public" if name == "public" else "prevalidated")
                self.assertIn("64K", command)
                self.assertIn("128K", command)
                self.assertNotIn("16K", command)

    def test_source_hashes_include_actual_workflow_implementation(self):
        hashes = suite.source_hashes()
        expected = [
            "python/sglang/srt/layers/attention/redknot/driver.py",
            "python/sglang/srt/layers/attention/redknot/driver_batched.py",
            "python/sglang/srt/layers/attention/redknot/managed_query.py",
            "test/srt/redknot/benchmark_segpaged_workflow.py",
            "test/srt/redknot/benchmark_segpaged_head_sharing.py",
            "test/srt/redknot/report_segpaged_benchmark.py",
        ]
        for name in expected:
            with self.subTest(name=name):
                self.assertIn(name, hashes)
                self.assertEqual(hashes[name], hashlib.sha256((suite.REPO / name).read_bytes()).hexdigest())

    def test_nonempty_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            sentinel = Path(temporary) / "cache.json"
            sentinel.write_text("previous evidence")
            with mock.patch.object(suite.subprocess, "Popen") as spawn:
                with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                    suite.main(["--device", "cpu", "--components", "cache",
                                "--output-dir", temporary])
                spawn.assert_not_called()
            self.assertEqual(sentinel.read_text(), "previous evidence")
            self.assertFalse((Path(temporary) / "manifest.json").exists())


class VerifiedReportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "suite"
        self.directory.mkdir()
        self.manifest = {
            "schema": "segpaged-suite-v1", "status": "complete",
            "created_utc": "fixture-not-an-experiment", "source_sha256": {}, "jobs": [],
        }

    def write_manifest(self):
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest))

    def add_report(self, component="cache", payload=None, filename=None):
        path = self.directory / (filename or component + ".json")
        if payload is None:
            payload = {"status": "complete", "cases": []}
        path.write_text(json.dumps(payload))
        job = {
            "component": component, "status": "complete", "returncode": 0,
            "result": path.name, "log": component + ".log",
            "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        self.manifest["jobs"].append(job)
        self.write_manifest()
        return path, job

    def test_complete_report_is_verified_and_rendered_without_plotting(self):
        self.add_report()
        manifest, reports = reporter.load_verified(self.directory)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(reports["cache"]["cases"], [])
        with mock.patch.object(reporter, "make_plots") as plots:
            result = reporter.render(self.directory)
            plots.assert_not_called()
        text = result.read_text()
        self.assertIn("Concurrent serving QPS: NOT MEASURED", text)
        self.assertIn("Not run in this suite invocation", text)
        self.assertIn(self.manifest["jobs"][0]["result_sha256"], text)

    def test_incomplete_or_wrong_schema_manifest_is_rejected(self):
        self.add_report()
        for field, value in (("status", "running"), ("status", "failed"),
                             ("schema", "other-schema")):
            with self.subTest(field=field, value=value):
                original = self.manifest[field]
                self.manifest[field] = value
                self.write_manifest()
                with self.assertRaises(ValueError):
                    reporter.load_verified(self.directory)
                self.manifest[field] = original

    def test_failed_or_unfinished_job_is_rejected(self):
        _, job = self.add_report()
        for field, value in (("status", "failed"), ("status", "running"),
                             ("returncode", 1), ("returncode", -15)):
            with self.subTest(field=field, value=value):
                original = job[field]
                job[field] = value
                self.write_manifest()
                with self.assertRaises(ValueError):
                    reporter.load_verified(self.directory)
                job[field] = original

    def test_altered_result_bytes_are_rejected_before_report_write(self):
        path, _ = self.add_report()
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            reporter.render(self.directory)
        self.assertFalse((self.directory / "REPORT.md").exists())

    def test_checksum_correct_but_failed_child_is_rejected(self):
        self.add_report(payload={"status": "failed", "cases": []})
        with self.assertRaisesRegex(ValueError, "Incomplete child"):
            reporter.load_verified(self.directory)

    def test_duplicate_component_is_rejected(self):
        self.add_report()
        self.add_report(component="cache", filename="second-cache.json")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            reporter.load_verified(self.directory)

    def test_same_result_cannot_be_relabelled_as_another_component(self):
        _, first = self.add_report()
        duplicate = dict(first, component="operator")
        self.manifest["jobs"].append(duplicate)
        self.write_manifest()
        with self.assertRaises(ValueError):
            reporter.load_verified(self.directory)

    def test_empty_manifest_jobs_do_not_qualify_as_complete_evidence(self):
        self.write_manifest()
        with self.assertRaises(ValueError):
            reporter.load_verified(self.directory)

    def test_parent_traversal_and_absolute_outside_paths_are_rejected(self):
        path, job = self.add_report()
        outside = self.directory.parent / "outside.json"
        outside.write_bytes(path.read_bytes())
        for name in ("../outside.json", str(outside.resolve())):
            with self.subTest(name=name):
                job["result"] = name
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "suite directory"):
                    reporter.load_verified(self.directory)

    def test_symlink_to_outside_evidence_is_rejected(self):
        path, job = self.add_report()
        outside = self.directory.parent / "outside.json"
        outside.write_bytes(path.read_bytes())
        link = self.directory / "external.json"
        link.symlink_to(outside)
        job["result"] = link.name
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "suite directory"):
            reporter.load_verified(self.directory)

    def test_nested_result_path_is_rejected(self):
        path, job = self.add_report()
        subdirectory = self.directory / "nested"
        subdirectory.mkdir()
        child = subdirectory / "cache.json"
        child.write_bytes(path.read_bytes())
        job["result"] = "nested/cache.json"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "suite directory"):
            reporter.load_verified(self.directory)

    def test_model_rows_exclude_warmup_and_keep_raw_arm_medians(self):
        payload = {
            "status": "complete", "cases": [{
                "dataset": "hotpotqa", "source_index": 7,
                "runs": [
                    {"arm": "native", "warmup": True, "ttft_seconds": 999,
                     "request_seconds": 9999},
                    {"arm": "managed_paged", "warmup": True, "ttft_seconds": 999,
                     "request_seconds": 9999},
                    {"arm": "native", "warmup": False, "ttft_seconds": 2,
                     "request_seconds": 4},
                    {"arm": "native", "ttft_seconds": 4, "request_seconds": 8},
                    {"arm": "managed_paged", "warmup": False, "ttft_seconds": 6,
                     "request_seconds": 12},
                ],
            }],
        }
        before = copy.deepcopy(payload)
        rows = reporter.model_rows(payload)
        self.assertEqual(rows, [
            ["hotpotqa", 7, "managed_paged", 1, "6.0000", "12.0000"],
            ["hotpotqa", 7, "native", 2, "3.0000", "6.0000"],
        ])
        self.assertEqual(payload, before)
        self.add_report("workflow", payload)
        text = reporter.render(self.directory).read_text()
        self.assertIn("Actual model workflow", text)
        self.assertIn("3.0000", text)
        self.assertNotIn("999", text)
        self.assertIn("diagnostic query logits", text)

    def test_warmup_only_case_never_creates_a_model_measurement(self):
        payload = {"cases": [{"dataset": "fixture", "source_index": 0, "runs": [
            {"arm": "native", "warmup": True, "ttft_seconds": 99, "request_seconds": 99},
        ]}]}
        self.assertEqual(reporter.model_rows(payload), [])

    @staticmethod
    def head_sharing_case(*, local_ratio, independent=4, shared=4, censored=False,
                          verified=True, budget_gib=12):
        """Minimal schema fixture. These chosen values are not GPU measurements."""
        arms = []
        for backend, admitted in (("independent", independent), ("head_shared", shared)):
            is_shared = backend == "head_shared"
            arms.append({
                "backend": backend, "admitted_requests": admitted,
                "termination": "max_requests_guard" if censored and is_shared else "capacity_error",
                "admission_is_censored": censored and is_shared,
                "source_cache": {"allocated_bytes": int(3 * 2**30 * local_ratio) if is_shared else 0},
                "admission_steps": ([{"admitted": True, "resident_requests": 4,
                                     "after": {"allocated_bytes": 12 * 2**30}}]
                                    if admitted >= 4 else []),
                "correctness": {"verified_requests": ([{"attention_max_abs_error": 0.0002}]
                                                        if verified and admitted else [])},
            })
        return {
            "document_tokens": 65536, "document_full_head_logical_gib": 3.0,
            "config": {"local_head_ratio_actual": local_ratio,
                       "reserved_bytes": int(budget_gib * 2**30)},
            "summary": {"independent_admitted": independent, "head_shared_admitted": shared,
                        "observed_resident_request_ratio": shared / independent if independent else None,
                        "any_censored": censored, "throughput_or_qps_measured": False},
            "arms": arms,
        }

    def test_head_sharing_report_keeps_capacity_and_qps_distinct(self):
        payload = {"status": "complete", "cases": [
            self.head_sharing_case(local_ratio=0.0),
            self.head_sharing_case(local_ratio=0.75, shared=12),
        ]}
        raw, _ = self.add_report("head-sharing", payload)
        before = raw.read_bytes()
        with mock.patch.object(reporter, "make_plots") as plot:
            text = reporter.render(self.directory).read_text()
            plot.assert_not_called()
        self.assertIn("Same text, shared local heads, private global heads", text)
        self.assertIn("Capacity ratio (not QPS)", text)
        self.assertIn("A resident-capacity multiplier is not a QPS multiplier", text)
        self.assertIn("| 65536 | 3.000 | 0% | 12.0000 | 4 | 4 | 1.000x | False |", text)
        self.assertIn("| 65536 | 3.000 | 75% | 12.0000 | 4 | 12 | 3.000x | False |", text)
        self.assertIn("Concurrent serving QPS: NOT MEASURED", text)
        self.assertNotIn("QPS speedup", text)
        self.assertEqual(raw.read_bytes(), before)

    def test_head_sharing_report_marks_censored_and_unverified_results(self):
        self.add_report("head-sharing", {"status": "complete", "cases": [
            self.head_sharing_case(local_ratio=0.75, shared=12, censored=True, verified=False),
        ]})
        text = reporter.render(self.directory).read_text()
        self.assertIn("Guard-limited runs are censored", text)
        self.assertIn("| 4 | 12 | 3.000x | True |", text)
        self.assertIn("max_requests_guard", text)
        self.assertIn("Not verified", text)
        self.assertNotIn("verified accuracy", text)

    def test_head_sharing_zero_independent_capacity_has_no_infinite_multiplier(self):
        self.add_report("head-sharing", {"status": "complete", "cases": [
            self.head_sharing_case(local_ratio=1.0, independent=0, shared=2,
                                   verified=False, budget_gib=4),
        ]})
        text = reporter.render(self.directory).read_text()
        self.assertIn("| 65536 | 3.000 | 100% | 4.0000 | 0 | 2 | N/A | False |", text)
        self.assertIn("Not admitted", text)
        self.assertIn("Not verified", text)
        self.assertNotIn("infx", text)
        self.assertNotIn("nanx", text)


class SuiteExecutionTest(unittest.TestCase):
    """Drive the wrapper state machine with an inert, in-memory child process."""

    def run_fixture(self, directory, *, returncode=0, child_status="complete"):
        def spawn(command, **kwargs):
            if returncode == 0:
                destination = Path(command[command.index("--output") + 1])
                destination.write_text(json.dumps({"status": child_status, "cases": []}))
            child = mock.Mock()
            child.stdout = io.StringIO("fixture child log; no benchmark ran\n")
            child.wait.return_value = returncode
            child.poll.return_value = returncode
            return child

        with mock.patch.object(suite.subprocess, "Popen", side_effect=spawn), \
                mock.patch.object(suite.platform, "platform", return_value="fixture-platform"), \
                mock.patch.dict(sys.modules, {"report_segpaged_benchmark": reporter}), \
                redirect_stdout(io.StringIO()):
            return suite.main([
                "--components", "cache", "--device", "cpu", "--output-dir", str(directory),
            ])

    def test_success_records_checksum_commands_log_and_complete_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "fresh"
            self.assertEqual(self.run_fixture(directory), 0)
            manifest, reports = reporter.load_verified(directory)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(list(reports), ["cache"])
            job = manifest["jobs"][0]
            self.assertEqual(job["returncode"], 0)
            self.assertIn("--device", job["command"])
            self.assertIn("cpu", job["command"])
            self.assertEqual(job["result_sha256"],
                             hashlib.sha256((directory / "cache.json").read_bytes()).hexdigest())
            self.assertIn("fixture child log", (directory / job["log"]).read_text())
            self.assertTrue((directory / "REPORT.md").is_file())

    def test_failed_child_preserves_failed_manifest_and_does_not_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "fresh"
            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                self.run_fixture(directory, returncode=7)
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["jobs"][0]["status"], "failed")
            self.assertEqual(manifest["jobs"][0]["returncode"], 7)
            self.assertEqual(manifest["failure"]["type"], "RuntimeError")
            self.assertFalse((directory / "REPORT.md").exists())
            with self.assertRaises(ValueError):
                reporter.load_verified(directory)

    def test_zero_exit_without_complete_report_still_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "fresh"
            with self.assertRaisesRegex(RuntimeError, "complete report"):
                self.run_fixture(directory, child_status="running")
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["jobs"][0]["status"], "failed")
            self.assertFalse((directory / "REPORT.md").exists())


if __name__ == "__main__":
    unittest.main()
