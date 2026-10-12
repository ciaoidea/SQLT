"""Installer contracts for actual backend choice, upgrade, and safe fallback."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch_setup as setup

PYTHON = Path("/test/.venv/bin/python")
MODERN_GPU = dict(name="RTX test", driver="595.91.07", capability=(8, 6))
CPU = dict(version="2.8.0+cpu", cpu_usable=True, cuda_version=None,
           cuda_usable=False, cuda_busy=False)
CUDA = dict(version="2.8.0+cu128", cpu_usable=True, cuda_version="12.8",
            cuda_usable=True, cuda_busy=False)


class BackendTests(unittest.TestCase):
    def test_modern_gpu_selects_cuda_128(self):
        self.assertEqual(setup.preferred_build("auto", [MODERN_GPU])[0], "cu128")

    def test_cuda_12_minor_compatibility_uses_126_and_requires_probe(self):
        gpu = dict(name="older", driver="535.54.03", capability=(7, 0))
        self.assertEqual(setup.preferred_build("auto", [gpu])[0], "cu126")

    def test_unsupported_driver_and_gpu_use_cpu(self):
        for gpu in (dict(name="old driver", driver="510.0", capability=(8, 6)),
                    dict(name="old GPU", driver="595.0", capability=(3, 5)),
                    dict(name="new GPU with old driver", driver="560.0", capability=(12, 0))):
            with self.subTest(gpu=gpu):
                self.assertEqual(setup.preferred_build("auto", [gpu])[0], "cpu")

    def test_forced_cpu_ignores_detected_gpu(self):
        self.assertEqual(setup.preferred_build("cpu", [MODERN_GPU])[0], "cpu")

    def test_no_gpu_forced_cuda_still_falls_back_to_cpu(self):
        self.assertEqual(setup.preferred_build("cuda", [])[0], "cpu")

    def test_invalid_requested_device_is_rejected(self):
        with self.assertRaises(ValueError):
            setup.preferred_build("metal", [])

    def test_detection_parses_gpu_capability_without_torch(self):
        result = subprocess.CompletedProcess([], 0, "RTX test, 595.91.07, 8.6\n", "")
        with patch("torch_setup.shutil.which", return_value="/bin/nvidia-smi"), \
                patch("torch_setup.subprocess.run", return_value=result):
            self.assertEqual(setup.nvidia_devices(), [MODERN_GPU])

    def test_failed_or_malformed_detection_does_not_select_cuda(self):
        for result in (subprocess.CompletedProcess([], 1, "", "driver missing"),
                       subprocess.CompletedProcess([], 0, "GPU, N/A, N/A\n", "")):
            with patch("torch_setup.shutil.which", return_value="/bin/nvidia-smi"), \
                    patch("torch_setup.subprocess.run", return_value=result):
                self.assertEqual(setup.nvidia_devices(), [])
        with patch("torch_setup.shutil.which", return_value="/bin/nvidia-smi"), \
                patch("torch_setup.subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 10)):
            self.assertEqual(setup.nvidia_devices(), [])


class InstallationTests(unittest.TestCase):
    def run_install(self, probes, devices=(), requested="auto", install_error=None):
        with patch("torch_setup.nvidia_devices", return_value=list(devices)) as detection, \
                patch("torch_setup.probe_torch", side_effect=probes) as probe, \
                patch("torch_setup.install_build", side_effect=install_error) as install, \
                contextlib.redirect_stdout(io.StringIO()):
            report = setup.ensure_torch(PYTHON, requested)
        return report, detection, probe, install

    def test_existing_cpu_without_gpu_is_idempotent(self):
        report, _, _, install = self.run_install([CPU])
        install.assert_not_called()
        self.assertEqual(report["selected_device"], "cpu")
        self.assertFalse(report["packages_changed"])

    def test_existing_cuda_is_preserved_even_without_nvidia_smi(self):
        report, _, _, install = self.run_install([CUDA])
        install.assert_not_called()
        self.assertEqual(report["selected_device"], "cuda")

    def test_existing_same_version_cpu_is_upgraded_to_cuda(self):
        report, _, _, install = self.run_install([CPU, CUDA], [MODERN_GPU])
        install.assert_called_once_with(PYTHON, "cu128")
        self.assertEqual(report["selected_device"], "cuda")
        self.assertTrue(report["packages_changed"])

    def test_bad_or_missing_existing_version_is_replaced(self):
        for existing in ({}, {**CPU, "version": "2.7.1+cpu"}, {"version": 28}):
            report, _, _, install = self.run_install([existing, CPU])
            install.assert_called_once_with(PYTHON, "cpu")
            self.assertEqual(report["selected_device"], "cpu")

    def test_requested_cpu_never_probes_or_selects_gpu(self):
        report, detection, probe, install = self.run_install(
            [{**CUDA, "cuda_usable": False}], [MODERN_GPU], "cpu")
        detection.assert_not_called()
        probe.assert_called_once_with(PYTHON, cuda=False)
        install.assert_not_called()
        self.assertEqual(report["selected_device"], "cpu")

    def test_cuda_kernel_failure_falls_back_and_validates_cpu(self):
        failed = {**CUDA, "cuda_usable": False, "cuda_error": "unsupported kernel"}
        report, _, probe, install = self.run_install([CPU, failed, CPU], [MODERN_GPU])
        self.assertEqual([call.args[1] for call in install.call_args_list], ["cu128", "cpu"])
        self.assertEqual(report["selected_device"], "cpu")
        self.assertEqual(report["fallback_reason"], "unsupported kernel")
        self.assertEqual(probe.call_args.kwargs, {"cuda": False})

    def test_cuda_download_failure_attempts_cpu(self):
        failure = subprocess.CalledProcessError(1, ["pip"])
        report, _, _, install = self.run_install(
            [CPU, CPU], [MODERN_GPU], install_error=[failure, None])
        self.assertEqual([call.args[1] for call in install.call_args_list], ["cu128", "cpu"])
        self.assertEqual(report["selected_device"], "cpu")
        self.assertIn("installation failed", report["fallback_reason"])

    def test_cpu_install_failure_is_reported(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_install([{}], install_error=subprocess.CalledProcessError(1, ["pip"]))

    def test_failed_cpu_validation_cannot_report_success(self):
        bad = {**CPU, "cpu_usable": False}
        with self.assertRaisesRegex(RuntimeError, "failed CPU validation"):
            self.run_install([{}, bad])

    def test_failed_fallback_cpu_validation_cannot_report_success(self):
        bad_cuda = {**CUDA, "cuda_usable": False}
        bad_cpu = {**CPU, "cpu_usable": False}
        with self.assertRaisesRegex(RuntimeError, "failed CPU validation"):
            self.run_install([CPU, bad_cuda, bad_cpu], [MODERN_GPU])

    def test_existing_cuda_out_of_memory_retains_gpu_build(self):
        busy = {**CUDA, "cuda_usable": False, "cuda_busy": True}
        report, _, _, install = self.run_install([busy], [MODERN_GPU])
        install.assert_not_called()
        self.assertEqual(report["selected_device"], "cpu")
        self.assertTrue(report["cuda_busy"])
        self.assertEqual(report["build"], CUDA["version"])

    def test_new_cuda_out_of_memory_retains_gpu_build(self):
        busy = {**CUDA, "cuda_usable": False, "cuda_busy": True}
        report, _, _, install = self.run_install([CPU, busy], [MODERN_GPU])
        install.assert_called_once_with(PYTHON, "cu128")
        self.assertTrue(report["cuda_busy"])

    def test_gpu_probe_requires_a_cuda_build(self):
        impossible = {**CPU, "cuda_usable": True}
        report, _, _, install = self.run_install([CPU, impossible, CPU], [MODERN_GPU])
        self.assertEqual(install.call_count, 2)
        self.assertEqual(report["selected_device"], "cpu")

    def test_exact_build_suffix_forces_cpu_cuda_replacement(self):
        with patch("torch_setup.subprocess.run") as run:
            setup.install_build(PYTHON, "cpu")
            setup.install_build(PYTHON, "cu128")
        cpu, cuda = (call.args[0] for call in run.call_args_list)
        self.assertIn("torch==2.8.0+cpu", cpu)
        self.assertIn("torch==2.8.0+cu128", cuda)
        self.assertEqual(cuda[-1], "https://download.pytorch.org/whl/cu128")


class ProbeTests(unittest.TestCase):
    def test_real_probe_distinguishes_memory_pressure_from_kernel_failure(self):
        class OutOfMemory(RuntimeError):
            pass

        cpu_value = SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 2))
        for exception, expected_busy in (
                (OutOfMemory("temporarily full"), True),
                (RuntimeError("unsupported CUDA kernel"), False)):
            def ones(*args, **kwargs):
                if kwargs.get("device") == "cuda":
                    raise exception
                return cpu_value

            fake = SimpleNamespace(
                __version__="2.8.0+cu128", version=SimpleNamespace(cuda="12.8"),
                bfloat16=object(), ones=ones,
                cuda=SimpleNamespace(is_available=lambda: True, OutOfMemoryError=OutOfMemory),
            )
            with self.subTest(exception=type(exception).__name__), \
                    patch.dict(sys.modules, {"torch": fake}), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                exec(setup.PROBE, {"PROBE_CUDA": True})
            report = json.loads(output.getvalue())
            self.assertTrue(report["cpu_usable"])
            self.assertFalse(report["cuda_usable"])
            self.assertEqual(report["cuda_busy"], expected_busy)

    def test_invalid_or_failed_probe_does_not_report_success(self):
        for result in (subprocess.CompletedProcess([], 1, "{}", ""),
                       subprocess.CompletedProcess([], 0, "not json", ""),
                       subprocess.CompletedProcess([], 0, "[]", "")):
            with patch("torch_setup.subprocess.run", return_value=result):
                self.assertIn("error", setup.probe_torch(PYTHON))

    def test_timed_out_probe_does_not_report_success(self):
        with patch("torch_setup.subprocess.run", side_effect=subprocess.TimeoutExpired("python", 30)):
            self.assertIn("error", setup.probe_torch(PYTHON))


if __name__ == "__main__":
    unittest.main()
