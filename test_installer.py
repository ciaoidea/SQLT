"""Installer integration without downloads, DB writes, or model loading."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import install


class InstallerTests(unittest.TestCase):
    def test_full_install_validates_torch_with_requested_device(self):
        with patch("install.command") as command, patch("install.ensure_torch", return_value={"build": "tested"}) as ensure:
            python, report = install.environment(True, "auto")
        ensure.assert_called_once_with(install.ROOT / ".venv/bin/python", "auto")
        self.assertEqual(report, {"build": "tested"})
        self.assertTrue(any("requirements.txt" in str(call.args[0][-1]) for call in command.call_args_list))

    def test_schema_only_does_not_install_or_probe_torch(self):
        with patch("install.command") as command, patch("install.ensure_torch") as ensure:
            _, report = install.environment(False)
        ensure.assert_not_called()
        self.assertIsNone(report)
        self.assertFalse(any("torch" in str(call.args[0]) for call in command.call_args_list))

    def test_check_never_invokes_package_installation(self):
        with patch("install.nvidia_devices", return_value=[]), \
                patch("install.probe_torch", return_value={"version": "tested"}), \
                patch("install.compose_command", return_value=["docker", "compose"]), \
                patch("install.subprocess.run"), patch("install.command") as command, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            install.inspect()
        command.assert_not_called()
        self.assertIn('"candidate_build": "cpu"', output.getvalue())

    def test_gpu_probe_helper_requires_only_standard_library(self):
        # The schema-only bootstrap must import successfully before either
        # heavyweight dependency has been installed.
        result = subprocess.run(
            [sys.executable, "-c",
             'import sys; sys.modules["torch"] = None; '
             'sys.modules["transformers"] = None; import install'],
            cwd=Path(__file__).parent, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
