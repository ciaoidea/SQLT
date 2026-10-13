"""Installer integration without downloads, DB writes, or model loading."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
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
        self.assertEqual(command.call_args_list[-1].args[0],
                         [str(python), '-m', 'pip', 'check'])

    def test_dependency_conflict_stops_environment_setup(self):
        def run(args, **kwargs):
            if args[-2:] == ['pip', 'check']:
                raise subprocess.CalledProcessError(1, args)
        with patch('install.command', side_effect=run), patch('install.ensure_torch'):
            with self.assertRaises(subprocess.CalledProcessError):
                install.environment(True)

    def test_schema_only_does_not_install_or_probe_torch(self):
        with patch("install.command") as command, patch("install.ensure_torch") as ensure:
            _, report = install.environment(False)
        ensure.assert_not_called()
        self.assertIsNone(report)
        self.assertFalse(any("torch" in str(call.args[0]) for call in command.call_args_list))

    def test_launcher_forwards_arguments_and_refuses_unrelated_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'project with spaces'
            root.mkdir()
            entry=root/'sqlt'
            entry.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            entry.chmod(0o755)
            destination=Path(folder)/'bin'
            with patch('install.ROOT',root):
                with contextlib.redirect_stdout(io.StringIO()):
                    launcher=install.install_launcher(destination)
                result=subprocess.run([str(launcher),'--device','cpu'],capture_output=True,text=True,check=True)
                self.assertEqual(result.stdout.splitlines(),['--device','cpu'])
                launcher.write_text('unrelated command')
                with self.assertRaises(RuntimeError):install.install_launcher(destination)

    def test_launcher_path_is_available_to_login_and_interactive_shells(self):
        with tempfile.TemporaryDirectory(prefix='sqlt home ') as folder:
            home = Path(folder)
            destination = home / '.local/bin'
            destination.mkdir(parents=True)
            launcher = destination / 'sqlt'
            launcher.write_text('#!/bin/sh\nexit 0\n')
            launcher.chmod(0o755)
            (home / '.profile').write_text('# existing profile\n')
            (home / '.bashrc').write_text('# existing bash configuration\n')
            (home / '.bash_profile').write_text('# existing login configuration\n')
            install.configure_launcher_path(destination, home=home, shell='/bin/bash')
            install.configure_launcher_path(destination, home=home, shell='/bin/bash')
            env = dict(os.environ, HOME=str(home), PATH='/usr/bin:/bin')
            for name in ('.profile', '.bash_profile'):
                result = subprocess.run(
                    ['/bin/sh', '-c', '. "$1"; command -v sqlt', 'sh', str(home/name)],
                    env=env, capture_output=True, text=True, check=True)
                self.assertEqual(result.stdout.strip(), str(launcher))
            result = subprocess.run(
                ['bash', '--noprofile', '--rcfile', str(home/'.bashrc'),
                 '-ic', 'command -v sqlt'],
                env=env, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), str(launcher))
            for name in ('.profile', '.bash_profile', '.bashrc'):
                text = (home/name).read_text()
                self.assertTrue(text.startswith('# existing'))
                self.assertEqual(text.count('# >>> SQLT command PATH >>>'), 1)

    def test_full_install_registers_launcher_and_checks_features_before_download(self):
        events = []
        with tempfile.TemporaryDirectory() as folder, \
                patch('install.ROOT', Path(folder)), \
                patch('install.configure'), \
                patch('install.shutil.disk_usage', return_value=SimpleNamespace(free=40*1024**3)), \
                patch('install.environment', return_value=(Path(sys.executable), {})), \
                patch('install.stage', side_effect=lambda python, label, *args, **kwargs: events.append(label)), \
                patch('install.install_launcher', side_effect=lambda: events.append('launcher')), \
                patch('install.subprocess.run', return_value=subprocess.CompletedProcess(
                    [], 0, '{"server": {}, "model": null}')), \
                patch.object(sys, 'argv', ['install.py', '--existing-db']), \
                contextlib.redirect_stdout(io.StringIO()):
            install.main()
        self.assertLess(events.index('features-smoke'), events.index('download-progress'))
        self.assertLess(events.index('launcher'), events.index('download-progress'))
        self.assertEqual(events.count('launcher'), 1)

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
