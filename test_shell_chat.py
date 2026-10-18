"""Terminal integration regressions without model calls or memory writes."""
import io
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
from clients.shell_chat import Presentation, run_shell, safe_text, is_shell, prompt
from clients.persistent_codex import PersistentCodex


class ShellChatTests(unittest.TestCase):
    def test_commands_route_locally_but_multiline_paste_stays_in_chat(self):
        root = Path('/tmp')
        for command in ('ls -la', 'pwd', 'df -h', 'cd /tmp', 'export CORTEX_TEST=value', 'unset CORTEX_TEST'):
            self.assertTrue(is_shell(command, root), command)
        for message in ('come funziona questa rete?', 'ls\nquesto è un testo incollato', '/chat ls'):
            self.assertFalse(is_shell(message, root), message)

    def test_real_launcher_runs_plain_commands_without_model(self):
        root = Path(__file__).resolve().parent
        result = subprocess.run([str(root / '.venv/bin/python'), '-m', 'clients.shell_chat'],
                                cwd=root, input='cd /tmp\npwd\nprintf CORTEX_SHELL_OK\nexit\n',
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('/tmp', result.stdout)
        self.assertIn('CORTEX_SHELL_OK', result.stdout)
        self.assertNotIn('Errore:', result.stdout)
        self.assertIn('@', result.stdout)

    def test_environment_and_compound_cd(self):
        import os
        root = Path('/tmp')
        with patch.dict(os.environ):
            run_shell('export CORTEX_SHELL_TEST=value', root)
            self.assertEqual(os.environ['CORTEX_SHELL_TEST'], 'value')
            run_shell('unset CORTEX_SHELL_TEST', root)
            self.assertNotIn('CORTEX_SHELL_TEST', os.environ)
        with patch('clients.shell_chat.subprocess.run') as execute:
            self.assertEqual(run_shell('cd / && pwd', root), root)
            execute.assert_called_once()

    def test_stream_is_not_printed_twice_and_strips_clipboard_escape(self):
        output = io.StringIO()
        with patch('sys.stdout', output):
            view = Presentation()
            view.event('item/agentMessage/delta', {'itemId': 'a', 'delta': 'Hello '})
            view.event('item/agentMessage/delta', {'itemId': 'a', 'delta': '**world**'})
            view.event('item/completed', {'item': {'id': 'a', 'type': 'agentMessage', 'text': 'Hello **world**'}})
            view.boundary()
        self.assertEqual(output.getvalue().count('Hello'), 1)
        self.assertEqual(safe_text('\x1b]52;c;clipboard\x07OK\x1b[31m'), 'OK')

    def test_local_cd_resolves_without_changing_parent_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a b').mkdir()
            self.assertEqual(run_shell('cd "a b"', root), root / 'a b')
            self.assertIsNone(run_shell('exit', root))

    def test_agent_threads_and_events_keep_memory_transport_compatible(self):
        client = PersistentCodex(Path('/tmp'), agent=True, instructions='Use MCP')
        client.process = object()
        calls = []
        def request(method, params, deadline):
            calls.append((method, params))
            return {'thread': {'id': 't'}} if method == 'thread/start' else {'turn': {'id': 'u'}}
        messages = iter([
            {'method': 'item/agentMessage/delta', 'params': {'threadId': 'other', 'delta': 'ignore'}},
            {'method': 'item/agentMessage/delta', 'params': {'threadId': 't', 'turnId': 'u', 'delta': 'Hi'}},
            {'method': 'item/completed', 'params': {'threadId': 't', 'item': {'id': 'a', 'type': 'agentMessage', 'text': 'Hi'}}},
            {'method': 'turn/completed', 'params': {'threadId': 't', 'turn': {'id': 'u', 'status': 'completed'}}},
        ])
        seen = []
        try:
            with patch.object(client, '_request', side_effect=request), patch.object(client, '_read', side_effect=lambda _: next(messages)):
                events, fresh, _ = client.turn('hello', on_event=lambda method, params: seen.append(method))
            self.assertTrue(fresh)
            self.assertEqual(calls[0][1]['sandbox'], 'danger-full-access')
            self.assertEqual(calls[0][1]['developerInstructions'], 'Use MCP')
            self.assertEqual(seen.count('item/agentMessage/delta'), 1)
            self.assertEqual(events[0]['item']['text'], 'Hi')
        finally:
            client.process = None


if __name__ == '__main__':
    unittest.main()
