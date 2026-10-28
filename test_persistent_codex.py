"""Native history recovery; no model calls, credentials or database writes."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from clients.persistent_codex import PersistentCodex, CodexRPCError


class WriterRecoveryTests(unittest.TestCase):
    def test_inactivity_stops_before_total_budget(self):
        client = PersistentCodex(Path('/tmp'))
        client.last_transport_event = 1
        with patch('clients.persistent_codex.time.perf_counter',return_value=302):
            with self.assertRaisesRegex(TimeoutError,'Nessun evento'):
                client._read(1800)

    def test_protocol_activity_refreshes_idle_watchdog(self):
        client = PersistentCodex(Path('/tmp'))
        client.last_transport_event = 1
        client.messages.put({'method':'progress','params':{}})
        with patch('clients.persistent_codex.time.perf_counter',return_value=250):
            client._read(1800)
        self.assertEqual(client.last_transport_event,250)

    def test_conflicting_resume_forks_history_before_any_turn(self):
        client = PersistentCodex(Path('/tmp'))
        settings = {'threadId': 'saved-thread', 'model': 'gpt-6.1-sol', 'cwd': '/tmp'}
        conflict = CodexRPCError({'code': -32600, 'message': 'thread saved-thread already has an active writer'})
        with patch.object(client, '_request', side_effect=[conflict, {'thread': {'id': 'fork-thread'}}]) as request:
            result = client._resume_thread(settings, 123)
        self.assertEqual(result['thread']['id'], 'fork-thread')
        self.assertEqual([c.args[0] for c in request.call_args_list], ['thread/resume', 'thread/fork'])
        self.assertEqual(request.call_args_list[1].args[1]['threadId'], 'saved-thread')
        self.assertFalse(request.call_args_list[1].args[1]['ephemeral'])
        self.assertEqual(client.forked_from_thread_id, 'saved-thread')

    def test_unrelated_errors_never_fork_or_retry(self):
        for message in ('authentication required', 'thread other-thread already has an active writer'):
            client = PersistentCodex(Path('/tmp'))
            failure = CodexRPCError({'code': -32600, 'message': message})
            with patch.object(client, '_request', side_effect=failure) as request:
                with self.assertRaises(CodexRPCError):
                    client._resume_thread({'threadId': 'saved-thread'}, 123)
            request.assert_called_once()

    def test_normal_resume_reuses_original_thread(self):
        client = PersistentCodex(Path('/tmp'))
        expected = {'thread': {'id': 'saved-thread'}}
        with patch.object(client, '_request', return_value=expected) as request:
            self.assertEqual(client._resume_thread({'threadId': 'saved-thread'}, 123), expected)
        request.assert_called_once()
        self.assertIsNone(client.forked_from_thread_id)

    def test_saved_reference_points_to_recovered_thread_without_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            client = PersistentCodex(Path('/tmp'), session_dir=Path(directory))
            client._save_thread('fork-thread')
            self.assertEqual(client._saved_thread(), 'fork-thread')
            self.assertEqual(client._session_path().stat().st_mode & 0o777, 0o600)


if __name__ == '__main__':
    unittest.main()
