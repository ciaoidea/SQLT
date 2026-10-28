"""Bounded imports and context configuration; no operational memory access."""
import base64
from contextlib import contextmanager
from pathlib import Path
import unittest
from unittest.mock import patch
import semantic_memory
import mcp_server
from clients.persistent_codex import PersistentCodex

class ContinuousMemoryTests(unittest.TestCase):
    def test_long_source_encoded_in_bounded_requests_without_loss(self):
        text='A documented observation. '*1000
        calls=[]
        def request(operation,texts):
            calls.append(texts)
            return {'vectors':[[base64.b64encode(bytes(384*4)).decode()] for _ in texts]}
        with patch('encoder_runtime.request',side_effect=request):
            result=semantic_memory.embed_passages([text])
        self.assertEqual([p for p,v in result[0]],semantic_memory.chunks(text))
        self.assertTrue(all(len(c)<=32 for c in calls))
        self.assertGreater(len(calls),1)

    def test_large_import_partial_failure_reports_committed_offset(self):
        batches=[]
        class Memory:
            def remember_many(self,events):
                batches.append(events)
                if len(batches)==2:raise RuntimeError('failure')
                return dict(event_ids=list(range(len(events))),write_ms=1,semantic_encoding_seconds=0,semantic_chunks=len(events))
        @contextmanager
        def memory(namespace):yield Memory()
        events=[mcp_server.MemoryEvent(frame='test',roles=[dict(role='entity',value=str(i))],source='source') for i in range(135)]
        with patch.object(mcp_server,'memory',memory):
            result=mcp_server.remember_events(events)
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['next_offset'],64)
        self.assertEqual([len(b) for b in batches],[64,64])

    def test_native_compaction_is_not_overridden(self):
        client=PersistentCodex(Path('/tmp'))
        class Process:
            stdout=[]
        with patch('clients.persistent_codex.subprocess.Popen',return_value=Process()) as popen,patch.object(client,'_request'),patch.object(client,'_send'):
            client._start(0)
        self.assertFalse(any('model_auto_compact_token_limit' in value for value in popen.call_args.args[0]))
        client.process=None

if __name__=='__main__':unittest.main()
