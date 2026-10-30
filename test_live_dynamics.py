"""Synthetic telemetry checks: no database, private facts or model required."""
import inspect
import json
from pathlib import Path
import socket
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch
import live_dynamics as live

class LiveTests(unittest.TestCase):
    def setUp(self):
        self.namespace='live_test_'+uuid.uuid4().hex[:12]
        self.collector=live.LiveCollector(self.namespace,lease=.7,capacity=4)
    def tearDown(self):
        self.collector.close()
    def wait_events(self,n):
        end=time.monotonic()+1.5
        while time.monotonic()<end:
            result=self.collector.poll('test_client',0)
            if len(result['events'])>=n:return result
            time.sleep(.01)
        self.fail('Missing telemetry')
    def test_remote_context_boundary_and_final_focus_are_distinct_from_answer_support(self):
        self.collector.poll('test_client')
        live._active_cache.clear()
        request=live.notify_chat_turn(self.namespace,model='Example/Small',engine='Unsloth MCP')
        live.notify_context_focus(self.namespace,request,[{'event_id':22,'frame':'synthetic',
            'source':'PRIVATE SOURCE','roles':[{'role':'entity','value':'Synthetic'}]}])
        events=self.wait_events(3)['events']
        self.assertEqual([e['operation'] for e in events],['chat_turn','memory_selection','context_focus'])
        self.assertEqual(events[0]['engine'],'Unsloth MCP')
        self.assertEqual(events[-1]['request_id'],events[0]['call'])
        self.assertEqual(events[-1]['selected_ids'],[22])
        self.assertNotIn('PRIVATE SOURCE',json.dumps(events))

    def test_empty_context_explicitly_clears_final_selection(self):
        self.collector.poll('test_client');live._active_cache.clear()
        request=live.notify_chat_turn(self.namespace)
        live.notify_context_focus(self.namespace,request,[])
        self.assertEqual(self.wait_events(3)['events'][-1]['selected_ids'],[])

    def test_inactive_context_uses_no_transport(self):
        with patch.object(live,'watching',return_value=False),patch.object(live,'emit') as emit:
            self.assertIsNone(live.notify_chat_turn(self.namespace,engine='Unsloth MCP'))
            live.notify_context_focus(self.namespace,None,[])
            emit.assert_not_called()

    def test_idle_does_not_serialize_open_socket_or_transform_result(self):
        @live.observed
        def fake(namespace='main'):return {'event_id':7,'source':'do not transport'}
        with patch.object(live.socket,'socket',side_effect=AssertionError('idle socket')), patch.object(live.json,'dumps',side_effect=AssertionError('idle serialization')):
            self.assertEqual(fake(namespace=self.namespace)['event_id'],7)
        self.assertFalse(self.collector.path.exists())
        self.assertIsNone(self.collector.thread)
    def test_actual_calls_bounded_snapshot_without_source_or_query(self):
        self.collector.poll('test_client')
        live._active_cache.clear()
        @live.observed
        def query(query,namespace='main'):
            return {'events':[{'event_id':9,'frame':'synthetic','negated':True,'source':'SECRET SOURCE',
                               'roles':[{'role':'entity','value':'synthetic node'}]}],'total':1}
        signature=inspect.signature(query)
        self.assertIn('namespace',signature.parameters)
        query('SECRET QUERY',namespace=self.namespace)
        result=self.wait_events(2)
        self.assertEqual([e['phase'] for e in result['events']],['start','complete'])
        self.assertEqual(result['events'][-1]['records'][0]['id'],9)
        self.assertNotIn('SECRET',json.dumps(result))
        record=result['events'][-1]['records'][0]
        self.assertTrue(record['negated'])
        self.assertEqual(record['bindings'][0]['value'],'synthetic node')
    def test_only_selected_graph_roles_and_real_ids_are_transported(self):
        rows=[{'role':'password','value':'DO NOT SEND'}, {'role':'entity','value':'Sample City'},
              {'role':'source','value':'DO NOT SEND'}]
        values=live.graph_bindings(rows)
        self.assertEqual(len(values),1)
        self.assertEqual(values[0]['role'],'entity')
        import hashlib
        self.assertEqual(values[0]['id'],'v-'+hashlib.sha256(b'sample city').hexdigest()[:24])
        self.assertNotIn('DO NOT SEND',json.dumps(values))
    def test_prepared_context_reports_actual_selection_without_sources(self):
        self.collector.poll('test_client')
        live._active_cache.clear()
        live.notify_memory_selection(self.namespace,[{'event_id':22,'frame':'synthetic','source':'PRIVATE_SOURCE','roles':[{'role':'entity','value':'Synthetic'}]}])
        event=self.wait_events(1)['events'][0]
        self.assertEqual(event['operation'],'memory_selection')
        self.assertEqual(event['origin'],'gate')
        self.assertEqual(event['selected_ids'],[22])
        self.assertNotIn('PRIVATE_SOURCE',json.dumps(event))
    def test_stop_multiple_viewers_and_expiry(self):
        self.collector.poll('test_client')
        self.collector.poll('other_client')
        self.collector.poll('test_client',stop=True)
        self.assertTrue(self.collector.path.exists())
        self.collector.poll('other_client',stop=True)
        self.assertFalse(self.collector.path.exists())
        self.collector.poll('test_client')
        time.sleep(1.3)
        self.assertFalse(self.collector.path.exists())
        self.assertEqual(len(self.collector.events),0)
    def test_overflow_and_exception_do_not_change_call_semantics(self):
        self.collector.poll('test_client')
        live._active_cache.clear()
        @live.observed
        def fail(namespace='main'):raise RuntimeError('PRIVATE ERROR')
        for _ in range(4):
            with self.assertRaisesRegex(RuntimeError,'PRIVATE'):fail(namespace=self.namespace)
        result=self.wait_events(4)
        time.sleep(.1)
        result=self.collector.poll('test_client')
        self.assertLessEqual(len(result['events']),4)
        self.assertGreater(result['overflow'],0)
        self.assertNotIn('PRIVATE',json.dumps(result))
    def test_namespace_and_installation_are_isolated(self):
        self.assertNotEqual(live.socket_path('main'),live.socket_path('different'))
        self.assertNotEqual(live.socket_path('main',Path('/a')),live.socket_path('main',Path('/b')))
        with self.assertRaises(ValueError):self.collector.poll('../bad')
    def test_packet_validation_and_rate_bound(self):
        self.collector.poll('test_client')
        live.emit(self.namespace,{'phase':'complete','operation':'test','records':[],'count':0,'source':'DROP ME'})
        r=self.wait_events(1)
        self.assertNotIn('DROP ME',json.dumps(r))
        with self.assertRaises(ValueError):self.collector.poll('bad')
    def test_stale_socket_recovers_without_deleting_live_listener(self):
        path=self.collector.path;path.parent.mkdir(exist_ok=True,mode=0o700)
        with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as dead:dead.bind(str(path))
        self.collector.poll('test_client')
        other=live.LiveCollector(self.namespace)
        with self.assertRaises(ValueError):other.poll('different_client')
        # The failed second collector does not own/unlink the first socket.
        other.close()
        self.assertTrue(path.exists())

if __name__=='__main__':unittest.main()
