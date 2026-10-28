"""Real construction stages must preserve the canonical static query result."""
import asyncio
import importlib.util
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import AsyncMock, patch
import query_flow as q
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'sqlt.net'))
from server import graph, grouped_graph, object_graph, MemoryBridge

def event(i,entity):
 return {'event_id':i,'frame':'branch_connectivity','negated':False,
         'roles':[{'role':'entity','value':entity}], 'source':'PRIVATE SOURCE'}

class FlowTests(unittest.TestCase):
 def run_flow(self,fn):
  flow=q.Flow();flow.execute(fn)
  frames=list(flow.frames());self.assertEqual(frames[-1]['type'],'result')
  return frames
 def test_grouping_is_exactly_static_and_every_link_has_endpoints(self):
  rows=[event(1,'org | alpha | north | addr'),event(2,'org | beta | north | addr'),event(3,'org | alpha | north | addr')]
  expected=grouped_graph(rows,['company','region'])
  frames=self.run_flow(lambda:grouped_graph(rows,['company','region']))
  self.assertEqual(frames[-1]['data'],expected)
  seen=set();stages=[];links=set()
  for f in frames[:-1]:
   if f['type']!='progress':continue
   stages.append(f['stage'])
   for el in f['elements']:
    d=el['data']
    if not d.get('source'):seen.add(d['id'])
    else:
     self.assertIn(d['source'],seen);self.assertIn(d['target'],seen);links.add(d['id'])
    self.assertNotIn('PRIVATE SOURCE',str(el))
  self.assertIn('deduplicate',stages);self.assertIn('group_join',stages)
  self.assertEqual(links,{e['data']['id'] for e in expected['elements'] if e['data'].get('source')})
 def test_binding_and_object_joins_preserve_static_result(self):
  for fn in [lambda:graph([event(1,'entity')],'query'),
             lambda:object_graph([{'id':1,'experience_key':'x','case_key':'c','policy_key':'p'}],'experiences')]:
   expected=fn();frames=self.run_flow(fn)
   self.assertEqual(frames[-1]['data'],expected)
   self.assertTrue(any(f.get('elements') for f in frames[:-1]))
 def test_slow_reader_drops_progress_but_keeps_final_and_never_blocks(self):
  rows=[event(i,'org | city'+str(i)+' | north | addr') for i in range(100)]
  frames=self.run_flow(lambda:grouped_graph(rows,['company','region']))
  self.assertLessEqual(len(frames),33)
  self.assertGreater(frames[-1]['dropped_progress'],0)
  self.assertTrue(any(f.get('stage')=='group_join' for f in frames[:-1]),'Metadata must not starve graph mutations')
  self.assertEqual(frames[-1]['data'],grouped_graph(rows,['company','region']))
 def test_concurrent_requests_do_not_mix_graph_records(self):
  a=q.Flow();b=q.Flow()
  threads=[threading.Thread(target=f.execute,args=(lambda key=key:graph([event(key,str(key))]),)) for f,key in [(a,1),(b,2)]]
  for t in threads:t.start()
  for t in threads:t.join()
  fa=list(a.frames());fb=list(b.frames())
  self.assertNotIn('e-2',str(fa));self.assertNotIn('e-1',str(fb))
 def test_disconnected_observer_does_not_interrupt_computation(self):
  f=q.Flow();f.closed=True;f.execute(lambda:graph([event(1,'entity')]))
  self.assertIsNotNone(f.result);self.assertTrue(f.queue.empty())
 def test_only_errors_are_terminal_no_partial_result_claim(self):
  f=q.Flow();f.execute(lambda:1/0)
  self.assertEqual(list(f.frames())[-1]['type'],'error')

class AsyncFlowTests(unittest.IsolatedAsyncioTestCase):
 async def test_bridge_propagates_request_sink_across_async_boundary(self):
  bridge=MemoryBridge.__new__(MemoryBridge)
  f=q.Flow()
  async def op():q.progress('bind');return 7
  self.assertEqual(await bridge._with_flow(op(),f.emit),7)
  self.assertEqual(f.queue.get_nowait()['stage'],'bind')
  self.assertIsNone(q._sink.get())

if __name__=='__main__':unittest.main()

class StreamHTTPTests(unittest.TestCase):
 def test_progress_is_sent_before_execution_finishes(self):
  import inspect
  from http.server import ThreadingHTTPServer
  from http.client import HTTPConnection
  from unittest.mock import Mock
  from server import Handler
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
  server.daemon_threads=True
  server.bridge=Mock();server.access_verifier=Mock()
  server.access_verifier.authorized.return_value=False
  release=threading.Event()
  expected={'elements':[{'data':{'id':'one','kind':'entity','label':'Synthetic'}}],'events':1}
  def operation(*args):
   q.progress('bind',expected['elements'])
   release.wait(2)
   return expected
  server.bridge.browse.side_effect=operation
  worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
  host='cortex.internal' if 'cortex.internal' in inspect.getsource(Handler.do_GET) else '127.0.0.1:'+str(server.server_port)
  c=HTTPConnection(*server.server_address,timeout=3)
  try:
   c.request('GET','/api/graph?mode=initial&live=1',headers={'Host':host})
   response=c.getresponse()
   import json
   first=json.loads(response.readline())
   self.assertEqual(first['type'],'progress');self.assertEqual(first['stage'],'bind')
   self.assertFalse(release.is_set())
   release.set()
   rest=[json.loads(line) for line in response.read().splitlines()]
   self.assertEqual(rest[-1]['data'],expected)
  finally:release.set();c.close();server.shutdown();server.server_close();worker.join(3)
