"""Grouping and planner boundaries, using synthetic records only."""
import json
import subprocess
import unittest
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch
from server import grouped_graph, Handler
from planner import plan


def event(ident, label, team=None):
    roles=[{'role':'entity','value':label}]
    if team: roles.append({'role':'team','value':team})
    return {'event_id':ident,'frame':'branch_connectivity','roles':roles,'source':'Synthetic source','negated':False}


class ViewerTests(unittest.TestCase):
    def test_grouping_preserves_branch_identity_and_deduplicates_repeated_events(self):
        rows=[event(1,'org | alpha | north | address 1'),event(2,'org | alpha | north | address 1'),
              event(3,'org | alpha | north | address 2'),event(4,'org | beta | south | address 3')]
        result=grouped_graph(rows,'region')
        self.assertEqual(result['events'],4)
        self.assertEqual(result['entities'],3)
        self.assertEqual(result['groups'],2)
        self.assertEqual(sorted(e['data']['members_count'] for e in result['elements'] if e['data'].get('kind')=='group'),[1,2])
    def test_arbitrary_group_role_and_missing_group_are_not_inferred(self):
        result=grouped_graph([event(1,'first','security'),event(2,'second')],'team')
        groups={e['data']['group_value'] for e in result['elements'] if e['data'].get('kind')=='group'}
        self.assertEqual(groups,{'security','Not specified'})
    def test_hierarchy_keeps_same_region_separate_between_companies(self):
        rows=[event(1,'org-a | alpha | north | address 1'),event(2,'org-b | beta | north | address 2'),
              event(3,'org-a | alpha | north | address 1')]
        result=grouped_graph(rows,['company','region'])
        groups=[e['data'] for e in result['elements'] if e['data'].get('kind')=='group']
        self.assertEqual(result['entities'],2)
        self.assertEqual(len(groups),4)
        self.assertEqual(len([g for g in groups if g['group_level']==0]),2)
        self.assertEqual(len({g['id'] for g in groups if g['group_value']=='north'}),2)
        self.assertEqual(result['group_levels'],['company','region'])
        self.assertEqual(len([e for e in result['elements'] if e['data'].get('id','').startswith('hierarchy-')]),2)

    def test_planner_rejects_role_from_wrong_frame(self):
        answer={'title':'test','mode':'relational','queries':[{'frame':'branch_connectivity',
          'filters':[{'role':'technology','operator':'eq','value':'ftth'}]}],
          'semantic_query':'','group_by':[],'clarification':''}
        output=json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(answer)}})
        catalog={'frames':['branch_connectivity'],'roles':['technology','line_type'],
                 'frame_roles':{'branch_connectivity':['line_type']}}
        with patch('planner.subprocess.run',return_value=subprocess.CompletedProcess([],0,output,'')):
            with self.assertRaisesRegex(ValueError,'selected frame'):plan('synthetic request',catalog)


class HTTPBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.http.bridge = SimpleNamespace(namespace='main',browse=lambda mode, query: {
            'elements': [], 'events': 0, 'test_mode': mode, 'test_query': query})
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.http.server_port}'

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(2)

    def test_assets_are_served_without_persisting_memory(self):
        for path in ('/', '/map.js', '/map.css', '/vendor/cytoscape.min.js'):
            with urllib.request.urlopen(self.base + path) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
                self.assertIn("default-src 'self'", response.headers['Content-Security-Policy'])
        with urllib.request.urlopen(self.base + '/api/graph?mode=natural&q=synthetic') as response:
            result = json.load(response)
            self.assertEqual(result['test_mode'], 'natural')
            self.assertEqual(result['test_query'], 'synthetic')

    def test_private_files_and_untrusted_hosts_are_blocked(self):
        for path in ('/.env.json', '/server.py', '/../.env.json'):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(self.base + path)
            self.assertEqual(caught.exception.code, 404)
        request = urllib.request.Request(self.base + '/', headers={'Host': 'untrusted.example'})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 403)

    def test_unsupported_mode_is_rejected_before_mcp(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(self.base + '/api/graph?mode=raw_sql')
        self.assertEqual(caught.exception.code, 400)




class WorkbenchTests(unittest.TestCase):
 def test_apfc_graph_records_explicit_links_only(self):
  from server import object_graph
  r=object_graph([dict(id=1,experience_key='attempt',case_key='case',policy_key='p',action='retry',status='completed')],'experiences')
  self.assertEqual(len([e for e in r['elements'] if e['data'].get('source')]),2)
  self.assertTrue(all(e['data'].get('kind')!='similarity' for e in r['elements']))
 def test_apfc_plan_uses_its_own_fields(self):
  answer={'dataset':'experiences','title':'Completed experiences','mode':'relational','queries':[{'frame':None,'filters':[dict(role='status',operator='eq',value='completed')]}],'semantic_query':'','group_by':['case_key'],'clarification':''}
  output=json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(answer)}})
  with patch('planner.subprocess.run',return_value=subprocess.CompletedProcess([],0,output,'')):
   result,_=plan('Show completed experiences',{'frames':[],'roles':[]})
  self.assertEqual(result['dataset'],'experiences')
 def test_apfc_plan_does_not_claim_semantic_ranking_of_models(self):
  answer={'dataset':'policies','title':'Policies','mode':'semantic','queries':[],'semantic_query':'similar policies','group_by':[],'clarification':''}
  output=json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(answer)}})
  with patch('planner.subprocess.run',return_value=subprocess.CompletedProcess([],0,output,'')):
   with self.assertRaises(ValueError):plan('Similar policies',{'frames':[],'roles':[]})



class PlannerResponsivenessTests(unittest.IsolatedAsyncioTestCase):
 async def test_table_browsing_does_not_wait_for_language_generation(self):
  import asyncio,threading
  from unittest.mock import AsyncMock
  from server import MemoryBridge
  bridge=MemoryBridge.__new__(MemoryBridge);bridge.lock=asyncio.Lock();bridge.planning_lock=asyncio.Semaphore(1)
  from planner import PlanCache
  bridge.plan_cache=PlanCache()
  bridge._catalog=AsyncMock(return_value={});bridge._natural=AsyncMock(return_value={'done':True})
  bridge._table_unlocked=AsyncMock(return_value={'records':[]});started=threading.Event();release=threading.Event()
  def slow_plan(*args):started.set();release.wait(3);return {},0
  with patch('server.plan',side_effect=slow_plan):
   task=asyncio.create_task(bridge._browse('natural','question'))
   try:
    self.assertTrue(await asyncio.to_thread(started.wait,1))
    rows=await asyncio.wait_for(bridge._workbench('browse',{'dataset':'cases'}),timeout=.5)
    self.assertEqual(rows,{'records':[]})
   finally:release.set();await task

if __name__=='__main__':unittest.main()


class LiveHTTPTests(unittest.TestCase):
 def test_live_endpoint_requires_existing_host_boundary_and_valid_cursor(self):
  import threading
  from http.server import ThreadingHTTPServer
  from http.client import HTTPConnection
  from unittest.mock import Mock
  from server import Handler
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
  server.live=Mock();server.live.poll.return_value={'events':[],'cursor':0}
  server.access_verifier=Mock();server.access_verifier.authorized.return_value=False
  thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  import inspect
  host='cortex.internal' if 'cortex.internal' in inspect.getsource(Handler.do_GET) else '127.0.0.1:'+str(server.server_port)
  try:
   for target,headers,wanted in [('/api/live?client=test_viewer',{'Host':'evil.example'},403),
                                 ('/api/live?client=test_viewer&cursor=-1',{'Host':host},400),
                                 ('/api/live?client=test_viewer',{'Host':host},200)]:
    c=HTTPConnection(*server.server_address,timeout=2);c.request('GET',target,headers=headers)
    response=c.getresponse();self.assertEqual(response.status,wanted);response.read();c.close()
   self.assertEqual(server.live.poll.call_count,1)
  finally:server.shutdown();server.server_close();thread.join(2)


class CatalogCoverageTests(unittest.IsolatedAsyncioTestCase):
 async def test_union_of_roles_does_not_imply_every_record_has_every_role(self):
  from server import MemoryBridge
  bridge=object.__new__(MemoryBridge)
  async def call(name,args):
   return {'events':[event(1,'org | alpha | north | address'),
                     dict(event(2,'org | alpha | north | address'),roles=[{'role':'entity','value':'org | alpha'},{'role':'company','value':'org'}])],
           'has_more':False,'next_offset':None}
  bridge._call=call
  catalog=await bridge._catalog()
  self.assertEqual(catalog['field_coverage']['branch_connectivity']['entity'],1.)
  self.assertEqual(catalog['field_coverage']['branch_connectivity']['company'],.5)
  self.assertTrue(catalog['catalog_complete'])
