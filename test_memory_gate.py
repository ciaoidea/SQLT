import unittest
from pathlib import Path
from unittest.mock import patch
from clients.persistent_codex import PersistentCodex
class GateTests(unittest.TestCase):
 def test_failure_prevents_any_model_request(self):
  c=PersistentCodex(Path('/tmp'),memory_root=Path('/tmp'))
  with patch('clients.memory_gate.retrieve',side_effect=RuntimeError('offline')),patch.object(c,'_start') as start:
   with self.assertRaisesRegex(RuntimeError,'offline'):c.turn('question')
   start.assert_not_called()
 def test_read_happens_before_model_start(self):
  c=PersistentCodex(Path('/tmp'),memory_root=Path('/tmp'),namespace='example')
  with patch('clients.memory_gate.retrieve',return_value={'candidates':[]}) as recall,patch.object(c,'_start',side_effect=RuntimeError('stop')):
   with self.assertRaisesRegex(RuntimeError,'stop'):c.turn('question')
   recall.assert_called_once_with(Path('/tmp'),'example','question')

class CaseEvidenceTests(unittest.IsolatedAsyncioTestCase):
 async def test_completed_action_outside_semantic_results_is_retrieved(self):
  from clients.memory_gate import assemble
  seed=dict(event_id=1,frame='case_context',negated=False,
            roles=[dict(role='case_id',value='case-742')],source='Question: 103 items in the email.')
  sent=dict(seed,event_id=20,source='The request was already sent. The filtered table contains 101 items.')
  calls=[]
  async def call(name,args):
   calls.append((name,args))
   if name=='semantic_recall':return {'events':[seed]}
   return dict(events=[seed,sent],total=2,has_more=False,next_offset=None)
  result=await assemble(call,'main','What remains to do?')
  self.assertEqual([x['event_id'] for x in result['candidates']],[1,20])
  self.assertEqual(calls[1][1]['filters'][0],dict(role='case_id',operator='eq',value='case-742'))
  self.assertIn('103',result['candidates'][0]['source'])
  self.assertIn('101',result['candidates'][1]['source'])
  self.assertFalse(result['coverage']['complete_case_state'])

 async def test_pagination_is_bounded_and_incompleteness_explicit(self):
  from clients.memory_gate import assemble,MAX_CONTEXT_CHARS
  event=dict(event_id=1,frame='case_context',negated=True,
             roles=[dict(role='entity',value='Example')],source='x'*5000)
  calls=[]
  async def call(name,args):
   calls.append(args)
   if name=='semantic_recall':return {'events':[event]}
   if name=='apfc_context':return {'cases':[],'has_more':False}
   return dict(events=[dict(event,event_id=i+args['offset']) for i in range(25)],total=1000,has_more=True,next_offset=args['offset']+25)
  result=await assemble(call,'main','Question')
  self.assertEqual(len(calls),4)
  self.assertTrue(result['linked_searches'][-1]['has_more'])
  self.assertGreater(result['coverage']['omitted_events'],0)
  import json
  self.assertLess(sum(len(json.dumps(x,ensure_ascii=False)) for x in result['candidates']),MAX_CONTEXT_CHARS+1)
  self.assertTrue(result['candidates'][0]['negated'])

 async def test_link_failure_is_not_silently_ignored(self):
  from clients.memory_gate import assemble
  async def call(name,args):
   if name=='semantic_recall':return {'events':[dict(event_id=1,roles=[dict(role='case_id',value='case')])]}
   raise RuntimeError('lookup unavailable')
  with self.assertRaisesRegex(RuntimeError,'unavailable'):await assemble(call,'main','Question')


class OperationalGateTests(unittest.IsolatedAsyncioTestCase):
 async def test_operational_state_is_loaded_automatically(self):
  from clients.memory_gate import assemble
  event=dict(event_id=1,frame='case_context',roles=[],negated=False,source='Observed completed action')
  state=dict(case_key='case',revision=3,state={'completed':[{'text':'Already sent','evidence_event_ids':[1]}]})
  async def call(name,args):
   if name=='semantic_recall':return {'events':[event]}
   if name=='apfc_context':return {'cases':[state],'has_more':False}
   raise AssertionError(name)
  result=await assemble(call,'main','What remains?')
  self.assertEqual(result['operational_states'],[state])
 async def test_large_operational_state_does_not_expand_context_unboundedly(self):
  from clients.memory_gate import assemble,MAX_CONTEXT_CHARS
  import json
  event=dict(event_id=1,frame='case_context',roles=[],negated=False,source='source')
  async def call(name,args):
   if name=='semantic_recall':return {'events':[event]}
   return {'cases':[{'state':'x'*30000}],'has_more':True}
  result=await assemble(call,'main','Question')
  self.assertEqual(result['operational_states'],[])
  self.assertTrue(result['operational_state_coverage']['linked_cases_truncated'])
  self.assertEqual(result['operational_state_coverage']['omitted_for_budget'],1)

class HttpGateIsolationTests(unittest.IsolatedAsyncioTestCase):
 async def test_other_installation_is_rejected_before_using_evidence(self):
  from clients.memory_gate import _session_context
  from types import SimpleNamespace
  import tempfile
  class Session:
   async def call_tool(self,name,args):
    return SimpleNamespace(isError=False,structuredContent={'candidates':[], 'memory_origin':{'installation':'other','namespace':'main'}},content=[])
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(RuntimeError,'origin mismatch'):
    await _session_context(Session(),Path(d),'main','question',True)
 async def test_owner_only_http_configuration_is_required(self):
  from clients.memory_gate import recall
  import tempfile,json
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);token=root/'token';token.write_text('x'*48);token.chmod(0o600)
   config=root/'.recall-http.json';config.write_text(json.dumps({'url':'http://127.0.0.1:1/mcp','token_file':str(token)}));config.chmod(0o644)
   with self.assertRaisesRegex(ValueError,'owner-only'):await recall(root,'main','question')
