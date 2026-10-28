"""Real MCP lifecycle across server processes, no language-model dependency."""
import json
from pathlib import Path
import sys
import unittest
import uuid
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from episodic_memory import EpisodicMemory
from apfc import APFC

class LifecycleMCPTests(unittest.IsolatedAsyncioTestCase):
 async def test_pending_then_atomic_outcome_in_new_server(self):
  root=Path(__file__).resolve().parent;ns='check_'+uuid.uuid4().hex[:12];cleanup=EpisodicMemory(ns)
  async def call(session,name,**args):
   r=await session.call_tool(name,dict(namespace=ns,**args))
   if r.isError:raise AssertionError(str(r.content))
   return r.structuredContent or json.loads(r.content[0].text)
  try:
   async with stdio_client(StdioServerParameters(command=sys.executable,args=[str(root/'mcp_server.py')])) as streams:
    async with ClientSession(*streams) as session:
     await session.initialize()
     b=await call(session,'remember_event',frame='check_intervention',roles=[dict(role='entity',value='test router')],source='Synthetic observed intervention: Ethernet is connected and mobile data was reconnected.')
     started=await call(session,'apfc_start_experience',case_key='router',title='Router',experience_key='attempt',policy_key='mobile-v1',action='mobile_reconnect',features={'ethernet_connected_0_1':1},before_event_id=b['event_id'],outcome_definition='Internet is working by the next check: 1, otherwise observed failure: 0.',expected_revision=0)
     self.assertEqual(started['status'],'pending')
   async with stdio_client(StdioServerParameters(command=sys.executable,args=[str(root/'mcp_server.py')])) as streams:
    async with ClientSession(*streams) as session:
     await session.initialize()
     case=await call(session,'apfc_case',case_key='router')
     self.assertEqual(case['experience_summary']['experiences'][0]['status'],'pending')
     outcome=await call(session,'remember_event',frame='check_outcome',roles=[dict(role='entity',value='test router')],source='Synthetic measured outcome: Internet works at the next check.')
     args=dict(case_key='router',experience_key='attempt',outcome_event_id=outcome['event_id'],observed_quote='Internet works at the next check.',reward=1,evidence_kind='tool_observation',expected_revision=1)
     done=await call(session,'apfc_complete_experience',**args)
     self.assertTrue(done['learning_updated'])
     duplicate=await call(session,'apfc_complete_experience',**args)
     self.assertFalse(duplicate['learning_updated'])
     status=await call(session,'apfc_status')
     self.assertEqual(status['learned_observations'],1)
     self.assertEqual(status['experiences'],{'completed':1})
   self.assertEqual(APFC(cleanup).get_case('router')['state']['waiting'],[])
  finally:cleanup.delete_namespace();cleanup.close()

if __name__=='__main__':unittest.main()
