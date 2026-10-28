"""Real stdio MCP round trip including evidence ingestion and automatic gate."""
import json
from pathlib import Path
import sys
import unittest
import uuid
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from episodic_memory import EpisodicMemory
from clients.memory_gate import assemble

class MCPRoundTrip(unittest.IsolatedAsyncioTestCase):
 async def test_ingest_state_learn_and_reload_through_mcp(self):
  root=Path(__file__).resolve().parent;namespace='check_'+uuid.uuid4().hex[:12]
  cleanup=EpisodicMemory(namespace)
  try:
   async with stdio_client(StdioServerParameters(command=sys.executable,args=[str(root/'mcp_server.py')])) as streams:
    async with ClientSession(*streams) as session:
     await session.initialize()
     names={t.name for t in (await session.list_tools()).tools}
     self.assertIn('apfc_observe',names)
     async def call(name,args):
      response=await session.call_tool(name,{'namespace':namespace,**args})
      if response.isError:raise AssertionError(str(response.content))
      return response.structuredContent or json.loads(response.content[0].text)
     before=await call('remember_event',dict(frame='check_context',roles=[dict(role='entity',value='test project')],source='Synthetic test observation: the request has been sent and a reply is awaited.'))
     outcome=await call('remember_event',dict(frame='check_outcome',roles=[dict(role='entity',value='test project')],source='Synthetic test observed outcome: the reply arrived and the request was resolved.'))
     b=before['event_id'];o=outcome['event_id']
     await call('apfc_update_case',dict(case_key='case',title='Test project',patch={'completed':[dict(text='Request sent',evidence_event_ids=[b])],'confirmed':[dict(text='Reply received',evidence_event_ids=[o])]},expected_revision=0,request_id='first'))
     context=await assemble(call,namespace,'What remains to do for the test project?')
     self.assertEqual(context['operational_states'][0]['revision'],1)
     await call('apfc_register_policy',dict(policy_key='test',features=['x_unitless'],actions=['send']))
     await call('apfc_observe',dict(policy_key='test',action='send',features={'x_unitless':0.2},reward=1,before_event_id=b,outcome_event_id=o,request_id='outcome'))
     prediction=await call('apfc_predict',dict(policy_key='test',features={'x_unitless':0.2}))
     self.assertEqual(prediction['actions'][0]['examples'],1)
     self.assertFalse(prediction['automatic_execution'])
   # A separate server process must recover persisted state and numerical weights.
   async with stdio_client(StdioServerParameters(command=sys.executable,args=[str(root/'mcp_server.py')])) as streams:
    async with ClientSession(*streams) as session:
     await session.initialize()
     response=await session.call_tool('apfc_case',dict(namespace=namespace,case_key='case'))
     self.assertFalse(response.isError)
     result=response.structuredContent or json.loads(response.content[0].text)
     self.assertEqual(result['state']['completed'][0]['text'],'Request sent')
     response=await session.call_tool('apfc_predict',dict(namespace=namespace,policy_key='test',features={'x_unitless':.2}))
     result=response.structuredContent or json.loads(response.content[0].text)
     self.assertAlmostEqual(result['actions'][0]['estimated_reward'],prediction['actions'][0]['estimated_reward'])
  finally:cleanup.delete_namespace();cleanup.close()

if __name__=='__main__':unittest.main()
