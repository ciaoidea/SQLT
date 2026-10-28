"""Operational telemetry must expose lifecycle metadata, never tool payloads."""
import unittest
from unittest.mock import patch
import live_dynamics as live

class ExecutionTests(unittest.TestCase):
 def test_skill_lifecycle_redacts_arguments_and_output(self):
  item={'type':'commandExecution','id':'call-1','command':"/bin/bash -lc 'python3 /example/skills/robotic-arm/scripts/skill.py run status --secret PRIVATE'",'aggregatedOutput':'PRIVATE OUTPUT','status':'completed','exitCode':0}
  with patch.object(live,'watching',return_value=True),patch.object(live,'emit') as emit:
   live.trace_client_execution('main','a'*16,item,'complete')
   data=emit.call_args.args[1]
   self.assertEqual(data['category'],'skill');self.assertEqual(data['label'],'robotic-arm')
   self.assertEqual(data['status'],'completed');self.assertNotIn('PRIVATE',str(data))
 def test_reading_skill_is_generic_command_not_skill_execution(self):
  with patch.object(live,'watching',return_value=True),patch.object(live,'emit') as emit:
   live.trace_client_execution('main','a'*16,{'type':'commandExecution','id':'1','command':'cat /example/skills/example/SKILL.md'},'start')
   self.assertEqual(emit.call_args.args[1]['category'],'command')
 def test_connector_failure_and_memory_deduplication(self):
  with patch.object(live,'watching',return_value=True),patch.object(live,'emit') as emit:
   live.trace_client_execution('main','a'*16,{'type':'mcpToolCall','id':'1','server':'sample','tool':'read','arguments':{'secret':'PRIVATE'},'result':{'isError':True}},'complete')
   data=emit.call_args.args[1];self.assertEqual(data['status'],'failed');self.assertNotIn('PRIVATE',str(data))
   emit.reset_mock();live.trace_client_execution('main','a'*16,{'type':'mcpToolCall','id':'2','server':'cortex','tool':'query_memory'},'complete');emit.assert_not_called()
 def test_idle_no_serialization(self):
  with patch.object(live,'watching',return_value=False),patch.object(live.json,'dumps',side_effect=AssertionError('idle')),patch.object(live,'emit') as emit:
   live.trace_client_execution('main','a'*16,{'type':'commandExecution','id':'1','command':'true'},'start');emit.assert_not_called()

if __name__=='__main__':unittest.main()
