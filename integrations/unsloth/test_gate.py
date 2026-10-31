"""Request packing regressions; no model, network, credentials or private facts."""
import copy
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('gate',Path(__file__).with_name('cortex_memory_gate.py'))
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

class GatePackingTests(unittest.IsolatedAsyncioTestCase):
    async def pack(self, ident, *, readonly=False, fail=False):
        data={'candidates':[{'event_id':ident,'roles':[{'role':'code','value':'00017'}],
                             'negated':True,'source':'Explicit supplied source','source_truncated':False,'time':'2025-01-01'}],
              'memory_protocol':'Long shared shell protocol',
              'coverage':{'complete_case_state':False},'live_graph_request_id':str(ident)}
        async def retrieve(q):
            if fail:raise RuntimeError('offline')
            return copy.deepcopy(data)
        payload=SimpleNamespace(messages=[SimpleNamespace(role='user',content='Previous reference'),
                         SimpleNamespace(role='assistant',content='Unverified old answer'),
                         SimpleNamespace(role='user',content='What is its code?')],mcp_enabled=True,
                         tools=['existing'],enable_tools=True)
        model=ModuleType('models.inference');model.ChatMessage=lambda **kw:SimpleNamespace(**kw)
        with patch.dict(sys.modules,{'models.inference':model}),patch.object(gate,'enable_exploration',return_value='Static tool instructions'):
            await gate.enrich(payload,config={'enabled':True,'namespace':'test','explore':not readonly,
                              'read_answer_only':readonly},retrieve=retrieve)
        return payload,data

    async def test_fixed_policy_precedes_changing_evidence(self):
        first,_=await self.pack(11);second,_=await self.pack(22)
        self.assertEqual(first.messages[0].content,second.messages[0].content)
        self.assertNotEqual(first.messages[1].content,second.messages[1].content)
        self.assertNotIn('request_id=',first.messages[0].content)
        self.assertIn('request_id=11',first.messages[1].content)

    async def test_evidence_values_and_coverage_preserved(self):
        import json
        payload,data=await self.pack(11)
        body=payload.messages[1].content.split('\n',1)[1].split('\nBefore the final answer',1)[0]
        actual=json.loads(body)
        self.assertEqual(actual['candidates'],data['candidates'])
        self.assertEqual(actual['coverage'],data['coverage'])
        self.assertEqual(payload.messages[-1].content,'What is its code?')
        self.assertNotIn('Unverified old answer',' '.join(m.content for m in payload.messages))

    async def test_readonly_keeps_evidence_without_tools(self):
        payload,_=await self.pack(11,readonly=True)
        self.assertEqual(len(payload.messages),3)
        self.assertIn('00017',payload.messages[1].content)
        self.assertFalse(payload.mcp_enabled)
        self.assertIsNone(payload.tools)

    async def test_retrieval_failure_does_not_generate_without_memory(self):
        with self.assertRaisesRegex(RuntimeError,'offline'):
            await self.pack(11,fail=True)

if __name__=='__main__':unittest.main()
