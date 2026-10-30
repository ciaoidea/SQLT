"""Portable regression checks for model-independent retrieval and the Unsloth gate."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

gate=load('cortex_unsloth_gate',ROOT/'integrations/unsloth/cortex_memory_gate.py')
installer=load('cortex_unsloth_install',ROOT/'integrations/unsloth/install.py')

def event(i,entity='Example',source='original observation',negated=False):
    return dict(event_id=i,frame='fact',roles=[{'role':'entity','value':entity}],source=source,negated=negated,time='',place='')

class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_expansion_occurs_without_model_tool_choice(self):
        from memory_context import assemble_context
        calls=[]
        async def call(name,args):
            calls.append((name,args))
            if name=='semantic_recall':return {'events':[event(99,'Unrelated')]}
            return {'events':[event(2,source='verified prior source')],'total':30,'has_more':True}
        r=await assemble_context(call,'isolated','arbitrary question',{'events':[event(1)],'terms':[]})
        self.assertEqual([n for n,_ in calls],['semantic_recall','query_memory'])
        self.assertEqual(calls[1][1]['namespace'],'isolated')
        self.assertEqual(calls[1][1]['filters'],[{'role':'entity','operator':'eq','value':'Example'}])
        self.assertNotIn(99,[e['event_id'] for e in r['candidates']])
        self.assertTrue(r['linked_searches'][0]['has_more'])
        self.assertFalse(r['coverage']['complete'])
    async def test_negative_statements_preserved_and_context_bounded(self):
        from memory_context import assemble_context
        async def call(name,args):
            if name=='semantic_recall':return {'events':[event(1,source='x'*30000,negated=True)]}
            return {'events':[],'total':0,'has_more':False}
        r=await assemble_context(call,'main','q',{'events':[]})
        self.assertTrue(r['candidates'][0]['negated'])
        self.assertTrue(r['candidates'][0]['source_truncated'])
        self.assertLess(len(json.dumps(r)),12000)
    async def test_query_failure_is_not_presented_as_no_memory(self):
        from memory_context import assemble_context
        async def fail(*args):raise ConnectionError('offline')
        with self.assertRaises(ConnectionError):await assemble_context(fail,'main','q',{'events':[]})

class LexicalTests(unittest.TestCase):
    def test_long_questions_use_query_budget_not_label_budget(self):
        from memory_context import lexical_candidates
        class Cursor:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def execute(self,sql,params):
                self_case.assertIn('e.namespace=%s',sql)
                self_case.assertIn('isolated',params)
            def fetchall(self):return []
        self_case=self
        memory=SimpleNamespace(namespace='isolated',c=SimpleNamespace(cursor=lambda:Cursor()))
        r=lexical_candidates(memory,'A long question with numerical identifiers 00742. '*50)
        self.assertEqual(r['events'],[])
        self.assertFalse(r['complete'])

class GateTests(unittest.IsolatedAsyncioTestCase):
    async def test_gate_disabled_respects_chat_setting(self):
        async def fail(q):raise AssertionError('must not retrieve')
        self.assertFalse(await gate.enrich(SimpleNamespace(mcp_enabled=False),config={'enabled':True},retrieve=fail))
    async def test_memory_is_available_before_model_and_does_not_accumulate(self):
        class Message:
            def __init__(self,role,content):self.role,self.content=role,content
        models=ModuleType('models');module=ModuleType('models.inference');module.ChatMessage=Message
        p=SimpleNamespace(mcp_enabled=True,messages=[Message('user','question')])
        calls=[]
        async def retrieve(q):
            calls.append(q);return {'candidates':[event(1)],'scope':'untrusted evidence'}
        with patch.dict(sys.modules,{'models':models,'models.inference':module}):
            await gate.enrich(p,config={'enabled':True},retrieve=retrieve)
            self.assertIn('original observation',p.messages[0].content)
            await gate.enrich(p,config={'enabled':True},retrieve=retrieve)
        self.assertEqual(len(p.messages),2)
        self.assertEqual(calls,['question','question'])
        self.assertEqual(p.messages[-1].content,'question')
    async def test_failure_stops_before_injection(self):
        p=SimpleNamespace(mcp_enabled=True,messages=[SimpleNamespace(role='user',content='question')])
        async def fail(q):raise RuntimeError('offline')
        with self.assertRaises(RuntimeError):await gate.enrich(p,config={'enabled':True},retrieve=fail)
        self.assertEqual(len(p.messages),1)
    async def test_no_silent_input_truncation(self):
        p=SimpleNamespace(mcp_enabled=True,messages=[SimpleNamespace(role='user',content='x'*8001)])
        with self.assertRaises(ValueError):await gate.enrich(p,config={'enabled':True},retrieve=None)
    def test_mcp_error_not_treated_as_success(self):
        with self.assertRaises(RuntimeError):gate.unpack({'is_error':True})

class InstallerTests(unittest.TestCase):
    def test_version_check_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            b=Path(d);(b/'routes').mkdir();r=b/'routes/inference.py';r.write_text('unknown version\n')
            with self.assertRaises(RuntimeError):installer.install(b)
            self.assertEqual(r.read_text(),'unknown version\n')
    def test_install_is_idempotent_and_removable(self):
        with tempfile.TemporaryDirectory() as d:
            b=Path(d);(b/'routes').mkdir();(b/'core/inference').mkdir(parents=True)
            original='async def f(payload):\n'+installer.ANCHOR+'        None,\n    )\n'
            r=b/'routes/inference.py';r.write_text(original)
            with patch.object(Path,'home',return_value=b):
                installer.install(b);first=r.read_text();installer.install(b)
                self.assertEqual(r.read_text(),first)
                installer.install(b,True);self.assertEqual(r.read_text(),original)

if __name__=='__main__':unittest.main()
