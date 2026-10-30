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
        self.assertEqual(r['evidence_contract']['current_operational_status'],'not_verified')
        self.assertEqual(r['evidence_contract']['live_source_checks'],[])
        self.assertEqual(r['evidence_contract']['missing_update_means'],'unknown_not_still_broken')
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

class LiteralDiscoveryTests(unittest.TestCase):
    def test_partial_person_name_is_discovered_without_guessing_a_frame(self):
        from memory_context import lexical_candidates
        class Cursor:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def execute(self,sql,params):
                if 'SELECT b.event_id' in sql:
                    self.rows=([(1,0,15)] if 'MIN(CHAR_LENGTH' in sql else [(1,0)]) if params[0]=='surname' else []
                else:
                    self.rows=[(1,'case_notes',False,json.dumps([{'role':'person','value':'Example Surname'}]),'original source','2000-01-01','')]
            def fetchall(self):return self.rows
        memory=SimpleNamespace(namespace='isolated',c=SimpleNamespace(cursor=lambda:Cursor()))
        found=lexical_candidates(memory,'who is Surname',exact_anchor_required=False)
        self.assertEqual(found['events'][0]['frame'],'case_notes')
        # Existing strict-anchor hybrid behavior is unchanged.
        self.assertEqual(lexical_candidates(memory,'who is Surname')['events'],[])

class StructuralIdentityTests(unittest.TestCase):
    def test_descriptive_value_cannot_displace_subject_identity(self):
        from memory_context import lexical_candidates
        class Cursor:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def execute(self,sql,params):
                if 'SELECT b.event_id' in sql:
                    self.rows=[]
                    if params[0]=='surname':self.rows=[(11,0,15)]
                    if params[0]=='attribute':
                        # An exact label in descriptive text must not participate
                        # in identity discovery at all; SQL restricts typed roles.
                        if 'b.role IN' not in sql:self.rows=[(99,1)]
                else:
                    self.rows=[(11,'profile',False,json.dumps([{'role':'entity','value':'Example Surname'}]),'source','','')]
            def fetchall(self):return self.rows
        memory=SimpleNamespace(namespace='isolated',c=SimpleNamespace(cursor=lambda:Cursor()))
        found=lexical_candidates(memory,'attribute of Surname',exact_anchor_required=False)
        self.assertEqual([e['event_id'] for e in found['events']],[11])

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
    async def test_completed_read_plan_does_not_return_to_tool_search(self):
        class Message:
            def __init__(self,role,content):self.role,self.content=role,content
        models=ModuleType('models');module=ModuleType('models.inference');module.ChatMessage=Message
        p=SimpleNamespace(mcp_enabled=True,messages=[Message('user','question')],tools=[{'type':'function'}])
        async def retrieve(q):return {'candidates':[],'plan':{'mode':'relational'},'outcome':'no_matching_records_for_this_plan'}
        with patch.dict(sys.modules,{'models':models,'models.inference':module}):
            await gate.enrich(p,config={'enabled':True,'read_answer_only':True},retrieve=retrieve)
        self.assertEqual(p.tool_choice,'none')
        self.assertFalse(p.mcp_enabled)
        self.assertFalse(p.enable_tools)
        self.assertIsNone(p.tools)

    async def test_explicit_mandatory_mode_retrieves_with_ui_tools_disabled(self):
        class Message:
            def __init__(self,role,content):self.role,self.content=role,content
        models=ModuleType('models');module=ModuleType('models.inference');module.ChatMessage=Message
        p=SimpleNamespace(mcp_enabled=False,enable_tools=False,tool_choice='none',messages=[Message('user','question')])
        async def retrieve(q):return {'candidates':[event(8)],'plan':{'mode':'relational'}}
        with patch.dict(sys.modules,{'models':models,'models.inference':module}):
            self.assertTrue(await gate.enrich(p,config={'enabled':True,'always_retrieve':True},retrieve=retrieve))
        self.assertIn('original observation',p.messages[0].content)
        self.assertFalse(p.enable_tools)
        self.assertEqual(p.tool_choice,'none')

    async def test_mandatory_mode_failure_still_blocks_generation(self):
        p=SimpleNamespace(mcp_enabled=False,messages=[SimpleNamespace(role='user',content='question')])
        async def fail(q):raise RuntimeError('offline')
        with self.assertRaises(RuntimeError):
            await gate.enrich(p,config={'enabled':True,'always_retrieve':True},retrieve=fail)
        self.assertEqual(len(p.messages),1)

    async def test_read_answer_phase_excludes_previous_generated_false_facts(self):
        class Message:
            def __init__(self,role,content):self.role,self.content=role,content
        models=ModuleType('models');module=ModuleType('models.inference');module.ChatMessage=Message
        p=SimpleNamespace(mcp_enabled=True,messages=[Message('user','earlier subject'),
            Message('assistant','UNVERIFIED_OLD_ANSWER'),Message('tool','OLD_TOOL_RESULT'),Message('user','current subject')])
        async def retrieve(q):return {'candidates':[event(8)],'plan':{'mode':'relational'}}
        with patch.dict(sys.modules,{'models':models,'models.inference':module}):
            await gate.enrich(p,config={'enabled':True,'read_answer_only':True},retrieve=retrieve)
        self.assertEqual([m.role for m in p.messages],['system','user'])
        self.assertNotIn('UNVERIFIED_OLD_ANSWER',str([m.content for m in p.messages]))
        self.assertIn('original observation',p.messages[0].content)
        self.assertEqual(p.messages[-1].content,'current subject')

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

class ExplorationTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):gate.EXPLORATION_SCOPE.set(None)
    async def test_model_gets_real_tools_without_builtin_permission_changes(self):
        class Message:
            def __init__(self,role,content):self.role,self.content=role,content
        models=ModuleType('models');module=ModuleType('models.inference');module.ChatMessage=Message
        core=ModuleType('core');inference=ModuleType('core.inference');tools=ModuleType('core.inference.tools')
        tools._mcp_listing=lambda listed:[];inference.tools=tools
        p=SimpleNamespace(mcp_enabled=True,enable_tools=False,permission_mode='auto',
            tool_choice='none',messages=[Message('user','Earlier context'),Message('assistant','UNVERIFIED'),Message('user','Current question')])
        async def retrieve(q):return {'candidates':[event(8)],'plan':{'mode':'relational'},'live_graph_request_id':'1234567890abcdef'}
        with patch.dict(sys.modules,{'models':models,'models.inference':module,'core':core,'core.inference':inference}):
            await gate.enrich(p,config={'enabled':True,'server_id':'example','explore':True,'read_answer_only':False},retrieve=retrieve)
        self.assertEqual(p.tool_choice,'auto');self.assertTrue(p.mcp_enabled)
        self.assertFalse(p.enable_tools);self.assertEqual(p.permission_mode,'auto')
        self.assertNotIn('UNVERIFIED',str([m.content for m in p.messages]))
        self.assertEqual([m.role for m in p.messages],['system','system','user'])
        self.assertIn('1234567890abcdef',p.messages[0].content)
        self.assertEqual(p.messages[-1].content,'Current question')
    async def test_preload_uses_real_names_and_preserves_validation_schema(self):
        spec={'type':'function','function':{'name':'mcp__example__query_facts','description':'real tool',
              'parameters':{'type':'object','properties':{'limit':{'type':'integer','minimum':1,'maximum':100,'description':'help'}},'additionalProperties':False}}}
        gate.EXPLORATION_SCOPE.set('example')
        result=gate.exploration_listing(lambda listed:[],[({'id':'example'},[],[spec])])
        self.assertEqual(result[0]['function']['name'],spec['function']['name'])
        self.assertEqual(result[0]['function']['parameters']['properties']['limit'],{'type':'integer','minimum':1,'maximum':100})
        self.assertFalse(result[0]['function']['parameters']['additionalProperties'])
        self.assertIn('description',spec['function']['parameters']['properties']['limit'])
        gate.EXPLORATION_SCOPE.set(None)
        self.assertEqual(gate.exploration_listing(lambda listed:['ordinary'],[]),['ordinary'])

class InstallerTests(unittest.TestCase):
    def test_legacy_frontend_interceptor_removed_without_changing_other_scripts(self):
        with tempfile.TemporaryDirectory() as d:
            b=Path(d)/'backend';b.mkdir()
            p=Path(d)/'frontend/dist/index.html';p.parent.mkdir(parents=True)
            p.write_text('<script src="/assets/cortex-read-context-v4.js"></script><script src="app.js"></script>')
            with patch.object(installer.Path,'home',return_value=Path(d)):
                installer.retire_legacy_frontend(b)
            self.assertEqual(p.read_text(),'<script src="app.js"></script>')
            installer.retire_legacy_frontend(b)

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
