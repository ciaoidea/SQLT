"""APFC controls: actual MariaDB with disposable evidence namespaces; no LLM."""
import json
import time
import unittest
import uuid
import numpy as np
from apfc import APFC, LocalModel, ensure_schema, _CACHE
from episodic_memory import EpisodicMemory, tensor

class NumericalTests(unittest.TestCase):
    def test_learning_generalizes_beyond_training_points(self):
        rng=np.random.default_rng(42);model=LocalModel()
        w=np.zeros(17);w[:4]=[.4,.2,-.15,.1]
        for _ in range(80):
            x=np.zeros(17);x[0]=1;x[1:4]=rng.uniform(-1,1,3)
            model.observe(x,float(x@w))
        errors=[]
        for _ in range(50):
            x=np.zeros(17);x[0]=1;x[1:4]=rng.uniform(-1,1,3)
            errors.append((model.predict(x)-x@w)**2)
        self.assertLess(np.mean(errors),1e-5)

class APFCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m=EpisodicMemory('check_apfc_schema')
        try:ensure_schema(m.c)
        finally:m.close()
    def setUp(self):
        self.m=EpisodicMemory('check_'+uuid.uuid4().hex[:12]);self.a=APFC(self.m)
    def tearDown(self):
        self.m.delete_namespace();self.m.close()
    def event(self,source='Fixture observed evidence'):
        roles=[dict(role='entity',value='test office')]
        # Synthetic evidence fixtures deliberately have no semantic embeddings:
        # these tests exercise new SQL/numerical functionality, not the encoder.
        with self.m.c.cursor() as q:
            q.execute('INSERT INTO episodic_events(namespace,frame,negated,event_time,place,roles,source,provenance,representation) VALUES(%s,%s,0,%s,%s,%s,%s,%s,%s)',(self.m.namespace,'check_context','','',json.dumps(roles),source,'supervised_example',tensor(roles).tobytes()));ident=q.lastrowid
            q.execute('INSERT INTO episodic_bindings VALUES(%s,%s,%s)',(ident,'entity','test office'))
        self.m.c.commit();return ident
    def state(self,event):return {'completed':[{'text':'Request sent','evidence_event_ids':[event]}]}
    def test_identity_scopes_collisions_and_exact_links(self):
        e=self.event();self.a.bind_entity('org-a-it','IT','org-a',['ced'],[e]);self.a.bind_entity('org-b-it','IT','org-b',['ced'],[e])
        self.assertTrue(self.a.resolve('ced')['ambiguous'])
        self.assertEqual(self.a.resolve('ced','org-a')['entities'][0]['entity_key'],'org-a-it')
        with self.assertRaises(ValueError):self.a.bind_entity('other','Other','org-a',['ced'],[e])
        self.assertEqual(self.a.status()['counts']['apfc_entities'],2)
        self.a.link_entity('org-a-it',e,'entity')
        with self.assertRaises(ValueError):self.a.link_entity('org-b-it',e,'entity')
        self.assertEqual(self.a.entity_context('org-a-it')['events'][0]['event_id'],e)
        self.m.deactivate([e]);self.assertEqual(self.a.resolve('ced')['entities'],[])
    def test_case_restart_conflict_idempotency_and_patch(self):
        e=self.event();patch=self.state(e)
        r=self.a.update_case('case','Example',patch,0,'request-1');self.assertEqual(r['revision'],1)
        self.assertTrue(self.a.update_case('case','Example',patch,0,'request-1')['duplicate'])
        with self.assertRaises(ValueError):self.a.update_case('case','Example',patch,0,'request-2')
        other=EpisodicMemory(self.m.namespace)
        try:
            a=APFC(other);self.assertEqual(a.get_case('case')['state'],patch)
            a.update_case('case','Example',{'waiting':[{'text':'Reply awaited','evidence_event_ids':[e]}]},1,'request-2')
            self.assertEqual(len(a.get_case('case')['state']['completed']),1)
            self.assertEqual(len(a.related_cases([e])['cases']),1)
        finally:other.close()
    def test_concurrent_case_updates_do_not_lose_writes(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        e=self.event();self.a.update_case('case','Example',self.state(e),0,'initial')
        barrier=Barrier(2)
        def writer(index):
            m=EpisodicMemory(self.m.namespace)
            try:
                barrier.wait(timeout=5)
                try:
                    APFC(m).update_case('case','Example',{'questions':[{'text':str(index),'evidence_event_ids':[e]}]},1,'writer-'+str(index))
                    return 'written'
                except ValueError:return 'conflict'
            finally:m.close()
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(writer,[1,2]))
        self.assertCountEqual(results,['written','conflict'])
        self.assertEqual(self.a.get_case('case')['revision'],2)

    def test_archived_evidence_hidden_and_forgetting_scrubs_history(self):
        e=self.event();self.a.update_case('case','Example',self.state(e),0,'write')
        self.m.deactivate([e]);r=self.a.get_case('case');self.assertEqual(r['state']['completed'],[]);self.assertEqual(r['status'],'needs_review')
        self.m.restore([e]);self.assertEqual(len(self.a.get_case('case')['state']['completed']),1)
        self.m.forget([e]);self.assertEqual(self.a.get_case('case')['state']['completed'],[])
        with self.m.c.cursor() as q:
            q.execute('SELECT r.state FROM apfc_case_revisions r JOIN apfc_cases c ON c.id=r.case_id WHERE c.namespace=%s',(self.m.namespace,))
            self.assertTrue(all('Request sent' not in row[0] for row in q.fetchall()))
    def test_namespace_isolation_and_inactive_evidence(self):
        e=self.event();other=EpisodicMemory('check_'+uuid.uuid4().hex[:12])
        try:
            with self.assertRaises(ValueError):APFC(other).update_case('case','Example',self.state(e),0,'write')
            self.assertEqual(APFC(other).get_case('case')['status'],'unknown')
        finally:other.delete_namespace();other.close()
    def test_policy_validation_and_untrained_is_unknown(self):
        self.a.register_policy('p',['load_scaled'],['retry','wait'])
        self.assertTrue(all(x['estimated_reward'] is None for x in self.a.predict('p',{'load_scaled':0})['actions']))
        for f in ({},{'load_scaled':2},{'load_scaled':float('nan')}):
            with self.assertRaises(ValueError):self.a.predict('p',f)
        with self.assertRaises(ValueError):self.a.register_policy('p',['different_units'],['retry'])
    def test_policy_reload_feedback_duplicates_and_evidence_revocation(self):
        self.a.register_policy('p',['load_scaled'],['retry']);before=self.event();outcome=self.event()
        args=('p','retry',{'load_scaled':.2},1,before,outcome,'first')
        self.a.observe(*args);self.assertTrue(self.a.observe(*args)['duplicate'])
        with self.assertRaises(Exception):self.a.observe('p','retry',{'load_scaled':.2},0,before,outcome,'other')
        first=self.a.predict('p',{'load_scaled':.2})['actions'][0]
        _CACHE.clear();other=EpisodicMemory(self.m.namespace)
        try:self.assertAlmostEqual(APFC(other).predict('p',{'load_scaled':.2})['actions'][0]['estimated_reward'],first['estimated_reward'])
        finally:other.close()
        self.m.deactivate([outcome]);self.assertIsNone(self.a.predict('p',{'load_scaled':.2})['actions'][0]['estimated_reward'])
        self.m.restore([outcome]);self.assertEqual(self.a.predict('p',{'load_scaled':.2})['actions'][0]['examples'],1)
        self.m.forget([outcome]);self.assertEqual(self.a.predict('p',{'load_scaled':.2})['actions'][0]['examples'],0)
    def test_corrupt_operator_rejected_even_after_cache_hit(self):
        self.a.register_policy('p',['x'],['a']);b=self.event();o=self.event();self.a.observe('p','a',{'x':0},1,b,o,'one')
        self.a.predict('p',{'x':0})
        with self.m.c.cursor() as q:q.execute('UPDATE apfc_models m JOIN apfc_policies p ON p.id=m.policy_id SET m.checksum=%s WHERE p.namespace=%s',('0'*64,self.m.namespace))
        self.m.c.commit()
        with self.assertRaises(ValueError):self.a.predict('p',{'x':0})
    def test_reset_removes_derived_state(self):
        from memory_admin import reset_memory
        e=self.event();self.a.update_case('case','Example',self.state(e),0,'one')
        self.a.register_policy('p',['x'],['a'])
        reset_memory(self.m.namespace,'RESET '+self.m.namespace)
        self.assertEqual(self.a.status()['counts'],{'apfc_entities':0,'apfc_cases':0,'apfc_policies':0})
    def test_mcp_schema_and_calls(self):
        from mcp_server import apfc_case,apfc_update_case,APFCStatePatch
        e=self.event();patch=APFCStatePatch(**self.state(e))
        apfc_update_case('case','Example',patch,0,'one',namespace=self.m.namespace)
        self.assertEqual(apfc_case('case',namespace=self.m.namespace)['revision'],1)

if __name__=='__main__':unittest.main()
