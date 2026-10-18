"""SQL integration controls; each test owns a disposable namespace."""
import unittest
import uuid
import numpy as np
from episodic_memory import EpisodicMemory,tensor

class MemoryTests(unittest.TestCase):
    def setUp(self):self.m=EpisodicMemory('check_'+uuid.uuid4().hex[:12])
    def tearDown(self):self.m.delete_namespace();self.m.close()
    def roles(self,agent='marco',recipient='luca'):
        return [dict(role='agent',value=agent),dict(role='patient',value='libro'),dict(role='recipient',value=recipient)]
    def test_role_swap_and_negation_are_not_merged(self):
        a=self.m.remember('give',self.roles(),'source')
        self.m.remember('give',self.roles('luca','marco'),'reverse')
        self.m.remember('give',self.roles(),'negative',negated=True)
        result=self.m.find('give',[dict(role='agent',value='marco')])
        self.assertEqual([a['event_id']],[r['event_id'] for r in result])
        self.assertFalse(np.allclose(tensor(self.roles()),tensor(self.roles('luca','marco'))))
    def test_namespace_and_unknown_query(self):
        self.m.remember('give',self.roles(),'source')
        other=EpisodicMemory('check_'+uuid.uuid4().hex[:12])
        try:self.assertEqual(other.find('give',[]),[])
        finally:other.delete_namespace();other.close()
        self.assertEqual(self.m.find('give',[dict(role='patient',value='chiave')]),[])
    def test_empty_exact_recall_offers_separate_active_candidates(self):
        from mcp_server import recall_events,Binding
        roles=[dict(role='entity',value='research office'),dict(role='number',value='00001234')]
        stored=self.m.remember('phone',roles,'The research office number is 00001234.')
        archived=self.m.remember('phone',roles,'Archived observation')
        self.m.deactivate([archived['event_id']])
        self.m.remember('phone',roles,'Negative observation',negated=True)
        self.assertIn(dict(frame='phone',role='entity',value='research office'),self.m.vocabulary()['participants'])
        result=recall_events('phone',[Binding(role='entity',value='research')],namespace=self.m.namespace)
        self.assertEqual(result['events'],[])
        self.assertFalse(result['candidates_are_exact_matches'])
        self.assertEqual([r['event_id'] for r in result['candidate_events']],[stored['event_id']])
        exact=recall_events('phone',[Binding(role='entity',value='research office')],namespace=self.m.namespace)
        self.assertEqual(exact['events'][0]['event_id'],stored['event_id'])
        self.assertEqual(exact['candidate_events'],[])
    def test_invalid_event_is_not_partially_written(self):
        with self.assertRaises(ValueError):self.m.remember('give',self.roles()+[dict(role='agent',value='sara')],'source')
        self.assertEqual(self.m.status()['events'],0)
        with self.assertRaises(ValueError):self.m.remember('give',self.roles(),'source',provenance='generated')
    def test_operator_reload_and_no_self_training(self):
        for i in range(12):
            roles=self.roles('sender_'+str(i),'receiver_'+str(i))
            e=self.m.remember('give',roles,'supervised example')
            self.m.observe_outcome(e['event_id'],'holder','receiver_'+str(i),'Explicit observed outcome')
        other=EpisodicMemory(self.m.namespace)
        try:
            prediction=other.predict('give',self.roles('unknown_sender','unknown_receiver'),'holder')
            self.assertEqual(prediction['value'],'unknown_receiver')
            self.assertEqual(prediction['examples'],12)
            self.assertEqual(other.predict('give',self.roles(),'holder',negated=True)['status'],'unknown')
            self.assertEqual(other.predict('give',self.roles(),'holder')['examples'],12)
        finally:other.close()
    def test_duplicate_feedback_is_rejected(self):
        e=self.m.remember('give',self.roles(),'source')
        self.m.observe_outcome(e['event_id'],'holder','luca','Explicit observation')
        with self.assertRaises(ValueError):self.m.observe_outcome(e['event_id'],'holder','marco','Contradictory observation')
        self.assertEqual(self.m.predict('give',self.roles(),'holder')['examples'],1)
    def test_corrupted_operator_is_rejected(self):
        e=self.m.remember('give',self.roles(),'source')
        self.m.observe_outcome(e['event_id'],'holder','luca','Explicit observation')
        with self.m.c.cursor() as q:q.execute('UPDATE episodic_operators SET checksum=%s WHERE namespace=%s',('0'*64,self.m.namespace))
        self.m.c.commit()
        with self.assertRaises(ValueError):self.m.predict('give',self.roles(),'holder')
    def test_forget_rebuilds_only_affected_operator(self):
        first=self.m.remember('give',self.roles(),'source one')
        second=self.m.remember('give',self.roles('sara','paolo'),'source two')
        third=self.m.remember('lend',self.roles('anna','mario'),'unrelated event')
        self.m.observe_outcome(first['event_id'],'holder','luca','observed one')
        self.m.observe_outcome(second['event_id'],'holder','paolo','observed two')
        self.m.observe_outcome(third['event_id'],'holder','mario','observed unrelated')
        result=self.m.forget([first['event_id']])
        self.assertEqual(result['replayed_observations'],1)
        self.assertEqual(self.m.find('give',[dict(role='agent',value='marco')]),[])
        self.assertEqual(self.m.predict('give',self.roles('sara','paolo'),'holder')['examples'],1)
        self.assertEqual(self.m.predict('lend',self.roles('anna','mario'),'holder')['examples'],1)
        self.m.forget([second['event_id']])
        self.assertEqual(self.m.predict('give',self.roles(),'holder')['status'],'unknown')
        self.assertEqual(self.m.status()['events'],1)
    def test_forget_wrong_namespace_is_atomic(self):
        own=self.m.remember('give',self.roles(),'own')
        other=EpisodicMemory('check_'+uuid.uuid4().hex[:12])
        try:
            foreign=other.remember('give',self.roles(),'foreign')
            with self.assertRaises(ValueError):self.m.forget([own['event_id'],foreign['event_id']])
            self.assertEqual(self.m.status()['events'],1)
            self.assertEqual(other.status()['events'],1)
        finally:other.delete_namespace();other.close()
    def test_forget_empty_and_invalid_ids(self):
        for ids in ([],[True],[-1]):
            with self.assertRaises(ValueError):self.m.forget(ids)

    def test_deactivate_restore_and_operator_rebuild(self):
        first=self.m.remember('give',self.roles(),'original source')
        second=self.m.remember('give',self.roles('sara','paolo'),'second source')
        self.m.observe_outcome(first['event_id'],'holder','luca','first observation')
        self.m.observe_outcome(second['event_id'],'holder','paolo','second observation')
        self.m.deactivate([first['event_id']])
        self.assertEqual(self.m.status()['events'],1)
        self.assertEqual(self.m.status()['archived_events'],1)
        self.assertEqual(self.m.find('give',[dict(role='agent',value='marco')]),[])
        self.assertEqual(self.m.predict('give',self.roles(),'holder')['examples'],1)
        other=EpisodicMemory(self.m.namespace)
        try:
            archived=other.find('give',[dict(role='agent',value='marco')],active=False)
            self.assertEqual(archived[0]['event_id'],first['event_id'])
            self.assertEqual(archived[0]['source'],'original source')
            other.restore([first['event_id']])
            self.assertEqual(other.status()['archived_events'],0)
            self.assertEqual(other.status()['events'],2)
            self.assertEqual(other.predict('give',self.roles(),'holder')['examples'],2)
        finally:other.close()
    def test_inactive_observation_cannot_train(self):
        e=self.m.remember('give',self.roles(),'source')
        self.m.deactivate([e['event_id']])
        with self.assertRaises(ValueError):self.m.observe_outcome(e['event_id'],'holder','luca','new observation')
        self.assertEqual(self.m.vocabulary()['frames'],[])
        self.assertEqual(self.m.vocabulary(active=False)['frames'],['give'])

if __name__=='__main__':unittest.main()
