"""Lifecycle/atomicity/reload tests for chat-to-experience coordination."""
import json
import unittest
from unittest.mock import patch
import test_apfc as fixtures
from apfc import APFC,_CACHE
from episodic_memory import EpisodicMemory

class ExperienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.APFCTests.setUpClass()
    setUp=fixtures.APFCTests.setUp
    tearDown=fixtures.APFCTests.tearDown
    event=fixtures.APFCTests.event
    # Reuse disposable namespace and SQL evidence fixtures, not the inherited tests.
    def start(self,key='attempt-1',expected_revision=0,before=None):
        before=before or self.event('Observed: Ethernet connected; mobile reconnection performed.')
        return self.a.start_experience('router','Router stability',key,'mobile-v1','reconnect_mobile',
            {'ethernet_connected_0_1':1},before,'Internet works at the next scheduled check: reward 1; explicit failure: reward 0.',expected_revision)
    def finish(self,key='attempt-1',expected_revision=1,outcome=None,reward=1):
        outcome=outcome or self.event('Observed: Internet works at the next scheduled check.')
        return self.a.complete_experience('router',key,outcome,'Internet works at the next scheduled check.',reward,'tool_observation',expected_revision)
    def test_pending_does_not_train_and_survives_new_connection(self):
        self.start();self.assertEqual(self.a.status()['learned_observations'],0)
        other=EpisodicMemory(self.m.namespace)
        try:
            state=APFC(other).get_case('router')
            self.assertEqual(state['experience_summary']['experiences'][0]['status'],'pending')
            self.assertEqual(len(state['state']['waiting']),1)
            result=APFC(other).predict('mobile-v1',{'ethernet_connected_0_1':1})
            self.assertIsNone(result['actions'][0]['estimated_reward'])
        finally:other.close()
    def test_completion_trains_and_changes_state_once(self):
        before=self.event();outcome=self.event('Observed: Internet works at the next scheduled check.')
        first=self.start(before=before);result=self.finish(outcome=outcome)
        self.assertTrue(result['learning_updated']);self.assertEqual(result['case_revision'],2)
        self.assertLess(abs(result['prediction_after']-1),abs(result['prediction_before']-1))
        self.assertFalse(self.finish(outcome=outcome)['learning_updated'])
        self.assertEqual(self.a.get_case('router')['state']['waiting'],[])
        self.assertEqual(len(self.a.get_case('router')['state']['completed']),1)
        self.assertEqual(self.a.status()['learned_observations'],1)
        self.assertTrue(self.a.start_experience('router','Router stability','attempt-1','mobile-v1','reconnect_mobile',{'ethernet_connected_0_1':1},before,'Internet works at the next scheduled check: reward 1; explicit failure: reward 0.',0)['duplicate'])
        with self.assertRaises(ValueError):self.finish(outcome=outcome,reward=0)
    def test_no_invented_quote_or_hypothetical_evidence_kind(self):
        self.start();outcome=self.event('Maybe the connection will recover.')
        for kind in ('tool_observation','predicted'):
            with self.assertRaises(ValueError):self.a.complete_experience('router','attempt-1',outcome,'Internet works',1,kind,1)
        self.assertEqual(self.a.status()['learned_observations'],0)
        self.assertEqual(self.a.experiences('router')['experiences'][0]['status'],'pending')
    def test_state_failure_rolls_back_learning_and_can_be_retried(self):
        self.start();outcome=self.event('Observed: Internet works at the next scheduled check.')
        with patch.object(self.a,'_update_case',side_effect=RuntimeError('simulated state failure')):
            with self.assertRaises(RuntimeError):self.finish(outcome=outcome)
        self.assertEqual(self.a.status()['learned_observations'],0)
        self.assertIsNone(self.a.predict('mobile-v1',{'ethernet_connected_0_1':1})['actions'][0]['estimated_reward'])
        self.assertEqual(self.a.get_case('router')['revision'],1)
        self.assertEqual(self.a.experiences('router')['experiences'][0]['status'],'pending')
        self.assertTrue(self.finish(outcome=outcome)['learning_updated'])
    def test_stale_revision_and_missing_features_do_not_partially_create(self):
        with self.assertRaises(ValueError):self.a.start_experience('router','Router','attempt','p','a',{},self.event(),'Explicit criterion',0)
        self.assertEqual(self.a.get_case('router')['status'],'unknown')
        self.start();outcome=self.event()
        with self.assertRaises(ValueError):self.finish(expected_revision=0,outcome=outcome)
        self.assertEqual(self.a.status()['learned_observations'],0)
    def test_outcome_retraction_excludes_learning_and_marks_review(self):
        self.start();outcome=self.event('Observed: Internet works at the next scheduled check.');self.finish(outcome=outcome)
        self.m.deactivate([outcome]);r=self.a.get_case('router')
        self.assertEqual(r['experience_summary']['experiences'][0]['status'],'needs_review')
        self.assertIsNone(self.a.predict('mobile-v1',{'ethernet_connected_0_1':1})['actions'][0]['estimated_reward'])
        self.m.restore([outcome]);self.assertEqual(self.a.predict('mobile-v1',{'ethernet_connected_0_1':1})['actions'][0]['examples'],1)
        self.m.forget([outcome]);self.assertEqual(self.a.experiences('router')['total'],0)
    def test_one_compound_source_can_teach_without_duplicating_facts(self):
        e=self.event('Ethernet connected. Mobile data reconnected. Internet works at the next check.')
        self.a.start_experience('router','Router','compound','p','reconnect',{'ethernet_connected_0_1':1},e,'Internet works by next check: 1, observed failure: 0.',0,before_quote='Mobile data reconnected.')
        result=self.a.complete_experience('router','compound',e,'Internet works at the next check.',1,'user_report',1)
        self.assertTrue(result['learning_updated']);self.assertEqual(self.m.status()['events'],1)
        self.assertEqual(self.a.status()['learned_observations'],1)
        with self.assertRaises(ValueError):self.a.observe('p','reconnect',{'ethernet_connected_0_1':1},1,e,e,'cannot-bypass-anchor')
    def test_compound_source_requires_distinct_anchored_quotes(self):
        e=self.event('Mobile data reconnected. Internet works at the next check.')
        self.a.start_experience('router','Router','compound','p','reconnect',{'known':1},e,'Explicit criterion',0,before_quote='Mobile data reconnected.')
        with self.assertRaises(ValueError):self.a.complete_experience('router','compound',e,'Mobile data reconnected.',1,'user_report',1)
        self.assertEqual(self.a.status()['learned_observations'],0)
        self.assertEqual(self.a.experiences('router')['experiences'][0]['status'],'pending')
    def test_criteria_mismatch_rolls_back_new_intervention(self):
        self.start();before=self.event()
        with self.assertRaises(ValueError):self.a.start_experience('router','Router stability','attempt-2','mobile-v1','reconnect_mobile',{'ethernet_connected_0_1':1},before,'A different reward meaning',1)
        self.assertEqual(self.a.get_case('router')['revision'],1)
        self.assertEqual(self.a.experiences('router')['total'],1)
    def test_separate_process_model_state_after_completed_experience(self):
        self.start();self.finish();p=self.a.predict('mobile-v1',{'ethernet_connected_0_1':1})
        _CACHE.clear();other=EpisodicMemory(self.m.namespace)
        try:
            self.assertEqual(APFC(other).get_case('router')['revision'],2)
            self.assertEqual(APFC(other).predict('mobile-v1',{'ethernet_connected_0_1':1})['actions'][0]['estimated_reward'],p['actions'][0]['estimated_reward'])
        finally:other.close()
    def test_pagination_preserves_pending_experiences_beyond_working_window(self):
        for i in range(18):self.start('attempt-'+str(i),expected_revision=i)
        self.assertEqual(len(self.a.get_case('router')['state']['waiting']),16)
        a=self.a.experiences('router','pending',limit=16);self.assertTrue(a['has_more'])
        b=self.a.experiences('router','pending',limit=16,offset=a['next_offset'])
        self.assertEqual(len(a['experiences'])+len(b['experiences']),18)

if __name__=='__main__':unittest.main()
