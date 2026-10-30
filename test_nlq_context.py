import hashlib
from pathlib import Path
import unittest
from nlq_context import context_from_result

ROOT=Path(__file__).resolve().parent

def response(**kw):
    d={'memory_origin':{'installation':hashlib.sha256(str(ROOT).encode()).hexdigest()[:24],'namespace':'isolated'},
       'plan':{'mode':'relational','dataset':'memory','queries':[{'frame':'fact','filters':[]}],'clarification':''},
       'partial':False,'records':[]}
    d.update(kw);return d

class NaturalContextTests(unittest.TestCase):
    def test_empty_query_is_not_a_health_verdict(self):
        r=context_from_result(response(),'isolated')
        self.assertEqual(r['candidates'],[])
        self.assertEqual(r['outcome'],'no_matching_records_for_this_plan')
        self.assertEqual(r['evidence_contract']['current_operational_status'],'not_verified')
    def test_rejects_another_database_installation(self):
        r=response(memory_origin={'installation':'another','namespace':'isolated'})
        with self.assertRaises(ValueError):context_from_result(r,'isolated')
    def test_rejects_another_namespace(self):
        with self.assertRaises(ValueError):context_from_result(response(),'other')
    def test_clarification_does_not_fall_back_to_old_records(self):
        d=response(plan={'mode':'clarify','dataset':'memory','clarification':'Date not available'},records=[{'source':'old incident'}])
        r=context_from_result(d,'isolated')
        self.assertEqual(r['outcome'],'clarification_required')
        self.assertEqual(r['candidates'],[])
    def test_preserves_dates_negation_and_partial_coverage(self):
        e={'event_id':1,'frame':'fact','roles':[],'source':'x'*1200,'time':'2000-01-01','negated':True}
        r=context_from_result(response(records=[e],partial=True),'isolated')
        self.assertTrue(r['candidates'][0]['negated'])
        self.assertEqual(r['candidates'][0]['time'],'2000-01-01')
        self.assertTrue(r['candidates'][0]['source_truncated'])
        self.assertFalse(r['coverage']['complete_database_search'])
        self.assertTrue(r['coverage']['query_partial'])

if __name__=='__main__':unittest.main()
