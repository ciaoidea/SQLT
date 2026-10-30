"""Real MariaDB controls for complete filtering, pagination and isolation."""
import unittest
import uuid
from episodic_memory import EpisodicMemory
from relational_memory import query


class SubjectFactViewTests(unittest.TestCase):
    def test_distinct_scopes_and_multi_subject_relations_are_not_merged(self):
        from relational_memory import subject_facts
        def row(i,roles):return {'event_id':i,'negated':False,'roles':roles}
        groups=subject_facts([row(1,[{'role':'entity','value':'Same'},{'role':'company','value':'one'}]),
                              row(2,[{'role':'entity','value':'Same'},{'role':'company','value':'two'}]),
                              row(3,[{'role':'entity','value':'A'},{'role':'entity','value':'B'},{'role':'property','value':'unspecified owner'}])])
        self.assertEqual(len(groups),2)
        self.assertNotEqual(groups[0]['scope'],groups[1]['scope'])

    def test_exact_subject_grouping_preserves_conflicts_and_provenance(self):
        from relational_memory import subject_facts
        rows=[{'event_id':1,'negated':False,'time':'2000-01-01','roles':[{'role':'entity','value':'Example object'},{'role':'property','value':'old'}]},
              {'event_id':2,'negated':False,'time':'2001-01-01','roles':[{'role':'entity','value':'Example object'},{'role':'property','value':'new'},{'role':'function','value':'main purpose'}]},
              {'event_id':3,'negated':False,'time':'','roles':[{'role':'entity','value':'Example object note'},{'role':'context','value':'incidental'}]}]
        groups=subject_facts(rows)
        self.assertEqual(len(groups),2)
        self.assertEqual([v['value'] for v in groups[0]['attributes']['property']],['old','new'])
        self.assertEqual(groups[0]['attributes']['function'][0]['event_id'],2)
        self.assertNotIn('context',groups[0]['attributes'])

class RelationalTests(unittest.TestCase):
    def setUp(self):
        self.m=EpisodicMemory('check_sql_'+uuid.uuid4().hex[:10])
        self.other=EpisodicMemory(self.m.namespace+'_b')
        def event(name,line,speed,negated=False):
            return dict(frame='connection',roles=[dict(role='entity',value=name),
                dict(role='line_type',value=line),dict(role='download_mbps',value=speed)],
                source=f'{name}: {line}, {speed} Mbps.',negated=negated)
        rows=[event('alpha','FTTH','1000'),event('beta','FTTH','1000.0'),
              event('gamma','FTTC','1000'),event('delta','FTTH','100'),
              event('invalid','FTTH','1000 300'),event('negative','FTTH','1000',True),
              event('archived','FTTH','1000')]
        ids=self.m.remember_many(rows)['event_ids'];self.m.deactivate([ids[-1]])
        self.other.remember_many([event('other namespace','FTTH','1000')])
        self.filters=[dict(role='line_type',value='ftth'),dict(role='download_mbps',operator='number_eq',value=1000)]
    def tearDown(self):
        for m in (self.m,self.other):m.delete_namespace();m.close()
    def test_exact_numeric_filters_exclude_malformed_archived_and_other_namespaces(self):
        result=query(self.m,self.filters)
        self.assertEqual(result['total'],2);self.assertTrue(result['complete'])
        self.assertEqual([e['roles'][0]['value'] for e in result['events']],['alpha','beta'])
    def test_unknown_projection_is_rejected_instead_of_silent_empty_attributes(self):
        with self.assertRaisesRegex(ValueError,'Unknown stored attributes'):
            query(self.m,self.filters,select_roles=['unknown_container_key'])
        self.assertEqual(query(self.m,self.filters,select_roles=['entity'])['total'],2)

    def test_pagination_count_and_projection(self):
        first=query(self.m,self.filters,limit=1,select_roles=['entity'],include_source=False)
        self.assertEqual(first['total'],2);self.assertTrue(first['has_more']);self.assertFalse(first['complete'])
        self.assertNotIn('source',first['events'][0]);self.assertEqual(len(first['events'][0]['roles']),1)
        second=query(self.m,self.filters,limit=1,offset=first['next_offset'])
        self.assertFalse(second['has_more']);self.assertNotEqual(first['events'][0]['event_id'],second['events'][0]['event_id'])
        count=query(self.m,self.filters,count_only=True)
        self.assertEqual(count['total'],2);self.assertEqual(count['events'],[])
    def test_polarity_and_numeric_range(self):
        self.assertEqual(query(self.m,self.filters,negated=None)['total'],3)
        self.assertEqual(query(self.m,[dict(role='download_mbps',operator='lt',value=200)])['total'],1)
    def test_literal_substring_and_injection_are_not_sql(self):
        self.assertEqual(query(self.m,[dict(role='entity',operator='contains',value='alp')])['total'],1)
        for value in ("' OR 1=1 --",'%','_'):
            self.assertEqual(query(self.m,[dict(role='entity',operator='contains',value=value)])['total'],0)
    def test_presence_and_in(self):
        self.assertEqual(query(self.m,[dict(role='missing role',operator='missing')])['total'],5)
        self.assertEqual(query(self.m,[dict(role='missing role',operator='exists')])['total'],0)
        self.assertEqual(query(self.m,[dict(role='entity',operator='in',value=['alpha','beta'])])['total'],2)
    def test_executed_sql_trace_keeps_input_out_of_statement(self):
        value="' OR 1=1 --"
        result=query(self.m,[dict(role='entity',value=value)],include_sql=True)
        self.assertEqual(result['total'],0)
        self.assertEqual(len(result['sql_trace']),2)
        for trace in result['sql_trace']:
            self.assertNotIn(value,trace['sql'])
            self.assertIn(value.casefold(),trace['parameters'])
    def test_vector_sql_filters_before_ranking_and_reports_real_vector(self):
        import base64
        import semantic_memory
        result=semantic_memory.search(self.m,'fiber connection',limit=8,negated=False,
                                     filters=self.filters,frame='connection',include_sql=True)
        self.assertEqual({e['roles'][0]['value'] for e in result['events']},{'alpha','beta'})
        trace=result['sql_trace'][0]
        self.assertIn('VEC_DISTANCE_COSINE',trace['sql'])
        self.assertIn('EXISTS',trace['sql'])
        self.assertEqual(len(base64.b64decode(trace['parameters'][0]['value'])),384*4)
    def test_validation_and_no_role_cross_event_matching(self):
        self.assertEqual(query(self.m,[dict(role='entity',value='gamma'),dict(role='line_type',value='ftth')])['total'],0)
        for filters in ([dict(role='entity',operator='raw_sql',value='x')],
                        [dict(role='download_mbps',operator='gt',value='1000 Mbps')]):
            with self.assertRaises(ValueError):query(self.m,filters)
        with self.assertRaises(ValueError):query(self.m,[],limit=101)
