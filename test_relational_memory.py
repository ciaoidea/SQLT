"""Real MariaDB controls for complete filtering, pagination and isolation."""
import unittest
import uuid
from episodic_memory import EpisodicMemory
from relational_memory import query


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
    def test_validation_and_no_role_cross_event_matching(self):
        self.assertEqual(query(self.m,[dict(role='entity',value='gamma'),dict(role='line_type',value='ftth')])['total'],0)
        for filters in ([dict(role='entity',operator='raw_sql',value='x')],
                        [dict(role='download_mbps',operator='gt',value='1000 Mbps')]):
            with self.assertRaises(ValueError):query(self.m,filters)
        with self.assertRaises(ValueError):query(self.m,[],limit=101)
