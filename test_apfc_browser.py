"""APFC object browsing is parameterized, paginated and evidence-filtered."""
import unittest
import test_apfc as fixtures
from apfc_browser import browse
from mcp_server import memory_event

class BrowseTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):fixtures.APFCTests.setUpClass()
 setUp=fixtures.APFCTests.setUp
 tearDown=fixtures.APFCTests.tearDown
 event=fixtures.APFCTests.event
 def test_case_filters_pagination_and_trace(self):
  e=self.event()
  for i in range(3):self.a.update_case('case-'+str(i),'Example '+str(i),{'goals':[dict(text='Goal',evidence_event_ids=[e])]},0,'create')
  r=browse(self.m,'cases',[dict(role='title',operator='contains',value='Example')],limit=2,include_sql=True)
  self.assertEqual(r['total'],3);self.assertTrue(r['has_more']);self.assertEqual(len(r['sql_trace']),2)
  page=browse(self.m,'cases',limit=2,offset=r['next_offset']);self.assertEqual(len(page['records']),1)
  self.assertNotIn('state',r['records'][0])
 def test_literal_filters_reject_field_and_operator_injection(self):
  for f in ([dict(role='title; DROP TABLE apfc_cases',value='x')],[dict(role='title',operator='sql',value='x')]):
   with self.assertRaises(ValueError):browse(self.m,'cases',f)
  r=browse(self.m,'cases',[dict(role='title',operator='contains',value="%' OR 1=1 --")]);self.assertEqual(r['total'],0)
 def test_inactive_evidence_excludes_identity_and_source(self):
  e=self.event();self.a.bind_entity('office','office','test',[],[e])
  self.assertEqual(browse(self.m,'entities')['total'],1)
  self.m.deactivate([e]);self.assertEqual(browse(self.m,'entities')['total'],0)
  self.assertIsNone(memory_event(e,namespace=self.m.namespace)['event'])
 def test_memory_event_scope_and_active_source(self):
  e=self.event('Original exact source')
  self.assertEqual(memory_event(e,namespace=self.m.namespace)['event']['source'],'Original exact source')
  self.assertIsNone(memory_event(e,namespace='main')['event'])

if __name__=='__main__':unittest.main()
