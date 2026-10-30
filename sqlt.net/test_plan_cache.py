"""Safety and fresh-read guarantees for exact validated-plan reuse."""
import asyncio
from copy import deepcopy
import unittest
from unittest.mock import patch
from planner import PlanCache
from server import MemoryBridge

PLAN={'mode':'relational','dataset':'memory','queries':[{'frame':'fact','filters':[{'role':'entity','operator':'eq','value':'Example'}]}]}

class CacheTests(unittest.TestCase):
 def setUp(self):
  self.now=[0.];self.day=['2000-01-01'];self.c=PlanCache(capacity=2,ttl=10,clock=lambda:self.now[0],day=lambda:self.day[0])
 def test_exact_question_only_and_no_reference_mutation(self):
  self.c.put('Alpha',{},PLAN)
  self.assertIsNone(self.c.get('Beta',{}))
  self.assertIsNone(self.c.get('not Alpha',{}))
  p,_=self.c.get('Alpha',{});p['queries'][0]['frame']='changed'
  self.assertEqual(self.c.get('Alpha',{})[0]['queries'][0]['frame'],'fact')
 def test_catalog_change_day_change_and_expiry_invalidate(self):
  self.c.put('q',{'roles':['entity']},PLAN)
  self.assertIsNone(self.c.get('q',{'roles':['entity','other']}))
  self.day[0]='2000-01-02';self.assertIsNone(self.c.get('q',{'roles':['entity']}))
  self.day[0]='2000-01-01';self.now[0]=10.;self.assertIsNone(self.c.get('q',{'roles':['entity']}))
 def test_capacity_and_unstable_or_clarifying_plans(self):
  for q in ['a','b','c']:self.c.put(q,{},PLAN)
  self.assertEqual(len(self.c.items),2);self.assertIsNone(self.c.get('a',{}))
  self.c.put('clarify',{},dict(PLAN,mode='clarify'));self.assertIsNone(self.c.get('clarify',{}))
  p=deepcopy(PLAN);p['title']='as of 12:45';self.c.put('clock',{},p);self.assertIsNone(self.c.get('clock',{}))

class ExecutionTests(unittest.IsolatedAsyncioTestCase):
 async def make_bridge(self):
  b=object.__new__(MemoryBridge);b.lock=asyncio.Lock();b.planning_lock=asyncio.Semaphore(1);b.plan_cache=PlanCache()
  async def catalog():return {'frames':['fact'],'roles':['entity']}
  async def hints(name,args):return {'events':[]}
  b._call=hints;b._catalog=catalog;return b
 async def test_hit_skips_model_but_reads_updated_data(self):
  b=await self.make_bridge();reads=[]
  async def execute(q,p,seconds):
   reads.append(q);return {'records':[{'revision':len(reads)}],'planning_seconds':seconds}
  b._natural=execute
  with patch('server.plan',return_value=(PLAN,12.)) as model:
   first=await b._browse('natural','q');second=await b._browse('natural','q')
  self.assertEqual(model.call_count,1);self.assertEqual(len(reads),2)
  self.assertFalse(first['plan_cache']['hit']);self.assertTrue(second['plan_cache']['hit'])
  self.assertEqual(second['records'][0]['revision'],2)
  self.assertEqual(second['planning_seconds'],0)
 async def test_changed_identity_evidence_replans_even_with_same_schema(self):
  b=await self.make_bridge();current=[1]
  async def hints(name,args):return {'events':[{'event_id':current[0]}]}
  async def execute(*args):return {'records':[]}
  b._call=hints;b._natural=execute
  with patch('server.plan',return_value=(PLAN,1.)) as model:
   await b._browse('natural','q');current[0]=2
   await b._browse('natural','q')
  self.assertEqual(model.call_count,2)
 async def test_failed_execution_evicts_plan(self):
  b=await self.make_bridge()
  async def fail(*args):raise ValueError('invalid query')
  b._natural=fail
  with patch('server.plan',return_value=(PLAN,1.)):
   with self.assertRaises(ValueError):await b._browse('natural','q')
  self.assertIsNone(b.plan_cache.get('q',{'frames':['fact'],'roles':['entity'],'literal_query_evidence':[]}))

if __name__=='__main__':unittest.main()
