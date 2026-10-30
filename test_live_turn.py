"""Chat boundary must precede retrieval without exposing the user prompt."""
import unittest
from pathlib import Path
from unittest.mock import patch
from clients.persistent_codex import PersistentCodex
ROOT=Path(__file__).resolve().parent

class TurnTests(unittest.TestCase):
 def test_explicit_turn_start_is_before_retrieval(self):
  sequence=[]
  client=PersistentCodex(ROOT,memory_root=ROOT,namespace='turn_test')
  def emit(namespace,event):sequence.append(('start',namespace,event))
  def retrieve(*args):sequence.append(('retrieve',));raise RuntimeError('test stop')
  with patch('live_dynamics.watching',return_value=True),patch('live_dynamics.emit',side_effect=emit),patch('clients.memory_gate.retrieve',side_effect=retrieve):
   with self.assertRaisesRegex(RuntimeError,'test stop'):client.turn('PRIVATE USER PROMPT')
  self.assertEqual([e[0] for e in sequence],['start','retrieve'])
  self.assertEqual(sequence[0][1],'turn_test')
  self.assertEqual(sequence[0][2]['operation'],'chat_turn')
  self.assertNotIn('PRIVATE USER PROMPT',str(sequence))
 def test_closed_viewer_emits_nothing(self):
  client=PersistentCodex(ROOT,memory_root=ROOT)
  with patch('live_dynamics.watching',return_value=False),patch('live_dynamics.emit') as emit,patch('clients.memory_gate.retrieve',side_effect=RuntimeError('test stop')):
   with self.assertRaisesRegex(RuntimeError,'test stop'):client.turn('prompt')
   emit.assert_not_called()


 def test_remote_context_is_retained_and_next_turn_clears_it(self):
  import shutil, subprocess
  node=shutil.which("node")
  if not node:self.skipTest("Node unavailable")
  code="const fs=require('fs'),vm=require('vm'),assert=require('assert');\nconst elements=new Map(),calls=[];\nconst el=id=>{if(!elements.has(id))elements.set(id,{hidden:false,textContent:'',replaceChildren(){},addEventListener(){}});return elements.get(id);};\nconst sandbox={crypto:require('crypto').webcrypto,performance,Event:class{},document:{hidden:false,body:{dataset:{view:'graph'}},getElementById:el,createElement:()=>({}),addEventListener(){}},window:{dispatchEvent(){},CortexGraph:{beginChatTurn:()=>calls.push('clear'),focusMemory:(records,ids,prepared)=>{calls.push({ids,prepared,records:records.map(r=>r.id)});return true;}}}};\nconst code=fs.readFileSync(process.argv[2],'utf8').replace('return {modelInfo:','return {ingest,modelInfo:');\nvm.runInNewContext(code,sandbox);\nconst ingest=sandbox.window.CortexLive.ingest;\nconst packet=(operation,extra={})=>({operation,phase:'complete',origin:'gate',records:[],duration_ms:0,count:0,...extra});\ningest([packet('chat_turn',{phase:'start',call:'1111111111111111'}),packet('memory_selection',{records:[{id:7,frame:'synthetic',bindings:[]}]}),packet('context_focus',{request_id:'1111111111111111',selected_ids:[7]})]);\nassert.equal(calls[0],'clear');assert.equal(calls[1].prepared,true);assert.deepEqual(calls[1].ids,[7]);assert.deepEqual(calls[1].records,[7]);\ningest([packet('chat_turn',{phase:'start',call:'2222222222222222'}),packet('context_focus',{request_id:'2222222222222222',selected_ids:[]})]);\nassert.equal(calls[2],'clear');assert.deepEqual(calls[3].records,[]);assert.deepEqual(calls[3].ids,[]);\ningest([packet('context_focus',{request_id:'1111111111111111',selected_ids:[7]})]);assert.equal(calls.length,4);\nconsole.log('PASS: successive turns, empty context, stale focus ignored, retrieval label distinct');\n"
  subprocess.run([node,"-",str(ROOT/"sqlt.net/live.js")],input=code,text=True,check=True,capture_output=True)

if __name__=='__main__':unittest.main()
