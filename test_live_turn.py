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

if __name__=='__main__':unittest.main()
