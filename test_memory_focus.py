"""Visual semantic selection is explicit, bounded, request-scoped and read-only."""
import unittest
from unittest.mock import patch
import live_dynamics as live

class FocusTests(unittest.TestCase):
 def test_focus_emits_only_ids_not_sources_or_prompts(self):
  with patch.object(live,'watching',return_value=True),patch.object(live,'emit') as emit:
   result=live.request_focus('scope','0123456789abcdef',[2,3,2])
   namespace,event=emit.call_args.args
   self.assertEqual(namespace,'scope')
   self.assertEqual(event['selected_ids'],[2,3])
   self.assertEqual(event['request_id'],'0123456789abcdef')
   self.assertEqual(event['records'],[])
   self.assertFalse(result['memory_changed'])
 def test_inactive_viewer_does_not_emit(self):
  with patch.object(live,'watching',return_value=False),patch.object(live,'emit') as emit:
   self.assertEqual(live.request_focus('main','0123456789abcdef',[])['status'],'viewer_inactive')
   emit.assert_not_called()
 def test_invalid_ids_or_request_are_rejected(self):
  for request,ids in [('bad',[]),('0123456789abcdef',[-1]),('0123456789abcdef',[True]),('0123456789abcdef',list(range(1,130)))]:
   with self.assertRaises(ValueError):live.request_focus('main',request,ids)
 def test_explicit_empty_selection_is_supported(self):
  with patch.object(live,'watching',return_value=True),patch.object(live,'emit') as emit:
   live.request_focus('main','0123456789abcdef',[])
   self.assertEqual(emit.call_args.args[1]['selected_ids'],[])

if __name__=='__main__':unittest.main()
