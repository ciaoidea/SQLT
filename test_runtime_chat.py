"""Check visible progress and its cleanup when a response starts."""
import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock,patch
import torch
from runtime import main


class ChatTests(unittest.TestCase):
    def test_wait_indicator_clears_before_answer_and_callback_is_removed(self):
        cache=SimpleNamespace(database_loads=12,progress_callback=None)
        model=SimpleNamespace(device=torch.device('cpu'),cache=cache,close=Mock())
        def generate(messages,max_tokens,callback):
            cache.progress_callback()
            callback('Ciao')
            return {'answer':'Ciao','finish_reason':'eos'}
        model.generate=generate
        output=io.StringIO()
        with patch('sys.argv',['runtime.py','--device','cpu','chat']), \
             patch('runtime.DatabaseMoE',return_value=model), \
             patch('builtins.input',side_effect=['Ciao','/quit']), \
             redirect_stdout(output):
            main()
        self.assertIn('Attendi:',output.getvalue())
        self.assertIn('\033[2KModel> Ciao',output.getvalue())
        self.assertIsNone(cache.progress_callback)
        model.close.assert_called_once()


if __name__=='__main__':
    unittest.main()
