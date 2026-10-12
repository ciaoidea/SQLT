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
            if getattr(model,'planning_mode',False):
                return {'answer':'{"action":"answer"}'}
            cache.progress_callback()
            if callback:callback('Ciao')
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

    def test_learn_file_uses_one_pass_and_reloads(self):
        import tempfile
        from pathlib import Path
        first=SimpleNamespace(device=torch.device('cpu'),close=Mock(),
            tokenizer=SimpleNamespace(encode=lambda *a,**k:list(range(50))))
        second=SimpleNamespace(close=Mock())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'example with spaces.txt'
            path.write_text('An explanation about triangles.',encoding='utf-8')
            with patch('sys.argv',['runtime.py','chat']), \
                 patch('runtime.DatabaseMoE',side_effect=[first,second]) as loader, \
                 patch('runtime.subprocess.run',return_value=SimpleNamespace(returncode=0)) as trainer, \
                 patch('builtins.input',side_effect=[f'/learn {path}','/quit']), \
                 redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(trainer.call_args.args[0][-2:],['--one-pass',str(path)])
            self.assertEqual(loader.call_count,2)
            self.assertTrue(path.exists())
            first.close.assert_called_once()
            second.close.assert_called_once()

    def test_missing_file_and_help_do_not_start_training(self):
        model=SimpleNamespace(device=torch.device('cpu'),close=Mock())
        with patch('sys.argv',['runtime.py','chat']), \
             patch('runtime.DatabaseMoE',return_value=model) as loader, \
             patch('runtime.subprocess.run') as trainer, \
             patch('builtins.input',side_effect=['/learn','/learn /nonexistent/sqlt-file.txt','/quit']), \
             redirect_stdout(io.StringIO()):
            main()
        loader.assert_called_once()
        trainer.assert_not_called()



if __name__=='__main__':
    unittest.main()
