import os
import termios
import threading
import time
import unittest
from unittest.mock import Mock
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from terminal_input import create_prompt_session,PauseControl

class InputTests(unittest.TestCase):
    def test_history_and_bracketed_multiline_paste(self):
        with create_pipe_input() as pipe:
            session=create_prompt_session(input=pipe,output=DummyOutput())
            pipe.send_text('first message\r')
            self.assertEqual(session.prompt('> '),'first message')
            pipe.send_text('\x1b[A\r')
            self.assertEqual(session.prompt('> '),'first message')
            pipe.send_text('\x1b[200~line one\nline two\x1b[201~\r')
            self.assertEqual(session.prompt('> '),'line one\nline two')
            pipe.send_text('\x1b[A\r')
            self.assertEqual(session.prompt('> '),'line one\nline two')

    def test_alt_enter_inserts_newline(self):
        with create_pipe_input() as pipe:
            session=create_prompt_session(input=pipe,output=DummyOutput())
            pipe.send_text('one\x1b\rtwo\r')
            self.assertEqual(session.prompt('> '),'one\ntwo')

    def test_escape_pauses_resumes_and_restores_terminal(self):
        master,slave=os.openpty()
        original=termios.tcgetattr(slave)
        ui=Mock()
        def wait_for(predicate):
            deadline=time.monotonic()+2
            while not predicate() and time.monotonic()<deadline: time.sleep(0.01)
            self.assertTrue(predicate())
        try:
            with PauseControl(ui,fd=slave,enabled=True) as pause:
                os.write(master,b'\x1b[A')
                time.sleep(0.1)
                self.assertFalse(pause.paused.is_set())
                os.write(master,b'\x1b')
                wait_for(pause.paused.is_set)
                finished=threading.Event()
                worker=threading.Thread(target=lambda:(pause.checkpoint(),finished.set()))
                worker.start()
                time.sleep(0.1)
                self.assertFalse(finished.is_set())
                os.write(master,b'\x1b')
                wait_for(finished.is_set)
                worker.join(timeout=1)
            self.assertEqual(termios.tcgetattr(slave),original)
            self.assertEqual(ui.pause_state.call_args_list[0].args,(True,))
            self.assertEqual(ui.pause_state.call_args_list[-1].args,(False,))
        finally:
            os.close(master);os.close(slave)

if __name__=='__main__':unittest.main()
