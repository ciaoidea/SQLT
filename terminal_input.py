"""Terminal editing and cooperative generation pause; no persistent chat log."""
import os
import select
import sys
import threading
import termios
import tty


def create_prompt_session(**kwargs):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    bindings = KeyBindings()
    @bindings.add('enter')
    def submit(event): event.current_buffer.validate_and_handle()
    @bindings.add('escape','enter')
    def newline(event): event.current_buffer.insert_text('\n')
    @bindings.add('up')
    def previous(event): event.current_buffer.history_backward()
    @bindings.add('down')
    def following(event): event.current_buffer.history_forward()
    return PromptSession(multiline=True, key_bindings=bindings,
                         prompt_continuation='  ', **kwargs)


class PauseControl:
    def __init__(self, ui, fd=None, enabled=None):
        self.ui = ui
        self.enabled = sys.stdin.isatty() if enabled is None else enabled
        self.fd = fd
        self.paused = threading.Event()
        self.stopped = threading.Event()
        self.thread = None
        self.original = None

    def __enter__(self):
        if not self.enabled: return self
        self.fd = sys.stdin.fileno() if self.fd is None else self.fd
        self.original = termios.tcgetattr(self.fd)
        try:
            tty.setcbreak(self.fd)
            self.thread = threading.Thread(target=self.listen, daemon=True)
            self.thread.start()
        except BaseException:
            termios.tcsetattr(self.fd,termios.TCSADRAIN,self.original)
            raise
        return self

    def listen(self):
        while not self.stopped.is_set():
            ready,_,_ = select.select([self.fd],[],[],0.05)
            if not ready: continue
            data = os.read(self.fd,64)
            if not data: break
            if data == b'\x1b':
                # Ignore escape sequences belonging to arrows/function keys.
                following,_,_ = select.select([self.fd],[],[],0.05)
                if following:
                    os.read(self.fd,64)
                    continue
                if self.paused.is_set(): self.paused.clear()
                else: self.paused.set()

    def checkpoint(self):
        if not self.paused.is_set(): return
        self.ui.pause_state(True)
        try:
            while self.paused.is_set() and not self.stopped.is_set():
                self.stopped.wait(0.05)
        finally:
            self.ui.pause_state(False)

    def __exit__(self,*args):
        self.stopped.set()
        self.paused.clear()
        if self.thread: self.thread.join(timeout=0.3)
        if self.original is not None:
            termios.tcsetattr(self.fd,termios.TCSADRAIN,self.original)
