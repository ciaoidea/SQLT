"""Terminal input preserving a bracketed multiline paste as one message."""
import sys


def create_session(**kwargs):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    bindings=KeyBindings()
    @bindings.add('escape','enter')
    def newline(event):
        event.current_buffer.insert_text('\n')
    return PromptSession(multiline=False,key_bindings=bindings,**kwargs)


class TerminalInput:
    def __init__(self):
        self.session=create_session() if sys.stdin.isatty() and sys.stdout.isatty() else None
    def read(self, prompt='❯ '):
        if self.session:
            from prompt_toolkit.formatted_text import ANSI
            return self.session.prompt(ANSI(prompt.replace('\001', '').replace('\002', '')))
        return input(prompt)
