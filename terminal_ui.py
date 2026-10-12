"""Optional rich terminal presentation, with a plain-output fallback."""
import os
import sys
import time
from terminal_input import create_prompt_session,PauseControl

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    Console = None


class TerminalChat:
    def __init__(self):
        self.enabled = bool(Console and sys.stdout.isatty() and sys.stdin.isatty()
                            and os.environ.get('TERM') != 'dumb' and 'NO_COLOR' not in os.environ)
        self.console = Console(highlight=False) if self.enabled else None
        self.live = None
        self.last_update = 0.0
        self.answer = ''
        self.input_session = None

    def banner(self, device):
        if not self.enabled:
            print('Device: ' + device, flush=True)
            return
        self.console.print()
        self.console.print(Panel(Group(
            Text('SQLT', style='bold cyan'),
            Text('Neural weights in MariaDB  ·  ' + device, style='dim'),
            Text('/learn FILE   /reset   /status   /help   /quit', style='dim')),
            border_style='bright_black', padding=(1, 2)))
        self.console.print()

    def prompt(self):
        return '\033[1;36m❯\033[0m ' if self.enabled else 'You> '

    def read_input(self):
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            return input(self.prompt())
        if self.input_session is None:
            try:
                self.input_session=create_prompt_session()
            except ImportError:
                import readline
                return input(self.prompt())
        from prompt_toolkit.formatted_text import ANSI
        return self.input_session.prompt(ANSI(self.prompt()),
            bottom_toolbar='↑↓ history  ·  paste text  ·  Enter send  ·  Alt+Enter newline  ·  Esc pause while generating')

    def pause_control(self):
        return PauseControl(self)

    def pause_state(self, paused):
        if self.live:
            header=Text('Ⅱ Paused — Esc to resume' if paused else '● SQLT',style='bold cyan')
            body=Markdown(self.answer) if self.answer else Text('Thinking…',style='dim')
            self.live.update(Group(header,body),refresh=True)
        elif paused:
            print('Paused — press Esc to resume.',flush=True)

    def start(self):
        if not self.enabled:
            return
        self.answer = ''
        self.last_update = 0.0
        self.console.print()
        self.live = Live(Text('● Thinking…', style='dim'), console=self.console,
                         auto_refresh=False, vertical_overflow='visible')
        self.live.start(refresh=True)

    def progress(self, seconds):
        if self.live and not self.answer:
            self.live.update(Text(f'● Thinking…  {seconds:.0f}s', style='dim'), refresh=True)

    def stream(self, value, force=False):
        if not self.live:
            return
        self.answer = value
        now = time.monotonic()
        if force or now - self.last_update >= 0.12:
            body = Markdown(value) if value else Text('● Thinking…', style='dim')
            self.live.update(Group(Text('● SQLT', style='bold cyan'), body), refresh=True)
            self.last_update = now

    def finish(self, result=None):
        if not self.enabled:
            return
        if self.live:
            if result:
                self.stream(result['answer'], force=True)
            self.live.stop()
            self.live = None
        if result and 'seconds' in result:
            tokens = result.get('tokens', 0)
            seconds = result['seconds']
            footer=f'{seconds:.2f}s  ·  local math' if result.get('engine')=='sympy' else f'{seconds:.1f}s  ·  {tokens} tokens'
            self.console.print(footer, style='dim')
        self.console.print()

    def error(self, message):
        self.finish()
        self.console.print(Text('Error: ' + str(message), style='red'))

    def help(self):
        text = ('/learn FILE  Train once over the file’s training blocks\n'
                '/reset       Clear conversation context\n'
                '/status      Show model and cache details\n'
                '/quit        Exit')
        if self.enabled:
            self.console.print(Panel(Text(text), title='Commands', border_style='bright_black'))
        else:
            print(text)
