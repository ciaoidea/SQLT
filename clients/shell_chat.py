"""Streaming Codex terminal agent; Cortex MCP remains the only memory backend."""
import argparse
import json
import os
from pathlib import Path
import re
import pwd
import shlex
import shutil
import socket
import subprocess
import sys

from .persistent_codex import PersistentCodex
from .terminal_input import TerminalInput
from .terminal_ui import AnswerStream, shell_prompt

ROOT = Path(__file__).resolve().parents[1]


def safe_text(text):
    # Model/tool output cannot inject terminal controls, OSC clipboard or titles.
    text = re.sub(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)', '', str(text))
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    return ''.join(c for c in text if c in '\n\t' or (ord(c) >= 32 and not 127 <= ord(c) < 160))


def instructions(namespace):
    return f'''You are Cortex, a terminal agent powered by Codex. Answer concisely in Italian.
Use native shell, file editing, web, skills and configured MCP/connectors for the user's task.
Operate within the user's request; never send messages or publish private data without authorization.
Do not claim a tool or connector exists before it is available. Report failures honestly.
Persistent knowledge is exclusively Cortex MCP, namespace {namespace}. Never create a second
Markdown, JSON or SQLite knowledge store, or load legacy APFC/MD-OS memory.
For memory questions consult Cortex even if the answer appears in conversation. Use query_memory
for exact filters, counts and exhaustive lists, following pagination; semantic_recall for meaning.
For memory questions naming a person or entity, also use query_memory with a literal
contains filter on role entity and a distinctive name part actually supplied by the user.
Do this even when semantic_recall has no relevant match: top-k results and the bounded
memory_status catalog do not prove absence. If entity has no match, try known alias
and person roles separately; never AND different role alternatives. Follow pagination.
Verify identity from sources, then retrieve all matching person_context/case_context
records and relevant linked context before summarizing. Preserve the dates and distinguish
a draft or request from a sent message or completed action. Do not say no memory exists
until the applicable structured name searches have also returned no matching evidence.
Preserve evidence, identity, negation and dates. Similarity is not proof. Store new explicit user
facts with remember_events, quoting the original evidence; keep related entity fields together.
Do not save generated guesses or ordinary task instructions as knowledge. Forgetting follows
Cortex MCP archival/deletion semantics. Never reset a memory namespace without explicit confirmation.
Use tools to verify actual changes. Keep replies clean: no internal DB IDs or trace JSON unless asked.
For memory lookups, return only the answer. Do not announce searching, consulting
the database or using memory, and do not send interim commentary for these lookups.
'''


class Presentation:
    def __init__(self):
        self.renderer = AnswerStream()
        self.current = None
        self.streamed = set()
        self.calls = []
        self.phases = {}

    def boundary(self):
        self.renderer.finish()
        if self.current is not None:
            print('\n', flush=True)
        self.current = None
        self.renderer = AnswerStream()

    def event(self, method, params):
        item = params.get('item', {})
        if method == 'item/started' and item.get('type') == 'agentMessage':
            self.phases[item.get('id')] = item.get('phase')
        if (method == 'item/agentMessage/delta' and
                self.phases.get(params.get('itemId')) == 'commentary'):
            return
        if method == 'item/agentMessage/delta':
            item_id = params.get('itemId')
            if self.current != item_id:
                self.boundary()
                self.current = item_id
                print('● ', end='', flush=True)
            self.streamed.add(item_id)
            self.renderer.write(safe_text(params.get('delta', '')))
        elif method == 'item/started' and item.get('type') in ('commandExecution', 'mcpToolCall', 'fileChange', 'webSearch'):
            if item.get('type') == 'mcpToolCall' and item.get('server') == 'cortex':
                return
            self.boundary()
            kind = item['type']
            label = (item.get('command') if kind == 'commandExecution' else
                     f"{item.get('server', '')}.{item.get('tool', '')}" if kind == 'mcpToolCall' else
                     'Modifica file' if kind == 'fileChange' else 'Ricerca web')
            print('› ' + safe_text(label or kind)[:240], flush=True)
        elif method == 'item/completed':
            if item.get('type') == 'agentMessage' and (item.get('phase') or self.phases.get(item.get('id'))) == 'commentary':
                return
            if item.get('type') == 'agentMessage' and item.get('id') not in self.streamed:
                self.boundary()
                self.current = item.get('id', 'answer')
                print('● ', end='', flush=True)
                self.renderer.write(safe_text(item.get('text', '')))
            elif item.get('type') in ('commandExecution', 'mcpToolCall', 'fileChange', 'webSearch'):
                self.calls.append({key: item[key] for key in ('type', 'command', 'server', 'tool', 'status', 'exitCode') if key in item})
                if item.get('exitCode') not in (None, 0) or item.get('status') == 'failed':
                    print('› Operazione non riuscita: ' + safe_text(item.get('exitCode', item.get('status'))), flush=True)


def is_shell(text, cwd):
    # Multiline paste remains one language message unless explicitly prefixed.
    if '\n' in text or '\r' in text:
        return False
    try:
        words = shlex.split(text)
    except ValueError:
        return False
    if not words:
        return False
    first = words[0]
    if first in {'cd', 'pwd', 'export', 'unset'}:
        return True
    if '/' in first:
        path = Path(first).expanduser()
        path = path if path.is_absolute() else cwd / path
        return path.is_file() and os.access(path, os.X_OK)
    return shutil.which(first) is not None


def prompt(cwd):
    directory = str(cwd)
    home = str(Path.home())
    if directory == home or directory.startswith(home + '/'):
        directory = '~' + directory[len(home):]
    identity = safe_text(f'{pwd.getpwuid(os.geteuid()).pw_name}@{socket.gethostname().split(".")[0]}')
    return shell_prompt(identity, safe_text(directory))


def run_shell(command, cwd):
    if command.strip() in ('exit', 'quit'):
        return None
    parts = shlex.split(command)
    if not parts:
        return cwd
    lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|<>()')
    lexer.whitespace_split = True
    simple = not any(token and all(c in ';&|<>()' for c in token) for token in lexer)
    if simple and parts[0] == 'cd':
        if len(parts) > 2:
            raise ValueError('Uso: cd PERCORSO')
        target = Path(os.path.expandvars(parts[1])).expanduser() if len(parts) == 2 else Path.home()
        if not target.is_absolute():
            target = cwd / target
        target = target.resolve(strict=True)
        if not target.is_dir():
            raise ValueError('La destinazione non è una cartella')
        return target
    if simple and parts[0] == 'export':
        updates = {}
        for entry in parts[1:]:
            key, separator, value = entry.partition('=')
            if not separator or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', key):
                raise ValueError('Uso: export NAME=value')
            updates[key] = value
        if updates:
            os.environ.update(updates)
            return cwd
    if simple and parts[0] == 'unset':
        for key in parts[1:]:
            os.environ.pop(key, None)
        return cwd
    subprocess.run(command, shell=True, executable=os.environ.get('SHELL', '/bin/bash'), cwd=cwd)
    return cwd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--namespace', default='main')
    p.add_argument('--cwd', type=Path, default=ROOT)
    p.add_argument('--model', default='gpt-6.1-sol')
    p.add_argument('--ask')
    p.add_argument('--timeout', type=float, default=1800)
    a = p.parse_args()
    if not a.namespace.isidentifier() or len(a.namespace) > 32:
        p.error('Invalid namespace')
    cwd = a.cwd.expanduser().resolve(strict=True)
    if not cwd.is_dir():
        p.error('cwd must be a directory')
    client = PersistentCodex(cwd, model=a.model, instructions=instructions(a.namespace), agent=True)
    terminal = TerminalInput() if a.ask is None else None
    last = None
    try:
        while True:
            try:
                text = a.ask if a.ask is not None else terminal.read(prompt(cwd))
            except (EOFError, KeyboardInterrupt):
                print(); break
            if not text.strip():
                if a.ask is not None: break
                continue
            if text.strip() in ('exit', 'quit', '/quit', '/exit', '!exit', '/shell exit', '/private exit'):
                break
            if text == '/help':
                print('Comandi Linux diretti: ls, pwd, cd, df -h… Nessuna chiamata al modello.\nParla liberamente per richieste Codex; /chat testo forza la chat.\n!comando, /shell comando o /private comando: esecuzione diretta.\n/pwd · /new: nuova conversazione, memoria conservata · /why: strumenti dell’ultimo turno.\n/verify DOMANDA: sessione nuova e sola memoria. /learn TESTO: memorizza.\n/reset-memory: reset con conferma · Ctrl+C: interrompe · exit: esce.\nIl copia-incolla multilinea resta un messaggio; Alt+Enter aggiunge una riga.')
                continue
            if text == '/pwd': print(cwd); continue
            if text == '/why': print(json.dumps(last, ensure_ascii=False, indent=2)); continue
            if text == '/new':
                client.close(); print('Nuova conversazione. Memoria conservata.'); continue
            try:
                forced_chat = text.startswith('/chat ')
                if forced_chat:
                    text = text[6:]
                if not forced_chat and (text.startswith('!') or text.startswith(('/shell ', '/private ')) or is_shell(text, cwd)):
                    command = text[1:] if text.startswith('!') else text.split(' ', 1)[1] if text.startswith(('/shell ', '/private ')) else text
                    target = run_shell(command, cwd)
                    if target is None: break
                    if target != cwd or command.lstrip().startswith(('export ', 'unset ')):
                        cwd = target
                        client.close()
                        client.root = cwd
                    if a.ask is not None:
                        break
                    continue
                if text == '/reset-memory':
                    from .codex_chat import CodexMemoryClient, reset_interactively
                    client.close()
                    reset_interactively(CodexMemoryClient(a.namespace))
                    continue
                if text.startswith(('/verify ', '/learn ', '/ask ')):
                    from .codex_chat import CodexMemoryClient, display
                    prefix, message = text.split(' ', 1)
                    memory = CodexMemoryClient(a.namespace, persistent=True)
                    try:
                        last = memory.turn(message, 'learn' if prefix == '/learn' else 'query', fresh=True)
                        display(last)
                    finally:
                        memory.transport.close()
                    continue
                view = Presentation()
                try:
                    events, fresh, timing = client.turn(text, timeout=a.timeout, on_event=view.event)
                    last = {'tools': view.calls, 'timing': timing, 'new_conversation': fresh}
                finally:
                    view.boundary()
            except KeyboardInterrupt:
                client.close()
                print('\nInterrotto. Le operazioni già completate restano valide; conversazione riavviata.')
            except Exception as exc:
                print('Errore: ' + safe_text(exc), flush=True)
            if a.ask is not None: break
    finally:
        client.close()


if __name__ == '__main__':
    main()
