"""Long-lived native Codex transport; authentication and MCP stay in Codex."""
from .local_mcp import local_mcp_args
import atexit
import json
import queue
import subprocess
import threading
import time


class PersistentCodex:
    def __init__(self, root, *, model='gpt-6.1-sol', instructions=None, agent=False):
        self.root = root
        self.model = model
        self.instructions = instructions
        self.agent = agent
        self.process = None
        self.messages = queue.Queue()
        self.sequence = 0
        self.thread_id = None
        atexit.register(self.close)

    def close(self):
        process, self.process = self.process, None
        self.thread_id = None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdin.close()
            process.stdout.close()

    def _send(self, message):
        self.process.stdin.write(json.dumps(message) + '\n')
        self.process.stdin.flush()

    def _read(self, deadline):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError('Codex response timed out; completed DB writes remain committed')
        try:
            message = self.messages.get(timeout=remaining)
        except queue.Empty:
            raise TimeoutError('Codex response timed out; completed DB writes remain committed') from None
        if message is None:
            raise RuntimeError('Codex connection closed; no automatic write retry was made')
        if 'method' in message and 'id' in message:
            # Never grant an unexpected approval or impersonate a user response.
            self._send({'id': message['id'], 'error': {
                'code': -32601, 'message': 'Interactive requests are unsupported; approval policy is never'}})
        return message

    def _request(self, method, params, deadline):
        self.sequence += 1
        request_id = self.sequence
        self._send({'id': request_id, 'method': method, 'params': params})
        while True:
            message = self._read(deadline)
            if message.get('id') == request_id and 'method' not in message:
                if 'error' in message:
                    raise RuntimeError(str(message['error']))
                return message.get('result', {})

    def _start(self, deadline):
        self.messages = queue.Queue()
        self.process = subprocess.Popen(
            ['codex', 'app-server', *local_mcp_args(), '-c', 'approval_policy="never"'],
            cwd=self.root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        process, messages = self.process, self.messages

        def receive():
            try:
                for line in process.stdout:
                    try:
                        messages.put(json.loads(line))
                    except ValueError:
                        continue
            finally:
                messages.put(None)

        threading.Thread(target=receive, daemon=True).start()
        self._request('initialize', {'clientInfo': {
            'name': 'cortex_memory_chat', 'version': '1.1'}}, deadline)
        self._send({'method': 'initialized', 'params': {}})

    def turn(self, prompt, fresh=False, timeout=180, on_item=None, on_event=None):
        started = time.perf_counter()
        deadline = started + timeout
        events = []
        try:
            if self.process is None:
                self._start(deadline)
            thread_id = None if fresh else self.thread_id
            new_thread = thread_id is None
            if new_thread:
                settings = {
                    'model': self.model, 'cwd': str(self.root),
                    'ephemeral': True, 'approvalPolicy': 'never',
                    'config': {'model_reasoning_effort': 'low'}}
                if self.instructions is not None:
                    settings['developerInstructions'] = self.instructions
                if self.agent:
                    settings['sandbox'] = 'danger-full-access'
                result = self._request('thread/start', settings, deadline)
                thread_id = result['thread']['id']
                if not fresh:
                    self.thread_id = thread_id
            ready = time.perf_counter() - started
            result = self._request('turn/start', {
                'threadId': thread_id, 'input': [{'type': 'text', 'text': prompt}],
                'effort': 'low'}, deadline)
            turn_id = result['turn']['id']
            first_tool = None
            while True:
                message = self._read(deadline)
                params = message.get('params', {})
                if params.get('threadId') != thread_id:
                    continue
                method = message.get('method')
                if params.get('turnId') not in (None, turn_id):
                    continue
                if on_event:
                    on_event(method, params)
                if method == 'item/started' and params['item']['type'] == 'mcpToolCall':
                    if first_tool is None:
                        first_tool = time.perf_counter() - started
                if method == 'item/completed':
                    item = dict(params['item'])
                    item['type'] = {'agentMessage': 'agent_message',
                                    'mcpToolCall': 'mcp_tool_call',
                                    'commandExecution': 'command_execution'}.get(item['type'], item['type'])
                    if item.get('result'):
                        item['result']['structured_content'] = item['result'].get('structuredContent')
                    events.append({'type': 'item.completed', 'item': item})
                    if on_item:on_item(item)
                if method == 'turn/completed' and params['turn']['id'] == turn_id:
                    if params['turn']['status'] != 'completed':
                        raise RuntimeError(str(params['turn'].get('error') or params['turn']['status']))
                    break
            timing = {'client_ready_seconds': ready, 'first_tool_seconds': first_tool,
                      'mcp_seconds': sum(e['item'].get('durationMs') or 0 for e in events
                                         if e['item']['type'] == 'mcp_tool_call') / 1000}
            if fresh:
                self._request('thread/unsubscribe', {'threadId': thread_id}, deadline)
            return events, new_thread, timing
        except BaseException:
            # Do not leave an interrupted turn running and consuming resources.
            self.close()
            raise
