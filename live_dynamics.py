"""Opt-in, best-effort live memory telemetry. No database or persistent event log.

One bounded UNIX datagram collector per installation and namespace, leased only
by visible Live viewers. Missing/full collectors must never delay memory tools.
"""
from collections import deque
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import socket
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent
MAX_PACKET = 8192
LEASE = 4.0
GRAPH_ROLES = frozenset(('entity','person','company','customer','provider','agent','patient','recipient','branch','site','city','contact','collaborator','alias','project','product','service','team','department','place','società','entità'))

def graph_bindings(bindings):
    result=[]
    if not isinstance(bindings,list):return result
    for binding in bindings:
        if not isinstance(binding,dict) or binding.get('role') not in GRAPH_ROLES:continue
        value=binding.get('value')
        if not isinstance(value,str) or not value.strip() or len(value)>160:continue
        ident='v-'+hashlib.sha256(value.casefold().strip().encode()).hexdigest()[:24]
        result.append({'id':ident,'role':binding['role'],'value':value})
        if len(result)==4:break
    return result
_active_cache = {}

def socket_path(namespace, root=ROOT):
    key = hashlib.sha256((str(root.resolve())+'\0'+namespace).encode()).hexdigest()[:24]
    return Path('/tmp') / ('cortex-live-'+str(os.getuid())) / (key+'.sock')

def watching(namespace):
    """At most four cheap stat checks/second; no socket/serialization when idle."""
    now = time.monotonic()
    previous = _active_cache.get(namespace)
    if previous and now-previous[0] < .25:
        return previous[1]
    try:
        active = socket_path(namespace).is_socket()
    except OSError:
        active = False
    if len(_active_cache)>64:
        _active_cache.clear()
    _active_cache[namespace] = (now, active)
    return active

def emit(namespace, payload):
    try:
        data = json.dumps(payload, ensure_ascii=False, separators=(',',':')).encode()
        if len(data)>MAX_PACKET:
            return
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
            channel.setblocking(False)
            channel.sendto(data, str(socket_path(namespace)))
    except (OSError, ValueError, TypeError):
        pass

def notify_chat_turn(namespace='main', *, model=None, engine=None, effort=None):
    """An actual application user-turn boundary, never inferred from tool timing."""
    try:
        if watching(namespace):
            request_id=uuid.uuid4().hex[:16]
            emit(namespace,{'operation':'chat_turn','phase':'start',
                            'call':request_id,'producer':os.getpid(),
                            'origin':'client','records':[],'count':0,'duration_ms':0,
                            'engine':engine if engine in ('Codex app-server','Codex exec','Unsloth MCP') else None,
                            'effort':effort if effort in ('low','medium','high','xhigh') else None,
                            'model':model if isinstance(model,str) and re.fullmatch(r'[A-Za-z0-9_.:/-]{1,120}',model) else None})
            return request_id
    except Exception:
        pass

def request_focus(namespace, request_id, event_ids):
    if not re.fullmatch(r'[a-f0-9]{16,32}',request_id):
        raise ValueError('Use the graph request ID supplied for this user turn')
    if len(event_ids)>128 or any(type(i) is not int or i<1 for i in event_ids):
        raise ValueError('Select up to 128 positive event IDs')
    ids=list(dict.fromkeys(event_ids))
    active=watching(namespace)
    if active:
        emit(namespace,{'operation':'memory_focus','phase':'complete','request_id':request_id,
                        'call':uuid.uuid4().hex[:16],'producer':os.getpid(),'origin':'client',
                        'records':[],'count':len(ids),'selected_ids':ids,'duration_ms':0})
    return {'status':'focus_requested' if active else 'viewer_inactive','selected_event_ids':ids,
            'scope':'Model-selected supporting evidence for this request; not independent truth verification',
            'memory_changed':False}

def notify_memory_selection(namespace, candidates):
    """Report the actual memory-gate context, not a relevance or truth judgment."""
    try:
        if not watching(namespace):return
        records,count=snapshot({'events':candidates,'count':len(candidates)})
        ids=[e['event_id'] for e in candidates if type(e.get('event_id')) is int][:32]
        base={'operation':'memory_selection','phase':'complete','call':uuid.uuid4().hex[:16],
              'producer':os.getpid(),'origin':'gate','count':len(candidates),'duration_ms':0,'selected_ids':ids}
        for start in range(0,max(1,len(records)),2):
            emit(namespace,{**base,'records':records[start:start+2]})
    except Exception:
        pass

def notify_context_focus(namespace, request_id, candidates):
    """Retain the actual injected context, NOT model-selected answer evidence."""
    if not request_id or not watching(namespace):return
    notify_memory_selection(namespace,candidates)
    ids=list(dict.fromkeys(e['event_id'] for e in candidates
                          if type(e.get('event_id')) is int and e['event_id']>0))[:128]
    emit(namespace,{'operation':'context_focus','phase':'complete','request_id':request_id,
                    'call':uuid.uuid4().hex[:16],'producer':os.getpid(),'origin':'gate',
                    'records':[],'count':len(ids),'selected_ids':ids,'duration_ms':0})


def snapshot(result):
    if not isinstance(result, dict):
        return [], 0
    rows = result.get('events', result.get('candidates', []))
    if not isinstance(rows, list): rows=[]
    if isinstance(result.get('event'),dict): rows=[result['event']]
    records=[]
    for row in rows[:12]:
        if not isinstance(row,dict): continue
        ident=row.get('event_id')
        if type(ident) is not int or ident<1: continue
        records.append({'id':ident,'frame':str(row.get('frame','memory'))[:120],
                        'bindings':graph_bindings(row.get('roles',[])), 'negated':bool(row.get('negated',False))})
    ids=result.get('event_ids',[])
    if type(result.get('event_id')) is int: ids=[result['event_id']]
    for ident in ids[:12] if isinstance(ids,list) else []:
        if type(ident) is int and ident>0 and not any(r['id']==ident for r in records):
            records.append({'id':ident,'frame':'written memory'})
    count=result.get('total',result.get('count',result.get('stored',len(rows))))
    return records[:12], count if type(count) is int else len(rows)

def observed(function):
    signature=inspect.signature(function)
    @wraps(function)
    def wrapper(*args, **kwargs):
        namespace=kwargs.get('namespace','main')
        # MCP dispatches keyword arguments. Positional calls retain correct scope.
        if args:
            namespace=signature.bind_partial(*args,**kwargs).arguments.get('namespace','main')
        if not watching(namespace):
            return function(*args,**kwargs)
        started=time.monotonic()
        base={'call':uuid.uuid4().hex[:16], 'operation':function.__name__,
              'producer':os.getpid(),'origin':os.environ.get('CORTEX_LIVE_ORIGIN','client')}
        emit(namespace,{**base,'phase':'start','records':[],'count':0,'duration_ms':0})
        try:
            result=function(*args,**kwargs)
        except Exception:
            emit(namespace,{**base,'phase':'error','records':[],'count':0,
                            'duration_ms':round((time.monotonic()-started)*1000,2)})
            raise
        try:
            records,count=snapshot(result)
            for start in range(0,max(1,len(records)),2):
                emit(namespace,{**base,'phase':'complete','records':records[start:start+2],'count':count,
                                'duration_ms':round((time.monotonic()-started)*1000,2)})
        except Exception:
            pass  # Telemetry may not change a successful tool result.
        return result
    return wrapper

class LiveCollector:
    def __init__(self, namespace, root=ROOT, capacity=256, lease=LEASE):
        self.path=socket_path(namespace,root)
        self.namespace=namespace
        self.lease=lease
        self.events=deque(maxlen=capacity)
        self.clients={}
        self.lock=threading.RLock()
        self.channel=None
        self.thread=None
        self.seq=0
        self.generation=uuid.uuid4().hex
        self.overflow=0
        self.closed=False
        self.window_start=time.monotonic()
        self.window_packets=0
        self.rate_drops=0

    def poll(self, client, cursor=0, stop=False):
        if not re.fullmatch(r'[a-zA-Z0-9_-]{8,64}',client):
            raise ValueError('Invalid live viewer ID')
        with self.lock:
            if self.closed: raise ValueError('Live collector closed')
            now=time.monotonic()
            self.clients={k:v for k,v in self.clients.items() if v>now}
            if stop:
                self.clients.pop(client,None)
                if not self.clients: self._stop_locked()
                return {'events':[],'cursor':self.seq,'active':bool(self.clients),'generation':self.generation}
            if client not in self.clients and len(self.clients)>=32:
                raise ValueError('Live viewer limit reached')
            self.clients[client]=now+self.lease
            if self.channel is None: self._start_locked()
            return {'events':[e for e in self.events if e['seq']>cursor][-128:],
                    'cursor':self.seq,'generation':self.generation,'active':True,
                    'overflow':self.overflow,'rate_drops':self.rate_drops,
                    'namespace':self.namespace,'scope':'Observed memory tool activity; not private LLM reasoning',
                    'lease_seconds':self.lease}

    def _start_locked(self):
        self.path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        if self.path.parent.is_symlink() or self.path.parent.stat().st_uid!=os.getuid():
            raise ValueError('Unsafe live socket directory')
        os.chmod(self.path.parent,0o700)
        if self.path.exists():
            with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as probe:
                probe.settimeout(.01)
                try:
                    probe.connect(str(self.path))
                except ConnectionRefusedError:
                    self.path.unlink()
                else:
                    raise ValueError('Live collector already running for this namespace')
        channel=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM)
        channel.settimeout(.5)
        try:
            channel.bind(str(self.path))
            os.chmod(self.path,0o600)
        except BaseException:
            channel.close()
            raise
        self.channel=channel
        self.thread=threading.Thread(target=self._receive,args=(channel,),daemon=True,name='cortex-live')
        self.thread.start()

    def _stop_locked(self):
        if self.channel is not None:
            self.channel.close()
            self.channel=None
            self.path.unlink(missing_ok=True)
        self.events.clear()

    def _receive(self, channel):
        while True:
            with self.lock:
                if self.channel is not channel: return
                now=time.monotonic()
                self.clients={k:v for k,v in self.clients.items() if v>now}
                if not self.clients:
                    self._stop_locked();return
            try:
                data=channel.recv(MAX_PACKET+1)
                if len(data)>MAX_PACKET: continue
                event=json.loads(data)
                # Explicit allowlist: never transport input prompts, sources, SQL or secrets.
                if not isinstance(event,dict) or event.get('phase') not in ('start','complete','error'): continue
                if not re.fullmatch(r'[a-z_]{1,80}',str(event.get('operation',''))): continue
                records=event.get('records',[])
                if not isinstance(records,list) or len(records)>12: continue
                clean=[]
                for r in records:
                    if not isinstance(r,dict) or type(r.get('id')) is not int or r['id']<1: continue
                    clean.append({'id':r['id'],'frame':str(r.get('frame','memory'))[:120],
                                  'bindings':graph_bindings(r.get('bindings',[])), 'negated':bool(r.get('negated',False))})
                item={'operation':event['operation'],'phase':event['phase'],
                      'call':str(event.get('call',''))[:32],'producer':int(event.get('producer',0)),
                      'origin':event.get('origin') if event.get('origin') in ('viewer','gate') else 'client',
                      'duration_ms':max(0,min(3600000,float(event.get('duration_ms',0)))),
                      'count':max(0,int(event.get('count',0))),'records':clean}
                if item['operation']=='execution_trace':
                    label=event.get('label','')
                    if not isinstance(label,str) or not re.fullmatch(r'[A-Za-z0-9_ ./-]{1,96}',label):continue
                    if event.get('category') not in ('skill','command','connector'):continue
                    if event.get('status') not in ('running','completed','failed'):continue
                    item.update(label=label,category=event['category'],status=event['status'],request_id=str(event.get('request_id',''))[:32])
                if item['operation']=='chat_turn':
                    item['engine']=event.get('engine') if event.get('engine') in ('Codex app-server','Codex exec','Unsloth MCP') else None
                    item['effort']=event.get('effort') if event.get('effort') in ('low','medium','high','xhigh') else None
                    model=event.get('model')
                    item['model']=model if isinstance(model,str) and re.fullmatch(r'[A-Za-z0-9_.:/-]{1,120}',model) else None
                if item['operation'] in ('memory_selection','memory_focus','context_focus'):
                    ids=event.get('selected_ids',[])
                    item['selected_ids']=[i for i in ids[:128] if type(i) is int and i>0] if isinstance(ids,list) else []
                if item['operation'] in ('memory_focus','context_focus'):
                    item['request_id']=str(event.get('request_id',''))[:32]
                with self.lock:
                    if self.channel is not channel: return
                    now=time.monotonic()
                    if now-self.window_start>=1:
                        self.window_start=now;self.window_packets=0
                    self.window_packets+=1
                    if self.window_packets>100:
                        self.rate_drops+=1;continue
                    self.seq+=1
                    if len(self.events)==self.events.maxlen:self.overflow+=1
                    self.events.append({**item,'seq':self.seq,'time':time.time()})
            except socket.timeout:
                continue
            except (OSError,ValueError,TypeError,OverflowError):
                with self.lock:
                    if self.channel is not channel: return

    def close(self):
        with self.lock:
            self.closed=True
            self.clients.clear()
            self._stop_locked()


from contextlib import contextmanager

@contextmanager
def trace_skill_execution(namespace='main', *, llm_bypassed=False):
    """Instrument an actual executor; retrieval/skill discovery must not call this.

    llm_bypassed means the execution path does not invoke the language model.
    No skill names, arguments or results are sent to the visual channel.
    """
    active=watching(namespace)
    base={'operation':'skill_bypass' if llm_bypassed else 'skill_execution',
          'call':uuid.uuid4().hex[:16],'producer':os.getpid(),'origin':'client',
          'records':[],'count':0,'duration_ms':0}
    started=time.monotonic()
    if active:emit(namespace,{**base,'phase':'start'})
    try:
        yield
    except BaseException:
        if active:emit(namespace,{**base,'phase':'error','duration_ms':(time.monotonic()-started)*1000})
        raise
    else:
        if active:emit(namespace,{**base,'phase':'complete','duration_ms':(time.monotonic()-started)*1000})


def running_chat_models(root=ROOT):
    """Read launch metadata only, on demand. Never read prompts/auth/environment.

    A configured model identifier is not proof of a server-side resolved version.
    Restrict discovery to this Unix user; report unverified installation scope
    when /proc permissions prevent reading the working directory.
    """
    import ast
    default=None
    try:
        tree=ast.parse((root/'clients'/'shell_chat.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument' and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=='--model':
                default=next((k.value.value for k in node.keywords if k.arg=='default' and isinstance(k.value,ast.Constant)),None)
    except (OSError,SyntaxError):pass
    sessions=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            installation_verified=False
            try:
                if (proc/'cwd').resolve(strict=True)!=root.resolve():continue
                installation_verified=True
            except OSError:pass
            args=(proc/'cmdline').read_bytes().decode().split('\0')
            if 'clients.shell_chat' not in args:continue
            model=default
            for index,arg in enumerate(args):
                if arg=='--model' and index+1<len(args):model=args[index+1]
                elif arg.startswith('--model='):model=arg.split('=',1)[1]
            if not isinstance(model,str) or not re.fullmatch(r'[A-Za-z0-9_.:/-]{1,120}',model):continue
            engine=None
            try:
                for child in (proc/'task'/proc.name/'children').read_text().split():
                    child_args=(Path('/proc')/child/'cmdline').read_bytes().decode().split('\0')
                    if 'app-server' in child_args:engine='Codex app-server';break
            except OSError:pass
            sessions.append({'name':model,'engine':engine,'effort':None,'producer':int(proc.name),
                'source':('Running client launch configuration; ' if installation_verified else 'Same-user running client; installation scope unverified; ')+('explicit --model' if any(a=='--model' or a.startswith('--model=') for a in args) else 'default model from installed launcher')+'; not server-side attestation',
                'seen':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        except (OSError,ValueError,UnicodeError):continue
    return {'sessions':sessions[:16],'scope':'Same-user process metadata; installation checked when permissions allow'}


def trace_client_execution(namespace, request_id, item, phase):
    """Observe actual app-server tool lifecycle; never transmit args or outputs."""
    if not request_id or not watching(namespace) or phase not in ('start','complete'):return
    kind=str(item.get('type','')).replace('_','').lower()
    label=None;category=None
    if kind=='mcptoolcall':
        server=str(item.get('server',''));tool=str(item.get('tool',''))
        if server=='cortex':return  # memory has its own richer trace
        if re.fullmatch(r'[A-Za-z0-9_.-]{1,48}',server) and re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',tool):
            label=(server+'/'+tool)[:96];category='connector'
    elif kind=='commandexecution':
        category='command';label='Shell command'
        # Attribute only executable paths, never arbitrary mentions in output.
        import shlex
        try:
            tokens=shlex.split(str(item.get('command',''))[:16384])
            if len(tokens)>=3 and Path(tokens[0]).name in ('bash','sh','zsh') and tokens[1] in ('-lc','-c'):
                tokens=shlex.split(tokens[2])
            executable=next((t for t in tokens if t and '=' not in t), '')
            candidate=tokens[1] if Path(executable).name in ('python','python3','node') and len(tokens)>1 else executable
            match=re.search(r'/skills/([A-Za-z0-9_-]{1,64})/scripts/[A-Za-z0-9_./-]+$',candidate)
            if match:category='skill';label=match.group(1)
        except ValueError:pass
    if not label:return
    ident=str(item.get('id',''))
    if not ident:return
    failed=bool(item.get('error')) or (isinstance(item.get('result'),dict) and item['result'].get('isError') is True) or str(item.get('status','')).lower() in ('failed','error','declined','cancelled') or (type(item.get('exitCode',item.get('exit_code'))) is int and item.get('exitCode',item.get('exit_code'))!=0)
    status='running' if phase=='start' else 'failed' if failed else 'completed'
    emit(namespace,{'operation':'execution_trace','phase':'start' if phase=='start' else 'error' if failed else 'complete',
        'request_id':request_id,'call':hashlib.sha256(ident.encode()).hexdigest()[:24],
        'producer':os.getpid(),'origin':'client','records':[],'count':0,'duration_ms':0,
        'category':category,'label':label,'status':status})
