"""Request-local progress for real query construction; final result is unchanged."""
from contextvars import ContextVar
import json
from queue import Queue, Full, Empty
import threading
import time

_sink=ContextVar('cortex_query_flow',default=None)

def progress(stage, elements=None, **details):
    sink=_sink.get()
    if sink is not None:
        sink(stage,elements,details)

class Flow:
    """Bounded visual queue; slow/disconnected viewers never block query workers."""
    def __init__(self):
        self.queue=Queue(maxsize=32)
        self.done=threading.Event()
        self.closed=False
        self.result=None
        self.error=None
        self.dropped=0
        self.seq=0
        self.started=time.monotonic()

    def emit(self,stage,elements,details):
        if self.closed:return
        # Intermediate snapshots omit sources and full object records. Final
        # response retains the existing, authorized static graph data.
        clean=[]
        for element in elements or []:
            data=dict(element['data'])
            data.pop('record',None)
            if isinstance(data.get('event'),dict):
                e=data['event']
                data['event']={k:e[k] for k in ('event_id','frame','negated','roles') if k in e}
            clean.append({'data':data})
        if not clean and self.queue.qsize()>=8:
            self.dropped+=1;return
        self.seq+=1
        event={'type':'progress','seq':self.seq,'stage':stage,'elements':clean,
               'elapsed_ms':round((time.monotonic()-self.started)*1000,2),**details}
        try:
            if len(json.dumps(event,ensure_ascii=False).encode())>65536:
                self.dropped+=1;return
            self.queue.put_nowait(event)
        except Full:
            # Preserve the newest actual graph mutations, not a queue full of
            # classification notices. Each builder patch contains its endpoints.
            self.dropped+=1
            try:self.queue.get_nowait()
            except Empty:pass
            try:self.queue.put_nowait(event)
            except Full:self.dropped+=1
        except (TypeError,ValueError):self.dropped+=1

    def execute(self,operation):
        token=_sink.set(self.emit)
        try:self.result=operation()
        except Exception as e:self.error=type(e).__name__
        finally:_sink.reset(token);self.done.set()

    def frames(self):
        while not self.done.is_set() or not self.queue.empty():
            try:yield self.queue.get(timeout=.2)
            except Empty:
                if self.closed:return
                # Keeps authenticated proxies alive without invented graph activity.
                yield {'type':'heartbeat'}
        if self.error:
            yield {'type':'error','error':'Query failed: '+self.error}
        else:
            yield {'type':'result','data':self.result,'dropped_progress':self.dropped}
