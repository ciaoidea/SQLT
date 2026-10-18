"""One CPU encoder for local MCP clients; no SQL, public port or text log."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import base64
from collections import OrderedDict
import hashlib
import json
import signal
import socket
import struct
import time
from encoder_runtime import SOCKET,MAX_BYTES
import semantic_memory as semantic

IDLE_SECONDS=300
CACHE_LIMIT=64


def run():
    cache=OrderedDict()
    running=True
    def stop(*_):
        nonlocal running
        running=False
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    server.bind(str(SOCKET));os.chmod(SOCKET,0o600)
    inode=SOCKET.stat().st_ino
    server.listen(8);server.settimeout(1)
    last_request=time.monotonic()
    try:
        while running and time.monotonic()-last_request<IDLE_SECONDS:
            try:connection,_=server.accept()
            except socket.timeout:continue
            with connection:
                connection.settimeout(30)
                # Directory and socket permissions already isolate access;
                # peer credentials additionally require the same Unix user.
                _,uid,_=struct.unpack('3i',connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
                if uid!=os.getuid():continue
                try:
                    with connection.makefile('rb') as reader:raw=reader.readline(MAX_BYTES+1)
                    if len(raw)>MAX_BYTES or not raw.endswith(b'\n'):raise ValueError('Invalid request size')
                    message=json.loads(raw);operation=message.get('operation')
                    if operation not in ('identity','query','passage','passages','status'):raise ValueError('Unknown operation')
                    started=time.perf_counter()
                    response={}
                    if operation=='query':
                        text=message.get('text');semantic.chunks(text)
                        key=hashlib.sha256(text.encode()).digest()
                        hit=key in cache
                        vector=cache.pop(key) if hit else semantic._local_embed_query(text)
                        cache[key]=vector
                        while len(cache)>CACHE_LIMIT:cache.popitem(last=False)
                        response=dict(vector=base64.b64encode(vector).decode(),cache_hit=hit)
                    elif operation=='passage':
                        # Do not retain source passages or their plaintext in a cache.
                        encoded=semantic._local_embed_passage(message.get('text'))
                        response=dict(vectors=[base64.b64encode(v).decode() for _,v in encoded])
                    elif operation=='passages':
                        texts=message.get('text')
                        if not isinstance(texts,list):raise ValueError('Expected a text batch')
                        encoded=semantic._local_embed_passages(texts)
                        response=dict(vectors=[[base64.b64encode(v).decode() for _,v in group] for group in encoded])
                    encoder_id,config=semantic._local_identity()
                    rss=int(next(line for line in open('/proc/self/status') if line.startswith('VmRSS:')).split()[1])/1024
                    response.update(encoder_id=encoder_id,config=config,pid=os.getpid(),rss_mib=rss,
                                    compute_seconds=time.perf_counter()-started,cached_queries=len(cache))
                    data=json.dumps(response).encode()+b'\n'
                    if len(data)>MAX_BYTES:raise ValueError('Encoder response budget exceeded')
                except Exception as error:
                    # Error classes, not user text, cross the diagnostics boundary.
                    data=json.dumps({'error':type(error).__name__}).encode()+b'\n'
                try:connection.sendall(data)
                except (BrokenPipeError,ConnectionResetError):pass
                last_request=time.monotonic()
    finally:
        server.close()
        if SOCKET.exists() and SOCKET.stat().st_ino==inode:SOCKET.unlink()


if __name__=='__main__':run()
