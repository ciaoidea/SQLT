"""Private, bounded Unix-socket client for one shared local encoder process."""
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'.run'
SOCKET=RUNTIME/'encoder.sock'
MAX_BYTES=131072


def _connect():
    stream=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    stream.settimeout(30)
    try:
        stream.connect(str(SOCKET))
        return stream
    except BaseException:
        stream.close()
        raise


def _connection():
    try:return _connect()
    except (FileNotFoundError,ConnectionRefusedError):pass
    RUNTIME.mkdir(mode=0o700,exist_ok=True)
    os.chmod(RUNTIME,0o700)
    descriptor=os.open(RUNTIME/'encoder.lock',os.O_CREAT|os.O_RDWR,0o600)
    with os.fdopen(descriptor,'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:return _connect()
        except (FileNotFoundError,ConnectionRefusedError):pass
        SOCKET.unlink(missing_ok=True)
        process=subprocess.Popen([sys.executable,str(ROOT/'encoder_worker.py')],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,start_new_session=True)
        threading.Thread(target=process.wait,daemon=True).start()
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('Local encoder worker failed to start')
            try:return _connect()
            except (FileNotFoundError,ConnectionRefusedError):time.sleep(.02)
        raise TimeoutError('Local encoder worker did not become ready')


def request(operation,text=None):
    payload=json.dumps(dict(operation=operation,text=text),ensure_ascii=False).encode()+b'\n'
    if len(payload)>MAX_BYTES:raise ValueError('Encoder request budget exceeded')
    for attempt in range(2):
        try:
            with _connection() as stream:
                stream.sendall(payload)
                with stream.makefile('rb') as reader:raw=reader.readline(MAX_BYTES+1)
            if not raw:raise ConnectionResetError('Encoder closed an idle connection')
            break
        except (BrokenPipeError,ConnectionResetError):
            # Pure encoding can safely retry an idle-exit race, not a DB write.
            if attempt:raise
            time.sleep(.02)
    if not raw.endswith(b'\n') or len(raw)>MAX_BYTES:
        raise RuntimeError('Invalid or oversized encoder response')
    response=json.loads(raw)
    if 'error' in response:raise RuntimeError('Local semantic encoder: '+response['error'])
    return response
