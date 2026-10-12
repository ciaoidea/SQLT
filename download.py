"""Fetch pinned official files with validated ranged transfers and SHA-256."""
import concurrent.futures,hashlib,json,os,time,errno
from pathlib import Path
import requests
from store import REPO,COMMIT
ROOT=Path(__file__).parent/'.conversion';ROOT.mkdir(exist_ok=True)
BASE=f'https://huggingface.co/{REPO}/resolve/{COMMIT}/'
info=requests.get(f'https://huggingface.co/api/models/{REPO}/revision/{COMMIT}?blobs=true',timeout=60);info.raise_for_status()
files=[f for f in info.json()['siblings'] if f['rfilename'].endswith(('.json','.safetensors')) or f['rfilename']=='README.md']
for f in files:
    name=f['rfilename']
    if name.endswith('.safetensors'):continue
    r=requests.get(BASE+name,timeout=60);r.raise_for_status();(ROOT/name).write_bytes(r.content)
for f in files:
    name=f['rfilename']
    if not name.endswith('.safetensors'):continue
    target=ROOT/name;size=f['size'];sha=f['lfs']['sha256']
    if target.exists():
        with target.open('rb') as h:actual=hashlib.file_digest(h,'sha256').hexdigest()
        if actual==sha:continue
        raise ValueError('Existing checkpoint checksum mismatch')
    partial=ROOT/(name+'.partial')
    candidates=list((ROOT/'.cache/huggingface/download').glob('*.'+sha+'.incomplete'))
    if not partial.exists() and candidates:candidates[0].rename(partial)
    fd=os.open(partial,os.O_CREAT|os.O_RDWR,0o600)
    if os.fstat(fd).st_size<size:os.ftruncate(fd,size)
    def prefix_ready():
        return min(size,os.lseek(fd,0,os.SEEK_HOLE))
    missing=[];offset=0
    while offset<size:
        try:data=min(size,os.lseek(fd,offset,os.SEEK_DATA))
        except OSError as exc:
            if exc.errno!=errno.ENXIO:raise
            data=size
        if data>offset:missing.append((offset,data))
        if data==size:break
        offset=os.lseek(fd,data,os.SEEK_HOLE)
    chunk=8*1024**2
    tasks=[(start,min(start+chunk,end)-1) for low,end in missing for start in range(low,end,chunk)]
    prefix=prefix_ready()
    print(json.dumps({'file':name,'bytes':size,'resumed_bytes':prefix,'missing_mib':round(sum(b-a for a,b in missing)/1024**2),'chunks':len(tasks)}),flush=True)
    def fetch(bounds):
        start,end=bounds
        for attempt in range(5):
            try:
                with requests.get(BASE+name+f'?download=true&part={start}',headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(30,120)) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise ValueError('Unexpected HTTP range')
                    offset=start
                    for data in r.iter_content(1024**2):
                        if offset+len(data)>end+1:raise ValueError('Oversized HTTP range')
                        written=os.pwrite(fd,data,offset)
                        if written!=len(data):raise OSError('Short write')
                        offset+=len(data)
                    if offset!=end+1:raise ValueError('Incomplete HTTP range')
                return end-start+1
            except Exception:
                if attempt==4:raise
                time.sleep(attempt+1)
    started=time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(fetch,bounds) for bounds in tasks]
            for future in concurrent.futures.as_completed(futures):
                future.result()
                print(json.dumps({'file':name,'bytes':size,'resumed_bytes':prefix_ready(),'seconds':round(time.perf_counter()-started,1)}),flush=True)
    finally:os.close(fd)
    with partial.open('rb') as h:actual=hashlib.file_digest(h,'sha256').hexdigest()
    if actual!=sha:
        partial.unlink()
        raise ValueError('Downloaded checkpoint checksum mismatch; partial file removed. Rerun install.sh: '+name)
    partial.rename(target)
    print(json.dumps({'file':name,'sha256_verified':True}),flush=True)
print(json.dumps({'download_complete':True}),flush=True)
