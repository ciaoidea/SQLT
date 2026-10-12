"""Map pretrained Granite core and individual experts to native SQL vectors."""
import hashlib,json,time,struct,os
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from database import connect
from schema import initialize
REPO='ibm-granite/granite-3.1-3b-a800m-instruct'
COMMIT='a02780686e08a03fe0d2679a293b5c74a90efa89'


def put_parameter(c,name,tensor,layer=None,expert=None):
    array=tensor.float().numpy().astype('<f4',copy=False)
    shape=list(array.shape);dim=shape[-1] if len(shape)==2 else 0
    if len(shape) not in (1,2) or dim not in (0,512,1536):raise ValueError((name,shape))
    role='expert' if expert is not None else ('router' if '.router.' in name else 'core')
    with c.cursor() as cur:
        cur.execute('SELECT id,sha256,shape FROM moe_parameters WHERE name=%s',(name,))
        existing=cur.fetchone()
        if existing:
            if existing['sha256']!=hashlib.sha256(array.tobytes()).hexdigest() or json.loads(existing['shape'])!=shape:
                raise ValueError('Existing parameter differs from source: '+name)
            return existing['id']
        cur.execute('INSERT INTO moe_parameters(name,layer_no,expert_no,shape,dimensions,row_count,sha256,role) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)',(name,layer,expert,json.dumps(shape),dim,shape[0],hashlib.sha256(array.tobytes()).hexdigest(),role))
        pid=cur.lastrowid
        for start in range(0,shape[0],128):
            if dim:
                cur.executemany(f'INSERT INTO moe_vectors_{dim} VALUES(%s,%s,%s)',[(pid,i,array[i].tobytes()) for i in range(start,min(start+128,shape[0]))])
            else:
                cur.executemany('INSERT INTO moe_scalars VALUES(%s,%s,%s)',[(pid,i,float(array[i])) for i in range(start,min(start+128,shape[0]))])
    c.commit();return pid


@contextmanager
def open_source(folder):
    """Convert complete tensors while the official checkpoint finishes downloading."""
    from safetensors import safe_open
    import torch
    index_path=folder/'model.safetensors.index.json'
    if index_path.exists():
        from contextlib import ExitStack
        mapping=json.loads(index_path.read_text())['weight_map']
        with ExitStack() as stack:
            shards={}
            def downloaded_prefix(name):
                prefix=0;size=0
                progress=folder.parent/'download-progress.log'
                if progress.exists():
                    for line in progress.read_text().splitlines():
                        try:record=json.loads(line)
                        except ValueError:continue
                        if record.get('file')!=name:continue
                        if 'resumed_bytes' in record:prefix=record['resumed_bytes'];size=record['bytes']
                        elif 'completed_mib' in record:prefix=min(size,prefix+32*1024**2)
                return prefix
            def shard(name):
                if name in shards:return shards[name]
                path=folder/name
                if path.exists():
                    shards[name]=stack.enter_context(safe_open(path,framework='pt',device='cpu'))
                    return shards[name]
                partial=folder/(name+'.partial')
                while downloaded_prefix(name)<8:time.sleep(1)
                handle=stack.enter_context(partial.open('rb'))
                length=struct.unpack('<Q',handle.read(8))[0]
                while downloaded_prefix(name)<8+length:time.sleep(1)
                header=json.loads(handle.read(length));base=8+length
                class PartialShard:
                    def get_tensor(self,key):
                        entry=header[key];start,end=entry['data_offsets']
                        if entry['dtype']!='BF16':raise ValueError('Expected BF16 source')
                        announced=False
                        while not path.exists() and downloaded_prefix(name)<base+end:
                            if not announced:
                                print(json.dumps({'waiting_for_download':key,'shard':name}),flush=True);announced=True
                            time.sleep(2)
                        handle.seek(base+start);raw=bytearray(handle.read(end-start))
                        if len(raw)!=end-start:raise ValueError('Incomplete tensor read')
                        return torch.frombuffer(raw,dtype=torch.bfloat16).reshape(entry['shape'])
                shards[name]=PartialShard();return shards[name]
            class ShardedSource:
                def keys(self):return sorted(mapping)
                def get_tensor(self,name):return shard(mapping[name]).get_tensor(name)
            yield ShardedSource()
        return
    final=folder/'model.safetensors'
    if final.exists():
        with safe_open(final,framework='pt',device='cpu') as source:yield source
        return
    candidates=list((folder/'.cache/huggingface/download').glob('*.incomplete'))
    if len(candidates)!=1:raise ValueError('Expected one downloading checkpoint')
    with candidates[0].open('rb') as handle:
        length=struct.unpack('<Q',handle.read(8))[0]
        header=json.loads(handle.read(length));base=8+length
        class StreamingSource:
            def keys(self):return sorted(k for k in header if k!='__metadata__')
            def get_tensor(self,name):
                entry=header[name];start,end=entry['data_offsets']
                if entry['dtype']!='BF16':raise ValueError('Expected BF16 source')
                announced=False
                while os.fstat(handle.fileno()).st_size<base+end:
                    if final.exists() and final.stat().st_size<base+end:raise ValueError('Truncated source')
                    if not announced:
                        print(json.dumps({'waiting_for_download':name}),flush=True);announced=True
                    time.sleep(1)
                handle.seek(base+start);raw=bytearray(handle.read(end-start))
                if len(raw)!=end-start:raise ValueError('Incomplete source read')
                return torch.frombuffer(raw,dtype=torch.bfloat16).reshape(entry['shape'])
        yield StreamingSource()

def import_model(folder):
    from safetensors import safe_open
    folder=Path(folder);initialize()
    config=json.loads((folder/'config.json').read_text())
    if config['model_type']!='granitemoe':raise ValueError('Expected Granite MoE')
    with connect() as c,c.cursor() as cur:
        cur.execute('SELECT repo,source_commit,status FROM moe_model WHERE id=1')
        existing=cur.fetchone()
        if existing and (existing['repo']!=REPO or existing['source_commit']!=COMMIT or existing['status']!='importing'):
            raise ValueError('Refusing to overwrite an imported, verified, or different model')
        tokenizer=json.loads((folder/'tokenizer.json').read_text())
        vocabulary=tokenizer['model'].pop('vocab');merges=tokenizer['model'].pop('merges');added=tokenizer.pop('added_tokens')
        cur.execute('INSERT IGNORE INTO moe_model VALUES(1,%s,%s,%s,%s,%s,%s,%s,%s,1)',(REPO,COMMIT,json.dumps(config),(folder/'generation_config.json').read_text(),(folder/'tokenizer_config.json').read_text(),json.dumps(tokenizer),(folder/'README.md').read_text(),'importing'))
        extra={r['id']:r for r in added}
        rows=[(idx,piece,idx in extra,True,json.dumps(extra[idx]) if idx in extra else None) for piece,idx in vocabulary.items()]
        seen=set(vocabulary.values())
        rows.extend((r['id'],r['content'],True,False,json.dumps(r)) for r in added if r['id'] not in seen)
        cur.executemany('INSERT IGNORE INTO moe_tokens VALUES(%s,%s,%s,%s,%s)',rows)
        pairs=[m.split(' ') if isinstance(m,str) else m for m in merges]
        cur.executemany('INSERT IGNORE INTO moe_merges VALUES(%s,%s,%s)',[(i,*pair) for i,pair in enumerate(pairs)])
        c.commit()
        with open_source(folder) as f:
            names=list(f.keys())
            for index,name in enumerate(names):
                tensor=f.get_tensor(name)
                layer=int(name.split('.')[2]) if name.startswith('model.layers.') else None
                if tensor.ndim==3:
                    for expert in range(tensor.shape[0]):
                        put_parameter(c,name+f'.expert.{expert}',tensor[expert],layer,expert)
                else:put_parameter(c,name,tensor,layer)
                print(json.dumps({'tensor':index+1,'total':len(names),'name':name}),flush=True)
        for layer in range(config['num_hidden_layers']):
            prefix=f'model.layers.{layer}.block_sparse_moe.'
            cur.execute('SELECT id FROM moe_parameters WHERE name=%s',(prefix+'router.layer.weight',));router=cur.fetchone()['id']
            cur.execute('INSERT IGNORE INTO moe_layers VALUES(%s,%s,%s,%s)',(layer,config['num_local_experts'],config['num_experts_per_tok'],router))
            for expert in range(config['num_local_experts']):
                ids=[]
                for part in ('input_linear','output_linear'):
                    cur.execute('SELECT id FROM moe_parameters WHERE name=%s',(prefix+part+f'.weight.expert.{expert}',));ids.append(cur.fetchone()['id'])
                cur.execute('INSERT IGNORE INTO moe_experts VALUES(%s,%s,%s,%s)',(layer,expert,*ids))
        cur.execute("UPDATE moe_model SET status='imported' WHERE id=1")
        cur.execute("INSERT INTO moe_events(kind,metrics) VALUES('import',%s)",(json.dumps({'source':REPO,'commit':COMMIT}),));c.commit()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder');args=p.parse_args();import_model(args.folder)
