"""Pretrained MoE inference with SQL-selected experts and a bounded LRU cache.

Core parameters and routers are resident; routing loads experts from MariaDB
into bounded RAM and device caches. No checkpoint is read at runtime.
"""
import argparse,hashlib,json,os,time,warnings
import gc,subprocess,sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import numpy as np
import pymysql
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from transformers import AutoConfig,AutoModelForCausalLM,PreTrainedTokenizerFast
from database import connect
from terminal_ui import TerminalChat
from model_tools import SYSTEM_INSTRUCTIONS,generate_with_tools,decode_protocol_call,decode_decision
from resource_policy import memory_budgets,preferred_device,calibrate

torch.set_num_threads(int(os.environ.get('MODEL_THREADS','2')))

def execution_device(requested):
    if requested=='cpu':return torch.device('cpu')
    try:
        if torch.cuda.is_available():
            torch.empty(1,device='cuda')
            return torch.device('cuda')
    except RuntimeError as exc:
        warnings.warn(f'CUDA unavailable: {exc}. Using CPU.')
    if requested=='cuda':warnings.warn('CUDA unavailable. Using CPU.')
    return torch.device('cpu')

def load_parameter(connection,metadata):
    shape=json.loads(metadata['shape']);dim=metadata['dimensions']
    if metadata['role']=='expert':
        if dim not in (512,1536):raise ValueError('Invalid expert dimension')
        # Aggregate only this addressed expert matrix for transport. Storage
        # remains one native VECTOR per neuron; nothing is stored as a blob.
        with connection.cursor() as cur:
            cur.execute(f"SELECT GROUP_CONCAT(CAST(coefficients AS BINARY) ORDER BY unit_no SEPARATOR '') AS coefficients,COUNT(*) AS row_count FROM moe_vectors_{dim} WHERE parameter_id=%s",(metadata['id'],))
            row=cur.fetchone()
        raw=row['coefficients']
        if row['row_count']!=shape[0] or raw is None or len(raw)!=int(np.prod(shape))*4 or hashlib.sha256(raw).hexdigest()!=metadata['sha256']:
            raise ValueError('Expert parameter integrity check failed: '+metadata['name'])
        return torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy().reshape(shape))
    result=torch.empty(shape,dtype=torch.float32);digest=hashlib.sha256();offset=0
    table=f'moe_vectors_{dim}' if dim else 'moe_scalars'
    if dim not in (0,512,1536):raise ValueError('Invalid parameter dimension')
    field='coefficients' if dim else 'coefficient'
    with connection.cursor(pymysql.cursors.SSDictCursor) as cur:
        cur.execute(f'SELECT unit_no,{field} FROM {table} WHERE parameter_id=%s ORDER BY unit_no',(metadata['id'],))
        while rows:=cur.fetchmany(128):
            if [r['unit_no'] for r in rows]!=list(range(offset,offset+len(rows))):raise ValueError('Missing parameter rows')
            raw=b''.join(r[field] for r in rows) if dim else np.asarray([r[field] for r in rows],dtype='<f4').tobytes()
            digest.update(raw)
            result[offset:offset+len(rows)].copy_(torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy().reshape((-1,dim) if dim else (-1,))))
            offset+=len(rows)
    if offset!=shape[0] or digest.hexdigest()!=metadata['sha256']:raise ValueError('Parameter checksum mismatch: '+metadata['name'])
    return result

def load_tokenizer(connection,meta):
    structure=json.loads(meta['tokenizer_structure'])
    with connection.cursor() as cur:
        cur.execute('SELECT * FROM moe_tokens ORDER BY id');tokens=cur.fetchall()
        cur.execute('SELECT * FROM moe_merges ORDER BY rank_no');merges=cur.fetchall()
    structure['model']['vocab']={r['piece']:r['id'] for r in tokens if r['in_vocabulary']}
    structure['model']['merges']=[[r['first_piece'],r['second_piece']] for r in merges]
    structure['added_tokens']=[json.loads(r['properties']) for r in tokens if r['added']]
    backend=Tokenizer.from_str(json.dumps(structure,ensure_ascii=False))
    cfg=json.loads(meta['tokenizer_config'])
    def content(v):return v['content'] if isinstance(v,dict) else v
    kwargs={k:content(cfg[k]) for k in ('bos_token','eos_token','pad_token','unk_token') if cfg.get(k) is not None}
    return PreTrainedTokenizerFast(tokenizer_object=backend,chat_template=cfg['chat_template'],**kwargs)

class Int8Projection:
    """Optional approximate CPU execution; authoritative SQL weights stay FP32."""
    def __init__(self,weight):
        scales=weight.abs().amax(dim=1).clamp_min(1e-12)/127
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message=r'torch\.quantize_per_tensor, torch\.quantize_per_channel.*deprecated.*',category=UserWarning)
            quantized=torch.quantize_per_channel(weight,scales,torch.zeros(weight.shape[0],dtype=torch.int64),0,torch.qint8)
        # Pack the real coefficients once, avoiding the temporary full-size
        # weights packed by the dynamic Linear constructor.
        self.packed=torch.ops.quantized.linear_prepack(quantized,None)
        self.bytes=weight.numel()+weight.shape[0]*16
    def __call__(self,inputs):
        return torch.ops.quantized.linear_dynamic(inputs,self.packed,reduce_range=True).to(inputs.dtype)

def stored_bytes(value):
    return value.bytes if isinstance(value,Int8Projection) else value.numel()*value.element_size()

def available_ram_mib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])//1024
    return 0

class ExpertCache:
    def __init__(self,connection,parameters,experts,limit_mib,quantized=False,device='cpu',host_limit_mib=0,
                 db_workers=0,pin_host=False,batch_experts=True):
        if limit_mib<12:raise ValueError('Expert cache must be at least 12 MiB')
        if host_limit_mib<0:raise ValueError('RAM cache size must be nonnegative')
        self.connection=connection;self.parameters=parameters;self.experts=experts;self.quantized=quantized
        self.device=torch.device(device)
        if quantized:torch.backends.quantized.engine='fbgemm'
        self.limit=int(limit_mib*1024**2);self.items=OrderedDict();self.bytes=0
        self.layer_count=len({layer for layer,_ in experts});self.layer_limit=self.limit//self.layer_count
        self.layer_bytes={layer:0 for layer,_ in experts}
        self.host_limit=int(host_limit_mib*1024**2);self.host_items=OrderedDict();self.host_bytes=0
        self.host_hits=0;self.database_loads=0;self.progress_callback=None
        self.pin_host=pin_host;self.batch_experts=batch_experts
        self.executor=ThreadPoolExecutor(max_workers=db_workers) if db_workers else None
        self.futures={};self.worker_local=threading.local();self.worker_connections=[];self.worker_lock=threading.Lock()
        self.hits=0;self.misses=0;self.transferred_bytes=0;self.load_seconds=0.;self.router_seconds=0.;self.read_seconds=0.;self.prepare_seconds=0.;self.routes=[]
    def prepare(self,layer,selected):
        # Protect cached experts needed later in the same routed layer.
        for expert in selected:
            key=(layer,expert)
            if key in self.items:self.items.move_to_end(key)
            elif self.executor and key not in self.host_items and key not in self.futures:
                self.futures[key]=self.executor.submit(self.read_expert,key,True)
    def read_expert(self,key,worker=False):
        connection=self.connection
        if worker:
            if not hasattr(self.worker_local,'connection'):
                self.worker_local.connection=connect()
                with self.worker_lock:self.worker_connections.append(self.worker_local.connection)
            connection=self.worker_local.connection
        binding=self.experts[key]
        metadata=[self.parameters[binding[k]] for k in ('input_parameter','output_parameter')]
        tensors=[];wire_bytes=0;read_seconds=0.;prepare_seconds=0.
        for meta in metadata:
            started=time.perf_counter()
            tensor=load_parameter(connection,meta)
            read_seconds+=time.perf_counter()-started
            wire_bytes+=tensor.numel()*tensor.element_size()
            started=time.perf_counter()
            if self.quantized:tensors.append(Int8Projection(tensor))
            else:
                compact=tensor.to(torch.bfloat16)
                tensors.append(compact if torch.equal(tensor,compact.float()) else tensor)
            prepare_seconds+=time.perf_counter()-started
        return tuple(tensors),wire_bytes,read_seconds,prepare_seconds
    def cancel_prefetch(self):
        for future in self.futures.values():future.cancel()
        self.futures.clear()
    def close(self):
        self.cancel_prefetch()
        if self.executor:self.executor.shutdown(wait=True,cancel_futures=True)
        for connection in self.worker_connections:connection.close()
        self.worker_connections.clear()
        self.items.clear();self.host_items.clear()
    def trim_host(self,limit_mib):
        self.host_limit=min(self.host_limit,int(max(0,limit_mib)*1024**2))
        while self.host_bytes>self.host_limit:
            _,old=self.host_items.popitem(last=False)
            self.host_bytes-=sum(stored_bytes(t) for t in old)
    def trim_execution(self,limit_mib):
        self.limit=min(self.limit,int(limit_mib*1024**2));self.layer_limit=self.limit//self.layer_count
        for key in list(self.items):
            layer=key[0]
            if self.layer_bytes[layer]>self.layer_limit:
                old=self.items.pop(key);released=sum(stored_bytes(t) for t in old)
                self.bytes-=released;self.layer_bytes[layer]-=released
    def get(self,layer,expert):
        if self.progress_callback:self.progress_callback()
        key=(layer,expert)
        if key in self.items:
            self.hits+=1;self.items.move_to_end(key);return self.items[key]
        self.misses+=1;start=time.perf_counter()
        if key in self.host_items:
            self.host_hits+=1;self.host_items.move_to_end(key);tensors=self.host_items[key]
        else:
            future=self.futures.pop(key,None)
            tensors,wire_bytes,read_seconds,prepare_seconds=future.result() if future else self.read_expert(key)
            self.transferred_bytes+=wire_bytes;self.read_seconds+=read_seconds
            self.prepare_seconds+=prepare_seconds;self.database_loads+=1
            host_required=sum(stored_bytes(t) for t in tensors)
            if self.host_limit and host_required<=self.host_limit:
                while self.host_bytes+host_required>self.host_limit:
                    _,old=self.host_items.popitem(last=False)
                    self.host_bytes-=sum(stored_bytes(t) for t in old)
                if self.pin_host:
                    try:tensors=tuple(t.pin_memory() for t in tensors)
                    except RuntimeError:
                        self.pin_host=False
                self.host_items[key]=tensors;self.host_bytes+=host_required
        required=sum(stored_bytes(t) for t in tensors)
        if required>self.layer_limit:raise ValueError('Expert exceeds per-layer cache budget; increase --cache-mib')
        while self.layer_bytes[layer]+required>self.layer_limit:
            victim=next(k for k in self.items if k[0]==layer)
            old=self.items.pop(victim);released=sum(stored_bytes(t) for t in old)
            self.bytes-=released;self.layer_bytes[layer]-=released
            del old
        prepare_started=time.perf_counter()
        if self.device.type=='cuda':tensors=tuple(t.to(self.device,non_blocking=t.is_pinned()) for t in tensors)
        self.prepare_seconds+=time.perf_counter()-prepare_started
        self.items[key]=tensors;self.bytes+=required;self.layer_bytes[layer]+=required
        self.load_seconds+=time.perf_counter()-start
        return tensors
    def stats(self):
        return dict(cache_policy=('per-layer LRU, approximate INT8' if self.quantized else 'per-layer LRU, lossless BF16 when representable'),
            host_cache_limit_mib=self.host_limit/1024**2,host_cache_resident_mib=self.host_bytes/1024**2,
            host_cached_experts=len(self.host_items),host_cache_hits=self.host_hits,database_expert_loads=self.database_loads,
            database_workers=self.executor._max_workers if self.executor else 0,pinned_host_memory=self.pin_host,
            cache_limit_mib=self.limit/1024**2,cache_resident_mib=self.bytes/1024**2,cached_experts=len(self.items),hits=self.hits,misses=self.misses,expert_transfer_mib=self.transferred_bytes/1024**2,expert_load_seconds=self.load_seconds,router_seconds=self.router_seconds,expert_read_seconds=self.read_seconds,expert_prepare_seconds=self.prepare_seconds,last_routes=[dict(layer=r['layer'],experts=list(r['experts']),tokens=r['tokens']) for r in self.routes])

class SQLExperts(nn.Module):
    def __init__(self,layer,router,cache,activation):
        super().__init__();self.layer=layer;self.router=router;self.cache=cache;self.activation=activation
    def forward(self,layer_input):
        batch,length,width=layer_input.shape;flat=layer_input.reshape(-1,width)
        route_started=time.perf_counter()
        _,indices,gates,sizes,logits=self.router(flat)
        self.cache.router_seconds+=time.perf_counter()-route_started
        out=torch.zeros_like(flat)
        selected=[expert for expert,size in enumerate(sizes) if size]
        self.cache.prepare(self.layer,selected)
        if flat.is_cuda and len(flat)==1 and self.cache.batch_experts:
            matrices=[self.cache.get(self.layer,expert) for expert in selected]
            win=torch.stack([weights[0] for weights in matrices]).float()
            wout=torch.stack([weights[1] for weights in matrices]).float()
            projected=torch.bmm(win,flat[0].view(1,-1,1).expand(len(selected),-1,-1)).squeeze(-1)
            a,b=projected.chunk(2,dim=-1)
            values=torch.bmm(wout,(self.activation(a)*b).unsqueeze(-1)).squeeze(-1)
            out.index_add_(0,indices,values*gates[:,None])
        else:
            offset=0
            for expert,size in enumerate(sizes):
                if not size:continue
                tokens=indices[offset:offset+size];weights=gates[offset:offset+size]
                win,wout=self.cache.get(self.layer,expert)
                if isinstance(win,Int8Projection):
                    projected=win(flat[tokens]);a,b=projected.chunk(2,dim=-1)
                    values=wout(self.activation(a)*b)
                else:
                    win,wout=win.float(),wout.float()
                    projected=F.linear(flat[tokens],win);a,b=projected.chunk(2,dim=-1)
                    values=F.linear(self.activation(a)*b,wout)
                out.index_add_(0,tokens,values*weights[:,None]);offset+=size
        self.cache.routes.append({'layer':self.layer,'experts':selected,'tokens':len(flat)})
        self.cache.routes=self.cache.routes[-self.cache.layer_count:]
        return out.view(batch,length,width),logits

class DatabaseMoE:
    def __init__(self,cache_mib=1536,allow_unverified=False,quantized_experts=False,device='cpu',host_cache_mib=6144,
                 db_workers=2,batch_experts=True,pin_host=True):
        if host_cache_mib<0:raise ValueError('RAM cache size must be nonnegative')
        if db_workers not in (0,1,2,4):raise ValueError('Database workers must be 0, 1, 2 or 4')
        self.device=execution_device(device)
        started=time.perf_counter();self.connection=connect()
        with self.connection.cursor() as cur:
            cur.execute('SELECT * FROM moe_model WHERE id=1');self.metadata=cur.fetchone()
            if not self.metadata or self.metadata['status'] not in (('imported','ready') if allow_unverified else ('ready',)):
                raise ValueError('MoE database is not verified and ready')
            cur.execute('SELECT * FROM moe_parameters');rows=cur.fetchall()
            self.parameters={r['id']:r for r in rows}
            cur.execute('SELECT * FROM moe_experts');self.experts={(r['layer_no'],r['expert_no']):r for r in cur.fetchall()}
            cur.execute('SELECT * FROM moe_layers ORDER BY layer_no');layers=cur.fetchall()
        cfg=json.loads(self.metadata['config']);kind=cfg.pop('model_type')
        config=AutoConfig.for_model(kind,**cfg);config._attn_implementation='sdpa'
        core_size=sum(int(np.prod(json.loads(r['shape'])))*4 for r in rows if r['role']!='expert')
        expert_size=sum(int(np.prod(json.loads(r['shape'])))*2 for r in rows if r['role']=='expert')
        ram=available_ram_mib();free=torch.cuda.mem_get_info()[0]/1024**2 if self.device.type=='cuda' else 0
        budgets=memory_budgets(core_size/1024**2,expert_size/1024**2,cache_mib,host_cache_mib,ram,free)
        self.selection=dict(requested_device=device,available_ram_mib=ram,free_gpu_mib=free,**budgets)
        timings=None
        if device=='auto' and self.device.type=='cuda' and budgets['gpu_cache_mib']>=384:
            try:timings=calibrate(Int8Projection,quantized_experts,budgets['gpu_cache_mib'],expert_size/1024**2)
            except RuntimeError as exc:
                warnings.warn(f'CUDA calibration failed: {exc}. Using CPU.')
                budgets['gpu_cache_mib']=0
            finally:torch.cuda.empty_cache()
        chosen,reason=preferred_device(device if self.device.type=='cuda' else 'cpu',budgets,timings)
        if device!='cpu' and self.device.type=='cpu':reason='CUDA unavailable'
        self.device=torch.device(chosen);self.selection.update(reason=reason,calibration=timings)
        cache_mib=budgets['gpu_cache_mib'] if chosen=='cuda' else budgets['cpu_cache_mib']
        if cache_mib<96:
            self.connection.close()
            raise RuntimeError('Insufficient available RAM for SQLT; close other applications and retry')
        # Meta allocation creates architecture only: not one expert weight is allocated.
        with torch.device('meta'):self.model=AutoModelForCausalLM.from_config(config)
        host_budget=budgets['host_cache_mib'] if chosen=='cuda' else 0
        self.cache=ExpertCache(self.connection,self.parameters,self.experts,cache_mib,
            quantized=quantized_experts and self.device.type=='cpu',device=self.device,host_limit_mib=host_budget,
            db_workers=db_workers,pin_host=pin_host and chosen=='cuda',batch_experts=batch_experts)
        for layer in layers:
            number=layer['layer_no'];block=self.model.model.layers[number].block_sparse_moe
            if layer['expert_count']!=block.router.num_experts or layer['active_experts']!=block.router.top_k:raise ValueError('Router metadata mismatch')
            self.model.model.layers[number].block_sparse_moe=SQLExperts(number,block.router,self.cache,block.activation)
        core_bytes=0
        for meta in rows:
            if meta['role']=='expert':continue
            weight=load_parameter(self.connection,meta);core_bytes+=weight.numel()*4
            parent,attribute=meta['name'].rsplit('.',1)
            setattr(self.model.get_submodule(parent),attribute,nn.Parameter(weight,requires_grad=False))
        self.model.tie_weights()
        # The rotary buffer was constructed on meta and must be recreated on CPU.
        from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeRotaryEmbedding
        self.model.model.rotary_emb=GraniteMoeRotaryEmbedding(config=config,device='cpu')
        if any(p.is_meta for p in self.model.parameters()):raise ValueError('Unmapped core parameter')
        self.model.eval();self.tokenizer=load_tokenizer(self.connection,self.metadata)
        self._past=None;self._processed_ids=[]
        if self.device.type=='cuda':
            try:
                self.model.to(self.device)
            except RuntimeError as exc:
                warnings.warn(f'GPU initialization failed: {exc}. Using CPU.')
                self.fallback_to_cpu()
        self.core_bytes=core_bytes;self.load_seconds=time.perf_counter()-started
        self.connection.commit()
    def status(self):return dict(repo=self.metadata['repo'],device=str(self.device),
        gpu_name=torch.cuda.get_device_name() if self.device.type=='cuda' else None,
        startup_seconds=self.load_seconds,core_mib=self.core_bytes/1024**2,selection=self.selection,**self.cache.stats())
    def fallback_to_cpu(self):
        self.reset_context()
        self.cache.cancel_prefetch()
        self.cache.items.clear();self.cache.bytes=0
        self.cache.layer_bytes={layer:0 for layer in self.cache.layer_bytes}
        self.device=torch.device('cpu');self.cache.device=self.device
        self.selection['reason']='Switched to CPU after CUDA memory or initialization failure'
        self.cache.pin_host=False
        self.model.cpu()
        torch.cuda.empty_cache()
    def check_resources(self):
        ram=available_ram_mib()
        if ram<1536:
            if self.cache.host_bytes:
                self.cache.trim_host(max(0,self.cache.host_limit/1024**2-(2048-ram)))
            if self.device.type=='cpu':
                self.cache.trim_execution(max(96,self.cache.limit/1024**2-(2048-ram)))
        if self.device.type=='cuda':
            free=torch.cuda.mem_get_info()[0]+torch.cuda.memory_reserved()-torch.cuda.memory_allocated()
            if free<256*1024**2:
                budget=self.cache.limit/1024**2-(384-free/1024**2)
                if budget<384:
                    warnings.warn('GPU memory is now occupied. Using CPU.')
                    self.fallback_to_cpu()
                else:
                    self.cache.trim_execution(budget)
                    torch.cuda.empty_cache()
    @torch.inference_mode()
    def generate(self,messages,max_tokens=512,callback=None):
        self.check_resources()
        try:
            try:
                return self._generate(messages,max_tokens,callback)
            except torch.cuda.OutOfMemoryError:
                if self.device.type!='cuda':raise
            warnings.warn('GPU memory exhausted. Retrying on CPU.')
            self.fallback_to_cpu()
            if callback:callback('')
            return self._generate(messages,max_tokens,callback)
        finally:
            connection=getattr(self,'connection',None)
            if connection is not None:connection.commit()
    def _generate(self,messages,max_tokens=512,callback=None):
        # Publishing new weights waits for the current response to finish.
        self.connection.commit()
        with self.connection.cursor() as cur:
            cur.execute('SELECT revision FROM moe_model WHERE id=1 LOCK IN SHARE MODE')
            revision=cur.fetchone()['revision']
        if revision!=self.metadata['revision']:
            self.reset_context()
            raise RuntimeError('Database weights changed; restart this chat to load the new revision')
        ids=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt')
        if ids.shape[1]>1024:raise ValueError('Local context limit: 1024 tokens; use /reset')
        start=time.perf_counter();generated=[];first_token_seconds=None;finish_reason='length'
        prompt_ids=ids[0].tolist();common=0
        for a,b in zip(prompt_ids,self._processed_ids):
            if a!=b:break
            common+=1
        reused=min(common,len(prompt_ids)-1)
        past=self._past if reused else None
        if past is not None:past.crop(reused)
        else:reused=0
        processed=prompt_ids[:reused];current=ids[:,reused:].to(self.device)
        # Clear stored state before execution so an exception cannot retain a
        # partially advanced KV cache paired with stale token IDs.
        self._past=None;self._processed_ids=[]
        generation=json.loads(self.metadata['generation_config']);eos=generation.get('eos_token_id',self.tokenizer.eos_token_id)
        eos=set(eos if isinstance(eos,list) else [eos]);before=self.cache.stats()
        for _ in range(max_tokens):
            pause_callback=getattr(self,'pause_callback',None)
            if pause_callback:pause_callback()
            result=self.model(input_ids=current,past_key_values=past,use_cache=True,logits_to_keep=1)
            processed.extend(current[0].tolist())
            token=int(result.logits[0,-1].argmax());past=result.past_key_values
            if first_token_seconds is None:first_token_seconds=time.perf_counter()-start
            if token in eos:
                finish_reason='eos';break
            generated.append(token);current=torch.tensor([[token]],device=self.device)
            if callback:callback(self.tokenizer.decode(generated,skip_special_tokens=True))
            if getattr(self,'planning_mode',False) and decode_decision(self.tokenizer.decode(generated,skip_special_tokens=True)) is not None:
                finish_reason='tool_call';break
            if getattr(self,'tools_enabled',False) and (decode_protocol_call(self.tokenizer.decode(generated,skip_special_tokens=True)) is not None or decode_decision(self.tokenizer.decode(generated,skip_special_tokens=True)) is not None):
                finish_reason='tool_call';break
        seconds=time.perf_counter()-start
        self._past=past;self._processed_ids=processed
        return dict(device=str(self.device),finish_reason=finish_reason,reused_context_tokens=reused,answer=self.tokenizer.decode(generated,skip_special_tokens=True),tokens=len(generated),seconds=seconds,first_token_seconds=first_token_seconds,tokens_per_second=len(generated)/max(seconds,1e-9),router_seconds=self.cache.router_seconds-before['router_seconds'],expert_read_seconds=self.cache.read_seconds-before['expert_read_seconds'],expert_prepare_seconds=self.cache.prepare_seconds-before['expert_prepare_seconds'],expert_load_seconds=self.cache.load_seconds-before['expert_load_seconds'],expert_transfer_mib=(self.cache.transferred_bytes/1024**2)-before['expert_transfer_mib'],cache_misses=self.cache.misses-before['misses'],
            host_cache_hits=self.cache.host_hits-before['host_cache_hits'],
            database_expert_loads=self.cache.database_loads-before['database_expert_loads'])
    def reset_context(self):
        self._past=None;self._processed_ids=[]
    def close(self):self.reset_context();self.cache.close();self.connection.close()

def main():
    parser=argparse.ArgumentParser(prog="sqlt",description=__doc__);parser.add_argument('--cache-mib',type=int,default=1536);parser.add_argument('--int8',action='store_true',help='Approximate INT8 expert execution; SQL weights remain unchanged')
    parser.add_argument('--max-tokens',type=int,default=512,help='Maximum generated tokens per chat response')
    parser.add_argument('--device',choices=['auto','cuda','cpu'],default='auto',help='Prefer CUDA when available; fall back to CPU')
    parser.add_argument('--host-cache-mib',type=int,default=6144,help='Maximum RAM cache for GPU experts; 0 disables it')
    parser.add_argument('--db-workers',type=int,choices=(0,1,2,4),default=2,help='Parallel SQL readers; 0 disables prefetch')
    sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('chat');sub.add_parser('status')
    ask=sub.add_parser('ask');ask.add_argument('text');ask.add_argument('--tokens',type=int,default=512)
    args=parser.parse_args()
    if args.max_tokens<1 or (args.command=='ask' and args.tokens<1):parser.error('Token limit must be positive')
    if args.host_cache_mib<0:parser.error('RAM cache size must be nonnegative')
    if args.command=='chat':print('Loading SQLT…',flush=True)
    model=DatabaseMoE(args.cache_mib,quantized_experts=args.int8,device=args.device,host_cache_mib=args.host_cache_mib,db_workers=args.db_workers)
    model.tools_enabled=False
    messages=[{'role':'system','content':SYSTEM_INSTRUCTIONS}]
    try:
        if args.command=='status':print(json.dumps(model.status(),ensure_ascii=False));return
        if args.command=='ask':
            messages.append({'role':'user','content':args.text});print(json.dumps(generate_with_tools(model,messages,args.tokens),ensure_ascii=False,indent=2));return
        ui=TerminalChat()
        ui.banner(torch.cuda.get_device_name()+' (CUDA)' if model.device.type=='cuda' else 'CPU')
        while True:
            text=ui.read_input().strip()
            if text=='/help':ui.help();continue
            if text=='/quit':break
            if text=='/status':print(json.dumps(model.status(),ensure_ascii=False));continue
            if text=='/reset':messages=messages[:1];model.reset_context();continue
            if text=='/learn' or text.startswith('/learn '):
                path_text=text[len('/learn'):].strip()
                if not path_text:
                    print('Usage: /learn /path/to/file.txt — train each training block once.');continue
                # The remainder is a literal path, including any spaces.
                if len(path_text)>=2 and path_text[0]==path_text[-1] and path_text[0] in ('"', "'"):
                    path_text=path_text[1:-1]
                source=Path(path_text).expanduser()
                try:
                    with source.open('rb') as stream:
                        raw=stream.read(8*1024*1024+1)
                    if len(raw)>8*1024*1024:raise ValueError('File exceeds the 8 MiB limit')
                    content=raw.decode('utf-8-sig',errors='strict')
                    count=len(model.tokenizer.encode(content,add_special_tokens=False))
                    if count-int(count*0.9)<2 or int(count*0.9)<2:
                        raise ValueError('File too short: provide at least 20 tokens')
                except (OSError,UnicodeError,ValueError) as exc:
                    print(f'Cannot learn this file: {exc}');continue
                print('Training once over every training block; 10% is held out for validation. This may take minutes.',flush=True)
                model.close();model=None
                gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
                try:
                    result=subprocess.run([sys.executable,str(Path(__file__).with_name('learn.py')),
                        '--one-pass',str(source)],check=False)
                    print('Training completed.' if result.returncode==0 else 'Training did not complete successfully; see the trainer output.',flush=True)
                finally:
                    print('Reloading database weights...',flush=True)
                    model=DatabaseMoE(args.cache_mib,quantized_experts=args.int8,device=args.device,
                        host_cache_mib=args.host_cache_mib,db_workers=args.db_workers)
                    model.tools_enabled=False
                    messages=messages[:1]
                continue
            if not text:continue
            messages.append({'role':'user','content':text})
            try:
                printed=''
                if ui.enabled:ui.start()
                else:print('Model> ',end='',flush=True)
                waiting_started=time.perf_counter();last_progress=0.;waiting=True;progress_visible=False
                def progress():
                    nonlocal last_progress,progress_visible
                    pause_callback=getattr(model,'pause_callback',None)
                    if pause_callback:pause_callback()
                    now=time.perf_counter()
                    if waiting and now-last_progress>=2:
                        if ui.enabled:ui.progress(now-waiting_started)
                        else:print(f'\rAttendi: {now-waiting_started:.0f}s, {model.cache.database_loads} esperti letti dal database',end='',flush=True)
                        last_progress=now;progress_visible=True
                def stream(value):
                    nonlocal printed,waiting
                    waiting=False
                    if ui.enabled:
                        ui.stream(value);return
                    if progress_visible and not printed:print('\r\033[2KModel> ',end='',flush=True)
                    if value=='' and printed:
                        print('\nModel> ',end='',flush=True);printed='';return
                    if value.endswith('\ufffd'):return
                    if value.startswith(printed):print(value[len(printed):],end='',flush=True);printed=value
                model.cache.progress_callback=progress
                try:
                    with ui.pause_control() as pause:
                        model.pause_callback=pause.checkpoint
                        result=generate_with_tools(model,messages,max_tokens=args.max_tokens,callback=stream)
                finally:
                    model.cache.progress_callback=None
                    model.pause_callback=None
                stream(result['answer'])
                if ui.enabled:ui.finish(result)
                else:print()
                if result['finish_reason']=='length':print(f"[Response stopped at the {args.max_tokens}-token limit. Increase --max-tokens to allow longer answers.]")
                messages.append({'role':'assistant','content':result['answer']})
            except Exception as exc:
                messages.pop()
                if ui.enabled:ui.error(exc)
                else:print('\r\033[2KError: '+str(exc))
    except (EOFError,KeyboardInterrupt):
        if 'ui' in locals():ui.finish()
        print()
    finally:
        if model is not None:model.close()
if __name__=='__main__':main()
