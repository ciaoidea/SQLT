"""Pretrained MoE inference with SQL-selected experts and a bounded LRU cache.

Core parameters and routers are resident; expert parameters remain in MariaDB
until actual pretrained routing selects them. No checkpoint is read at runtime.
"""
import argparse,hashlib,json,os,time,warnings
from collections import OrderedDict
import numpy as np
import pymysql
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from transformers import AutoConfig,AutoModelForCausalLM,PreTrainedTokenizerFast
from database import connect

torch.set_num_threads(int(os.environ.get('MODEL_THREADS','2')))

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
        from torch.ao.nn.quantized.dynamic import Linear
        scales=weight.abs().amax(dim=1).clamp_min(1e-12)/127
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message=r'torch\.quantize_per_tensor, torch\.quantize_per_channel.*deprecated.*',category=UserWarning)
            quantized=torch.quantize_per_channel(weight,scales,torch.zeros(weight.shape[0],dtype=torch.int64),0,torch.qint8)
        self.linear=Linear(weight.shape[1],weight.shape[0],bias_=False)
        self.linear.set_weight_bias(quantized,None);self.linear.eval()
        self.bytes=weight.numel()+weight.shape[0]*16
    def __call__(self,inputs):return self.linear(inputs)

def stored_bytes(value):
    return value.bytes if isinstance(value,Int8Projection) else value.numel()*value.element_size()

class ExpertCache:
    def __init__(self,connection,parameters,experts,limit_mib,quantized=False):
        if limit_mib<12:raise ValueError('Expert cache must be at least 12 MiB')
        self.connection=connection;self.parameters=parameters;self.experts=experts;self.quantized=quantized
        if quantized:torch.backends.quantized.engine='fbgemm'
        self.limit=int(limit_mib*1024**2);self.items=OrderedDict();self.bytes=0
        self.layer_count=len({layer for layer,_ in experts});self.layer_limit=self.limit//self.layer_count
        self.layer_bytes={layer:0 for layer,_ in experts}
        self.hits=0;self.misses=0;self.transferred_bytes=0;self.load_seconds=0.;self.router_seconds=0.;self.read_seconds=0.;self.prepare_seconds=0.;self.routes=[]
    def prepare(self,layer,selected):
        # Protect cached experts needed later in the same routed layer.
        for expert in selected:
            key=(layer,expert)
            if key in self.items:self.items.move_to_end(key)
    def get(self,layer,expert):
        key=(layer,expert)
        if key in self.items:
            self.hits+=1;self.items.move_to_end(key);return self.items[key]
        self.misses+=1;start=time.perf_counter();binding=self.experts[key]
        metadata=[self.parameters[binding[k]] for k in ('input_parameter','output_parameter')]
        wire_bytes=sum(int(np.prod(json.loads(m['shape'])))*4 for m in metadata)
        tensors=[]
        for meta in metadata:
            read_started=time.perf_counter()
            tensor=load_parameter(self.connection,meta)
            self.read_seconds+=time.perf_counter()-read_started
            prepare_started=time.perf_counter()
            if self.quantized:
                tensors.append(Int8Projection(tensor))
            else:
                compact=tensor.to(torch.bfloat16)
                # Lossless for original BF16 weights; retain any FP32 updates.
                tensors.append(compact if torch.equal(tensor,compact.float()) else tensor)
            self.prepare_seconds+=time.perf_counter()-prepare_started
        tensors=tuple(tensors)
        required=sum(stored_bytes(t) for t in tensors)
        if required>self.layer_limit:raise ValueError('Expert exceeds per-layer cache budget; increase --cache-mib')
        while self.layer_bytes[layer]+required>self.layer_limit:
            victim=next(k for k in self.items if k[0]==layer)
            old=self.items.pop(victim);released=sum(stored_bytes(t) for t in old)
            self.bytes-=released;self.layer_bytes[layer]-=released
            del old
        self.items[key]=tensors;self.bytes+=required;self.layer_bytes[layer]+=required;self.transferred_bytes+=wire_bytes
        self.load_seconds+=time.perf_counter()-start
        return tensors
    def stats(self):
        return dict(cache_policy=('per-layer LRU, approximate INT8' if self.quantized else 'per-layer LRU, lossless BF16 when representable'),cache_limit_mib=self.limit/1024**2,cache_resident_mib=self.bytes/1024**2,cached_experts=len(self.items),hits=self.hits,misses=self.misses,expert_transfer_mib=self.transferred_bytes/1024**2,expert_load_seconds=self.load_seconds,router_seconds=self.router_seconds,expert_read_seconds=self.read_seconds,expert_prepare_seconds=self.prepare_seconds,last_routes=[dict(layer=r['layer'],experts=list(r['experts']),tokens=r['tokens']) for r in self.routes])

class SQLExperts(nn.Module):
    def __init__(self,layer,router,cache,activation):
        super().__init__();self.layer=layer;self.router=router;self.cache=cache;self.activation=activation
    def forward(self,layer_input):
        batch,length,width=layer_input.shape;flat=layer_input.reshape(-1,width)
        route_started=time.perf_counter()
        _,indices,gates,sizes,logits=self.router(flat)
        self.cache.router_seconds+=time.perf_counter()-route_started
        out=torch.zeros_like(flat)
        self.cache.prepare(self.layer,[expert for expert,size in enumerate(sizes) if size])
        offset=0;selected=[]
        for expert,size in enumerate(sizes):
            if not size:continue
            selected.append(expert)
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
    def __init__(self,cache_mib=1536,allow_unverified=False,quantized_experts=False):
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
        # Meta allocation creates architecture only: not one expert weight is allocated.
        with torch.device('meta'):self.model=AutoModelForCausalLM.from_config(config)
        self.cache=ExpertCache(self.connection,self.parameters,self.experts,cache_mib,quantized=quantized_experts)
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
        self.core_bytes=core_bytes;self.load_seconds=time.perf_counter()-started
        self.connection.commit()
    def status(self):return dict(repo=self.metadata['repo'],startup_seconds=self.load_seconds,core_mib=self.core_bytes/1024**2,**self.cache.stats())
    @torch.inference_mode()
    def generate(self,messages,max_tokens=512,callback=None):
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
        processed=prompt_ids[:reused];current=ids[:,reused:]
        # Clear stored state before execution so an exception cannot retain a
        # partially advanced KV cache paired with stale token IDs.
        self._past=None;self._processed_ids=[]
        generation=json.loads(self.metadata['generation_config']);eos=generation.get('eos_token_id',self.tokenizer.eos_token_id)
        eos=set(eos if isinstance(eos,list) else [eos]);before=self.cache.stats()
        for _ in range(max_tokens):
            result=self.model(input_ids=current,past_key_values=past,use_cache=True,logits_to_keep=1)
            processed.extend(current[0].tolist())
            token=int(result.logits[0,-1].argmax());past=result.past_key_values
            if first_token_seconds is None:first_token_seconds=time.perf_counter()-start
            if token in eos:
                finish_reason='eos';break
            generated.append(token);current=torch.tensor([[token]])
            if callback:callback(self.tokenizer.decode(generated,skip_special_tokens=True))
        seconds=time.perf_counter()-start
        self._past=past;self._processed_ids=processed
        return dict(finish_reason=finish_reason,reused_context_tokens=reused,answer=self.tokenizer.decode(generated,skip_special_tokens=True),tokens=len(generated),seconds=seconds,first_token_seconds=first_token_seconds,tokens_per_second=len(generated)/max(seconds,1e-9),router_seconds=self.cache.router_seconds-before['router_seconds'],expert_read_seconds=self.cache.read_seconds-before['expert_read_seconds'],expert_prepare_seconds=self.cache.prepare_seconds-before['expert_prepare_seconds'],expert_load_seconds=self.cache.load_seconds-before['expert_load_seconds'],expert_transfer_mib=(self.cache.transferred_bytes/1024**2)-before['expert_transfer_mib'],cache_misses=self.cache.misses-before['misses'])
    def reset_context(self):
        self._past=None;self._processed_ids=[]
    def close(self):self.reset_context();self.connection.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cache-mib',type=int,default=1536);parser.add_argument('--int8',action='store_true',help='Approximate INT8 expert execution; SQL weights remain unchanged')
    parser.add_argument('--max-tokens',type=int,default=512,help='Maximum generated tokens per chat response')
    sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('chat');sub.add_parser('status')
    ask=sub.add_parser('ask');ask.add_argument('text');ask.add_argument('--tokens',type=int,default=512)
    args=parser.parse_args()
    if args.max_tokens<1 or (args.command=='ask' and args.tokens<1):parser.error('Token limit must be positive')
    model=DatabaseMoE(args.cache_mib,quantized_experts=args.int8)
    messages=[{'role':'system','content':'Sei un assistente utile. Rispondi in italiano in modo conciso.'}]
    try:
        if args.command=='status':print(json.dumps(model.status(),ensure_ascii=False));return
        if args.command=='ask':
            messages.append({'role':'user','content':args.text});print(json.dumps(model.generate(messages,args.tokens),ensure_ascii=False,indent=2));return
        while True:
            text=input('You> ').strip()
            if text=='/quit':break
            if text=='/status':print(json.dumps(model.status(),ensure_ascii=False));continue
            if text=='/reset':messages=messages[:1];model.reset_context();continue
            if not text:continue
            messages.append({'role':'user','content':text})
            try:
                printed=''
                print('Model> ',end='',flush=True)
                def stream(value):
                    nonlocal printed
                    if value.endswith('\ufffd'):return
                    if value.startswith(printed):print(value[len(printed):],end='',flush=True);printed=value
                result=model.generate(messages,max_tokens=args.max_tokens,callback=stream);stream(result['answer']);print()
                if result['finish_reason']=='length':print(f"[Response stopped at the {args.max_tokens}-token limit. Increase --max-tokens to allow longer answers.]")
                messages.append({'role':'assistant','content':result['answer']})
            except Exception as exc:messages.pop();print('Error: '+str(exc))
    except (EOFError,KeyboardInterrupt):print()
    finally:model.close()
if __name__=='__main__':main()
