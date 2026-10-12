"""Verify coefficients, tokenizer and lossless routed execution in separate phases."""
import argparse,gc,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from store import open_source
from database import connect
from runtime import DatabaseMoE,load_parameter
ROOT=Path(__file__).parent

def save(name,report):
    report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (ROOT/name).write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

def weights(folder):
    started=time.perf_counter();count=0
    with connect() as c,c.cursor() as cur,open_source(folder) as source:
        cur.execute('SELECT * FROM moe_parameters ORDER BY id');metadata=cur.fetchall()
        for i,meta in enumerate(metadata):
            base=meta['name'].rsplit('.expert.',1)[0] if meta['expert_no'] is not None else meta['name']
            expected=source.get_tensor(base)
            if meta['expert_no'] is not None:expected=expected[meta['expert_no']]
            expected=expected.float()
            actual=load_parameter(c,meta)
            if not torch.equal(actual,expected):raise ValueError('Coefficient mismatch: '+meta['name'])
            count+=actual.numel()
            del actual,expected
            if i%100==0:print(json.dumps({'verified':i+1,'total':len(metadata)}),flush=True)
    save('weight-verification.json',dict(parameter_blocks=len(metadata),learned_coefficients=count,all_coefficients_equal=True,seconds=time.perf_counter()-started))

def reference(folder):
    started=time.perf_counter();cfg=AutoConfig.from_pretrained(folder,local_files_only=True);cfg._attn_implementation='sdpa'
    with torch.device('meta'):model=AutoModelForCausalLM.from_config(cfg)
    # Materialize one tensor at a time; avoid holding a second full state_dict.
    with open_source(folder) as source:
        for name in source.keys():
            weight=source.get_tensor(name).float()
            parent,attribute=name.rsplit('.',1)
            setattr(model.get_submodule(parent),attribute,torch.nn.Parameter(weight,requires_grad=False))
    model.tie_weights()
    from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeRotaryEmbedding
    model.model.rotary_emb=GraniteMoeRotaryEmbedding(config=cfg,device='cpu');model.eval()
    if any(p.is_meta for p in model.parameters()):raise ValueError('Reference has unmapped parameters')
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    ids=torch.tensor([[tokenizer.encode('Ciao',add_special_tokens=False)[0]]])
    with torch.inference_mode():
        first=model(input_ids=ids,use_cache=True,logits_to_keep=1)
        next_id=first.logits[:,-1].argmax(-1).reshape(1,1)
        second=model(input_ids=next_id,past_key_values=first.past_key_values,use_cache=True,logits_to_keep=1)
    np.savez(ROOT/'reference-logits.npz',ids=ids.numpy(),next_id=next_id.numpy(),first=first.logits.numpy(),second=second.logits.numpy())
    save('reference-verification.json',dict(reference='Official architecture, original BF16 coefficients represented exactly as FP32',seconds=time.perf_counter()-started))

def compare(folder):
    started=time.perf_counter();model=DatabaseMoE(cache_mib=3072,allow_unverified=True)
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    texts=['Ciao','Il gatto è morto.','La casa di Giovanni è in montagna.']
    for text in texts:
        if model.tokenizer.encode(text)!=tokenizer.encode(text):raise ValueError('Tokenizer mismatch')
    messages=[{'role':'user','content':'Ciao'}]
    if model.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)!=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True):raise ValueError('Chat template mismatch')
    cfg=json.loads(model.metadata['config']);selected=cfg['num_hidden_layers']*cfg['num_experts_per_tok'];available=cfg['num_hidden_layers']*cfg['num_local_experts']
    ref=np.load(ROOT/'reference-logits.npz');cases=[]
    with torch.inference_mode():
        first=model.model(input_ids=torch.from_numpy(ref['ids']),use_cache=True,logits_to_keep=1)
        if model.cache.misses!=selected:raise ValueError('Unexpected expert selection')
        if len(model.cache.routes)!=cfg['num_hidden_layers'] or any(len(r['experts'])!=cfg['num_experts_per_tok'] for r in model.cache.routes):raise ValueError('Routing mismatch')
        second=model.model(input_ids=torch.from_numpy(ref['next_id']),past_key_values=first.past_key_values,use_cache=True,logits_to_keep=1)
        for name,actual in [('first',first),('second',second)]:
            expected=torch.from_numpy(ref[name]);diff=float((actual.logits-expected).abs().max())
            if not torch.allclose(actual.logits,expected,atol=3e-4,rtol=3e-4):raise ValueError(f'Logit mismatch: {name} {diff}')
            if int(actual.logits.argmax())!=int(expected.argmax()):raise ValueError('Greedy token mismatch')
            cases.append(dict(step=name,max_logit_difference=diff,same_greedy_token=True))
    report=dict(weight_verification=json.loads((ROOT/'weight-verification.json').read_text()),reference=json.loads((ROOT/'reference-verification.json').read_text()),tokenizer_cases=len(texts),chat_template_equal=True,selected_experts_first_token=selected,available_experts=available,cases=cases,startup=model.status(),seconds=time.perf_counter()-started)
    save('verification.json',report)
    with connect() as c,c.cursor() as cur:
        cur.execute("UPDATE moe_model SET status='ready' WHERE id=1")
        cur.execute("INSERT INTO moe_events(kind,metrics) VALUES('verification',%s)",(json.dumps(report),));c.commit()
    model.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['weights','reference','compare']);p.add_argument('folder',type=Path);args=p.parse_args()
    globals()[args.phase](args.folder)
