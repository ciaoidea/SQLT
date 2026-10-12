"""Measure real requests and verify the batched expert path."""
import argparse,json,resource
from pathlib import Path
import torch
from runtime import DatabaseMoE

parser=argparse.ArgumentParser()
parser.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
parser.add_argument('--output',type=Path,default=Path('performance-after.json'))
parser.add_argument('--db-workers',type=int,default=2)
args=parser.parse_args()
model=DatabaseMoE(cache_mib=3584,quantized_experts=True,device=args.device,db_workers=args.db_workers)
report={'startup':model.status(),'requests':[]}
try:
    for label,text,tokens in (
        ('cold','Rispondi solo con la parola: ciao',3),
        ('warm','Rispondi solo con la parola: ciao',3),
        ('novel','Quanto fa 2 + 2? Rispondi con il numero.',8)):
        model.reset_context()
        result=model.generate([
            {'role':'system','content':'Sei un assistente utile. Rispondi in italiano in modo conciso.'},
            {'role':'user','content':text}],max_tokens=tokens)
        report['requests'].append({'name':label,**result})
        report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
        print(json.dumps(report['requests'][-1],ensure_ascii=False),flush=True)
    if model.device.type=='cuda':
        ids=torch.tensor([[model.tokenizer.encode('Ciao',add_special_tokens=False)[0]]],device='cuda')
        with torch.inference_mode():
            batched=model.model(input_ids=ids,use_cache=False,logits_to_keep=1).logits
            model.cache.batch_experts=False
            sequential=model.model(input_ids=ids,use_cache=False,logits_to_keep=1).logits
        difference=float((batched-sequential).abs().max())
        assert torch.allclose(batched,sequential,atol=3e-4,rtol=3e-4),difference
        assert int(batched.argmax())==int(sequential.argmax())
        report['batch_verification']={'max_logit_difference':difference,'same_greedy_token':True}
    report['final_status']=model.status()
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
finally:model.close()
