"""Measure cold, repeated and novel SQL-backed requests in the default INT8 mode."""
import json,resource,time
from pathlib import Path
from runtime import DatabaseMoE
root=Path(__file__).parent
model=DatabaseMoE(cache_mib=3584,quantized_experts=True)
report={'startup':model.status(),'requests':[]}
questions=[('cold','Come si chiama la capitale italiana?'),('repeated','Come si chiama la capitale italiana?'),('novel','Quanto fa 2 + 2?'),('science','Spiega brevemente cosa è un atomo.')]
try:
    for name,text in questions:
        messages=[{'role':'system','content':'Rispondi in italiano in modo conciso.'},{'role':'user','content':text}]
        result=model.generate(messages,max_tokens=96)
        report['requests'].append({'name':name,'question':text,**result})
        report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        (root/'benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report['requests'][-1],ensure_ascii=False),flush=True)
    report['final_status']=model.status()
    (root/'benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
finally:model.close()
