"""Two fresh Codex GPT-6.1 Sol sessions exercising the local MCP server."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid
from episodic_memory import EpisodicMemory
from clients.local_mcp import local_mcp_args

ROOT=Path(__file__).resolve().parent
ns='check_gpt_'+uuid.uuid4().hex[:10]
base=['codex',*local_mcp_args(),'exec','--skip-git-repo-check','--ephemeral','--json','-m','gpt-6.1-sol','-C',str(ROOT),'-c','approval_policy="never"','-c','model_reasoning_effort="low"']
report={'model':'gpt-6.1-sol','namespace':ns,'scope':'Actual model via Codex and local MCP; separate teaching and questioning sessions'}
facts=['Marco ha dato un libro a Luca.','Sara non ha dato una chiave a Paolo.','Giulia ha ricevuto una penna da Andrea.','Luca ha dato un quaderno a Marco.']

def run(name,prompt):
    started=time.perf_counter()
    result=subprocess.run(base+['-o',str(ROOT/(name+'-answer.txt')),'-'],input=prompt,capture_output=True,text=True,timeout=180)
    events=[]
    for line in result.stdout.splitlines():
        try:events.append(json.loads(line))
        except ValueError:pass
    (ROOT/(name+'-events.json')).write_text(json.dumps(events,ensure_ascii=False,indent=2)+'\n')
    answer_path=ROOT/(name+'-answer.txt')
    return dict(exit_code=result.returncode,seconds=time.perf_counter()-started,answer=answer_path.read_text() if answer_path.exists() else '',stderr=result.stderr[-3000:],events=events)

def evaluate(report):
    calls=[]
    for event in report.get('questioning',{}).get('events',[]):
        item=event.get('item',{})
        if item.get('type')=='mcp_tool_call' and item.get('status')=='completed' and item.get('server')=='cortex' and item.get('tool')=='recall_events' and not item.get('error'):
            result=item['result'];payload=result.get('structured_content')
            if payload is None:payload=json.loads(result['content'][0]['text'])
            calls.append((item['arguments'],payload))
    def matching(agent=None,patient=None,recipient=None,negative=False):
        wanted={k:v for k,v in [('agent',agent),('patient',patient),('recipient',recipient)] if v is not None}
        return [payload for args,payload in calls if args.get('negated',False)==negative and all({b['role']:b['value'].casefold() for b in args['roles']}.get(k)==v for k,v in wanted.items())]
    def has_value(payload,role,value):
        return any({b['role']:b['value'] for b in e['roles']}.get(role)==value for e in payload['events'])
    book=matching(agent='marco',patient='libro')
    pen=matching(patient='penna',recipient='giulia')
    notebook=matching(agent='luca',patient='quaderno')
    positive=matching(agent='sara',patient='chiave')
    negative=matching(agent='sara',patient='chiave',negative=True)
    checks=dict(two_rephrased_queries_used_memory=sum(has_value(x,'recipient','luca') for x in book)>=2,
        receive_roles_retrieved=any(has_value(x,'agent','andrea') for x in pen),
        reversed_roles_retrieved=any(has_value(x,'recipient','marco') for x in notebook),
        no_positive_for_denied_event=bool(positive) and all(x['count']==0 for x in positive),
        explicit_negative_retrieved=any(x['count']==1 for x in negative))
    report['verified_tool_results']=checks
    lines=report.get('questioning',{}).get('answer','').casefold().splitlines()
    expected=['luca','luca','andrea','marco']
    report['answer_label_checks']=[any(line.startswith(str(i+1)+'.') and name in line for line in lines) for i,name in enumerate(expected)]
    report['negative_answer_check']=any(line.startswith('5.') and 'no' in line and 'negat' in line for line in lines)
    report['used_no_shell']=not any(e.get('item',{}).get('type')=='command_execution' for phase in ('teaching','questioning') for e in report.get(phase,{}).get('events',[]))
    report['passed']=all(report.get('semantic_storage_checks',[False])) and all(checks.values()) and all(report['answer_label_checks']) and report['negative_answer_check'] and report['used_no_shell'] and not report.get('error')


try:
    report['teaching']=run('gpt-teaching','Usa esclusivamente gli strumenti MCP cortex, niente shell o file. Nello spazio '+ns+' memorizza separatamente queste quattro dichiarazioni utente, conservando la negazione e i ruoli funzionali anche nella frase con ricevere:\n'+'\n'.join(facts)+'\nUsa frame give per questi trasferimenti. Chiama memory_status alla fine. Non rispondere a domande, registra soltanto e conferma il numero di episodi.')
    print('teaching session complete',report['teaching']['exit_code'],flush=True)
    if report['teaching']['exit_code']!=0:raise RuntimeError('Requested model teaching session failed')
    m=EpisodicMemory(ns)
    try:
        expectations=[('marco','libro','luca',False),('sara','chiave','paolo',True),('andrea','penna','giulia',False),('luca','quaderno','marco',False)]
        checks=[]
        for agent,patient,recipient,negative in expectations:
            events=m.find('give',[dict(role='agent',value=agent),dict(role='patient',value=patient)],negative)
            checks.append(len(events)==1 and {b['role']:b['value'] for b in events[0]['roles']}.get('recipient')==recipient)
        report['semantic_storage_checks']=checks
    finally:m.close()
    # No original teaching facts are supplied to this new model session.
    report['questioning']=run('gpt-questioning','Usa esclusivamente gli strumenti MCP cortex, niente shell o file. Consulta la memoria persistente nello spazio '+ns+'. Non hai le informazioni nel prompt: chiama recall_events per ogni domanda e non inventare fatti. Rispondi in italiano con una lista numerata:\n1. A quale persona Marco ha consegnato il libro?\n2. Chi ha ricevuto il libro da Marco?\n3. Chi ha dato la penna a Giulia?\n4. Chi ha ricevuto il quaderno da Luca?\n5. Posso affermare dalla memoria che Sara ha dato la chiave a Paolo? Controlla sia eventi positivi sia negati.')
    print('questioning session complete',report['questioning']['exit_code'],flush=True)
except Exception as e:report['error']=str(e)
finally:
    m=EpisodicMemory(ns)
    try:m.delete_namespace()
    finally:m.close()
    evaluate(report)
    (ROOT/'verification-gpt-mcp.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('teaching','questioning')},ensure_ascii=False),flush=True)
