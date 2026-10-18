"""Natural conversation learning and selective forgetting with fresh GPT sessions."""
import json
import uuid
from clients.codex_chat import CodexMemoryClient
from episodic_memory import EpisodicMemory
ns='check_'+uuid.uuid4().hex[:12];client=CodexMemoryClient(ns)
project='Vela-'+uuid.uuid4().hex[:6];secret='onice-'+uuid.uuid4().hex[:8]
report=dict(scope='Natural requests; reversible exclusion from active memory, not permanent physical erasure or GPT weight changes')
try:
    report['learn']=client.turn(f'Il codice del progetto {project} è {secret}.')
    print('statement stored',flush=True)
    report['recall']=client.turn(f'Quale codice è associato al progetto {project}?')
    print('recall complete',flush=True)
    report['forget']=client.turn(f'Non mi serve più il ricordo del codice del progetto {project}. Eliminalo dalla memoria.')
    print('forget complete',flush=True)
    report['after']=client.turn(f'Qual è il codice del progetto {project}?')
    m=EpisodicMemory(ns)
    try:report['events_remaining']=m.status()['events'];report['archived_events']=m.status()['archived_events']
    finally:m.close()
    report['passed']=(report['learn']['operation']=='learn' and report['learn']['verified_memory_access'] and secret in report['recall']['answer'] and report['recall']['verified_memory_access'] and report['forget']['operation']=='forget' and report['forget']['verified_memory_access'] and set(report['forget']['event_ids'])==set(report['learn']['event_ids']) and report['events_remaining']==0 and report['archived_events']==1 and secret not in report['after']['answer'] and report['after']['verified_memory_access'])
finally:
    m=EpisodicMemory(ns)
    try:m.delete_namespace()
    finally:m.close()
    open('verification-forgetting.json','w').write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('scope','events_remaining','archived_events','passed')},ensure_ascii=False),flush=True)
if not report.get('passed'):raise SystemExit(1)
