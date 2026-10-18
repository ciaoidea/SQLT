"""Real free-text acquisition without commands or lexical intent dispatch."""
import json
import uuid
from clients.codex_chat import CodexMemoryClient
from episodic_memory import EpisodicMemory

ns='check_'+uuid.uuid4().hex[:12]
client=CodexMemoryClient(ns)
project='Orione-'+uuid.uuid4().hex[:6];code='ambra-'+uuid.uuid4().hex[:8]
report=dict(scope='Default auto mode; independent GPT sessions; no natural-language trigger parser')
try:
    report['statement']=client.turn(f'Il progetto {project} usa il codice {code}.')
    print('statement complete',flush=True)
    report['question']=client.turn(f'Qual è il codice associato a {project}?')
    print('question complete',flush=True)
    m=EpisodicMemory(ns)
    try:report['events_after_question']=m.status()['events']
    finally:m.close()
    report['passed']=(report['statement']['requested_operation']=='auto' and report['statement']['operation']=='learn' and report['statement']['verified_memory_access'] and report['question']['operation']=='query' and report['question']['verified_memory_access'] and code in report['question']['answer'] and bool(set(report['statement']['event_ids']) & set(report['question']['event_ids'])))
finally:
    m=EpisodicMemory(ns)
    try:m.delete_namespace()
    finally:m.close()
    open('verification-natural-chat.json','w').write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('statement','question')},ensure_ascii=False),flush=True)
if not report.get('passed'):raise SystemExit(1)
