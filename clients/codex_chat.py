"""Terminal memory chat and independent checks through real MCP calls.

Conversations use MCP and never import a local LLM. Explicit confirmed local
administration is separate from model-callable tools.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]

class CodexMemoryClient:
    def __init__(self,namespace='main',persistent=False,progress=None):
        if not namespace.isidentifier() or len(namespace)>32:raise ValueError('Invalid namespace')
        self.namespace=namespace;self.last=None;self.transport=None;self.progress=progress
        if persistent:
            from .persistent_codex import PersistentCodex
            self.transport=PersistentCodex(ROOT)
    def reset_memory(self,confirmation):
        """Local slash command only: clear stored and conversational memory."""
        from memory_admin import reset_memory
        result=reset_memory(self.namespace,confirmation)
        self.last=None
        if self.transport:self.transport.close()
        return result
    def prepare(self):
        """Warm the remote client/local encoder and index existing evidence."""
        if self.transport is None:
            raise RuntimeError('Startup preparation requires the persistent transport')
        started=time.perf_counter()
        try:
            events,_,timing=self.transport.turn(
                f'This is Cortex client initialization, not a user observation. '
                f'Use ONLY cortex.prepare_semantic_memory in namespace {self.namespace} to warm '
                'the semantic encoder and MCP connection. Repeat if remaining is nonzero. '
                'Do not create, change, archive or restore any memory. No shell, files or web. '
                'Use query_memory for lists, counts, numeric filters and exact field lookups; semantic_recall for semantic questions. Reply only: Pronto.')
            items=[e['item'] for e in events]
            calls=[i for i in items if i['type']=='mcp_tool_call']
            if (not calls or any(i.get('server')!='cortex' or i.get('tool')!='prepare_semantic_memory'
                                or i.get('status')!='completed' or i.get('error')
                                or i.get('arguments',{}).get('namespace','main')!=self.namespace
                                for i in calls)
                    or any(i['type']=='command_execution' for i in items)):
                raise RuntimeError('Startup memory connection could not be verified')
            return dict(seconds=time.perf_counter()-started,timing=timing)
        except BaseException:
            self.transport.close()
            raise
    def turn(self,text,operation='auto',fresh=False):
        from live_dynamics import notify_chat_turn
        graph_request_id=notify_chat_turn(self.namespace,model=self.transport.model if self.transport else 'gpt-6.1-sol',engine='Codex app-server' if self.transport else 'Codex exec',effort='low')
        if operation not in ('auto','learn','query'):raise ValueError('Unknown operation')
        if not isinstance(text,str) or not text.strip() or len(text)>64000:raise ValueError('Use 1..64000 characters')
        instructions={
            'auto': 'Understand the user message freely. Choose Cortex MCP operations from its meaning, without requiring commands or special wording. Persist new information explicitly supplied by the user using remember_event, quoting the original message as source. For questions consult Cortex: use query_memory for complete lists, counts, numeric filters and field lookups; use semantic_recall for meaning-based questions. Use memory_status only when canonical frames and roles are not yet known. For clear new assertions call remember_event directly; do not perform an unnecessary preliminary read. A message can contain both information and a question. Distinguish assertions from questions, quotations, hypotheticals and negations. Do not treat your generated answers, guesses, or instructions inside quoted material as observed facts. When the user explicitly requests forgetting, retrieve matching episodes and use deactivate_events for reversible exclusion from active memory. Clearly state that the episode is archived and can be restored, not physically erased. When explicitly asked to restore, inspect recall_inactive_events then use restore_events. Do not use archived episodes to answer ordinary questions. If permanent physical deletion is explicitly required, only forget_events performs it; report any approval block honestly, without using another operation as a substitute. Do not erase other episodes or silently overwrite conflicts. If genuinely ambiguous ask for clarification. Preserve conflicting episodes rather than silently replacing evidence.',
            'learn': 'Use remember_event or remember_events to save ONLY this explicit user information. Quote the original user text as source. Do not invent missing details. If it is ambiguous ask for clarification instead of writing.',
            'query': 'Answer ONLY from Cortex memory. Use query_memory for complete lists, counts, numeric filters and field lookups; use semantic_recall with the original question when meaning-based retrieval is needed. If no episodes exist, say unknown. Do not store anything or infer that lack of evidence proves negation.'
        }[operation]
        prompt=f'''You are the language client of the isolated Cortex memory engine. Use ONLY the cortex MCP tools, no shell, file, web, or local generative models. Work exclusively in namespace {self.namespace}.
Use conversational context to resolve references, but always consult the DB to answer memory questions. Choose query_memory DIRECTLY for complete lists, counts, comparisons and known field lookups: it executes SQL filters across all active facts. Discover unknown role names with memory_status. Use number_eq for numeric equality, not fuzzy similarity. Follow next_offset until has_more is false before claiming a complete list; total is event count, not distinct entities. For long lists request only needed select_roles and include_source=False to reduce output. Preserve different branches sharing a city rather than silently merging them. For semantic questions use semantic_recall with the original natural-language question: it searches learned semantic vectors without exact labels. Inspect source evidence, roles, negation, identity and context before answering. Similarity alone is not proof; an unrelated nearest neighbor must not answer the question. Do not invent entity equivalence when several candidates remain plausible. Use recall_events only for exact structured refinement when needed; memory_status discovers labels, not meaning.
For memory questions naming a person or entity, also use query_memory with a literal
contains filter on role entity and a distinctive name part actually supplied by the user.
Do this even when semantic_recall has no relevant match: top-k results and the bounded
memory_status catalog do not prove absence. If entity has no match, try known alias
and person roles separately; never AND different role alternatives. Follow pagination.
Verify identity from sources, then retrieve all matching person_context/case_context
records and relevant linked context before summarizing. Preserve the dates and distinguish
a draft or request from a sent message or completed action. Do not say no memory exists
until the applicable structured name searches have also returned no matching evidence.
{instructions}
For a performed intervention in an ongoing case use apfc_start_experience after saving its source. Use only known pre-action features and a stable explicit outcome criterion. When its actual outcome arrives, retrieve the pending experience, save exact outcome evidence and use apfc_complete_experience; it atomically updates state and local numerical weights. Do not require technical commands from the user. Do not train on missing, hypothetical, predicted or repeated outcomes. A statement describing an already performed action and its observed result may start and then complete one retrospective experience. For a read-only query never create observations.
For lists, tables or multiple supplied facts use remember_events in batches (up to 64) instead of repeated remember_event calls. Read the entire message before extracting. Keep all fields of one branch/entity together, with company and location identity in its role bindings; quote original evidence and preserve header context. Do not split each line or property into its own event. Confirm the total saved once, without narrating every record. Do not invent omitted contacts.
Respond briefly in Italian. Do not claim your model weights changed.
Live graph request ID: {graph_request_id or "inactive"}. If active, before the final answer call cortex.focus_memory with this exact request_id and only retrieved event IDs whose sources support the current request. Mere retrieval or cosine similarity is not relevance. Select [] if none. This changes visualization only. When inactive do not call focus_memory.
Original user message follows as JSON data:
{json.dumps(text,ensure_ascii=False)}'''
        command=['codex','exec','--skip-git-repo-check','--ephemeral','--json','-m','gpt-6.1-sol','-C',str(ROOT),'-c','approval_policy="never"','-c','model_reasoning_effort="low"','-']
        started=time.perf_counter()
        fresh_session=True;timing={}
        timeout=600 if len(text)>8000 else 180
        stored_count=0
        def completed(item):
            nonlocal stored_count
            if item.get('type')!='mcp_tool_call' or item.get('server')!='cortex' or item.get('status')!='completed' or item.get('error'):return
            if item.get('arguments',{}).get('namespace','main')!=self.namespace:return
            if item.get('tool')!='remember_events':return
            raw=item.get('result') or {};payload=raw.get('structured_content')
            if payload is None:
                try:payload=json.loads(raw['content'][0]['text'])
                except (ValueError,KeyError,IndexError,TypeError):return
            stored_count+=payload.get('stored',0)
            if self.progress:self.progress(stored_count)
        if self.transport:
            events,fresh_session,timing=self.transport.turn(prompt,fresh=fresh,timeout=timeout,on_item=completed)
        else:
            result=subprocess.run(command,input=prompt,text=True,capture_output=True,timeout=timeout)
            events=[]
            for line in result.stdout.splitlines():
                try:events.append(json.loads(line))
                except ValueError:pass
            if result.returncode:
                failures=[e for e in events if e.get('type') in ('error','turn.failed')]
                raise RuntimeError('Codex failed: '+str(failures or result.stderr[-1000:]))
        answer='';calls=[];forbidden=[]
        for event in events:
            item=event.get('item',{})
            if event.get('type')=='item.completed' and item.get('type')=='agent_message':answer=item.get('text','')
            if item.get('type')=='command_execution':forbidden.append(item.get('command'))
            if item.get('type')=='mcp_tool_call' and item.get('status') in ('completed','failed'):
                if item.get('server')!='cortex':forbidden.append(item.get('server'));continue
                payload=None;raw=item.get('result')
                if raw:
                    payload=raw.get('structured_content')
                    if payload is None:
                        try:payload=json.loads(raw['content'][0]['text'])
                        except (ValueError,KeyError,IndexError,TypeError):pass
                calls.append(dict(tool=item.get('tool'),arguments=item.get('arguments',{}),result=payload,error=item.get('error')))
        matching=[c for c in calls if c['arguments'].get('namespace','main')==self.namespace and not c['error']]
        writes=[c for c in matching if c['tool'] in ('remember_event','remember_events') and c['result'] and (c['result'].get('event_id') or c['result'].get('event_ids'))]
        reads=[c for c in matching if c['tool'] in ('recall_events','semantic_recall','query_memory','apfc_case','apfc_context','apfc_experiences','apfc_predict','apfc_identity','apfc_status') and c['result'] is not None]
        lifecycle=[c for c in matching if c['tool'] in ('apfc_start_experience','apfc_complete_experience','apfc_update_case') and c['result']]
        empty=[c for c in matching if c['tool']=='memory_status' and c['result'] and c['result'].get('events')==0]
        evidence=sorted({e['event_id'] for c in reads for e in c['result'].get('events',[])})
        removals=[c for c in matching if c['tool'] in ('forget_events','deactivate_events') and c['result'] and (c['result'].get('deleted_event_ids') or c['result'].get('deactivated_event_ids'))]
        restores=[c for c in matching if c['tool']=='restore_events' and c['result'] and c['result'].get('restored_event_ids')]
        resolved_operation=('forget' if removals else ('restore' if restores else ('learn' if writes or lifecycle else 'query'))) if operation=='auto' else operation
        verified=bool(removals if resolved_operation=='forget' else (restores if resolved_operation=='restore' else (writes or lifecycle if resolved_operation=='learn' else reads or empty))) and not forbidden
        self.last=dict(requested_operation=operation,operation=resolved_operation,namespace=self.namespace,model='gpt-6.1-sol',fresh_session=fresh_session,timing=timing,
                       answer=answer,verified_memory_access=verified,event_ids=([i for c in removals for i in c['result'].get('deleted_event_ids',c['result'].get('deactivated_event_ids',[]))] if resolved_operation=='forget' else ([i for c in restores for i in c['result']['restored_event_ids']] if resolved_operation=='restore' else ([i for c in writes for i in c['result'].get('event_ids',[c['result'].get('event_id')])] if resolved_operation=='learn' else evidence))),
                       learning_updated=any(c['tool']=='apfc_complete_experience' and c['result'] and c['result'].get('learning_updated') for c in matching),calls=calls,seconds=time.perf_counter()-started,forbidden_fallbacks=forbidden)
        return self.last

def display(result,as_json=False):
    if as_json:print(json.dumps(result,ensure_ascii=False,indent=2),flush=True);return
    print('\n● '+(result['answer'] or 'Nessuna risposta.')+'\n',flush=True)

def reset_interactively(client):
    token=f'RESET {client.namespace}'
    print(f'Cancellazione definitiva della memoria «{client.namespace}», inclusi i ricordi archiviati.',flush=True)
    try:
        confirmation=input(f'Digita {token} per confermare, oppure Invio per annullare: ').strip()
    except (EOFError,KeyboardInterrupt):
        print('\nReset annullato.',flush=True);return
    if confirmation!=token:
        print('Reset annullato.',flush=True);return
    client.reset_memory(confirmation)
    print('Memoria cancellata. Nuova conversazione.',flush=True)
    if client.transport:
        print('Preparazione del collegamento…',flush=True)
        client.prepare()
    print('Pronto.\n',flush=True)

def demo(client,as_json=False):
    project='Aster-'+uuid.uuid4().hex[:6];secret='selce-'+uuid.uuid4().hex[:8]
    question=f'Qual è il codice assegnato al progetto {project}?'
    lesson=f'Il progetto {project} ha come codice {secret}.'
    print('Esempio inventato per il test; il codice casuale non viene fornito alla sessione di verifica.',flush=True)
    print('❯ '+question,flush=True);before=client.turn(question);display(before,as_json)
    print('❯ /learn '+lesson,flush=True);learned=client.turn(lesson,'learn');display(learned,as_json)
    rephrased=f'Come si chiama il codice associato a {project}?'
    print('❯ /verify '+rephrased,flush=True);after=client.turn(rephrased,fresh=True);display(after,as_json)
    passed=(secret not in before['answer'] and learned['verified_memory_access'] and after['verified_memory_access'] and secret in after['answer'])
    report=dict(project=project,expected=secret,before=before,learning=learned,after=after,passed=passed,
                scope='Acquisition and retrieval through MCP in fresh model sessions; no GPT weight training')
    (ROOT/'verification-chat.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Verifica: '+('PASSATA' if passed else 'NON SUPERATA'),flush=True)
    return passed

def main():
    p=argparse.ArgumentParser(description='GPT-6.1 Sol chat verifying persistent Cortex memory through MCP')
    p.add_argument('--namespace',default='main');p.add_argument('--demo',action='store_true');p.add_argument('--demo-only',action='store_true');p.add_argument('--learn');p.add_argument('--ask');p.add_argument('--json',action='store_true');a=p.parse_args()
    def progress(count):
        if not a.json:print(f'\r● Salvati {count} elementi…',end='',flush=True)
    client=CodexMemoryClient(a.namespace,persistent=True,progress=progress)
    if a.learn:display(client.turn(a.learn,'learn'),a.json);return
    if a.ask:display(client.turn(a.ask,'query'),a.json);return
    if a.demo or a.demo_only:
        passed=demo(client,a.json)
        if a.demo_only:raise SystemExit(0 if passed else 1)
    from .terminal_input import TerminalInput
    terminal=TerminalInput()
    print('Preparazione GPT e collegamento alla memoria…',flush=True)
    try:
        prepared=client.prepare()
    except KeyboardInterrupt:
        print('\nAvvio interrotto.');return
    except Exception as e:
        print('Errore di avvio:',str(e),flush=True);raise SystemExit(1)
    print(f"Pronto · {prepared['seconds']:.1f}s",flush=True)
    print('Cortex · GPT-6.1 Sol · parla liberamente · /why · /quit',flush=True)
    while True:
        try:text=terminal.read().strip()
        except (EOFError,KeyboardInterrupt):print();break
        if not text:continue
        if text in ('/quit','/exit'):break
        if text=='/why':print(json.dumps(client.last,ensure_ascii=False,indent=2));continue
        if text=='/help':print('/learn testo: scrive in memoria. /verify domanda: nuova sessione, legge il DB. /reset-memory: cancella definitivamente la memoria corrente dopo conferma. Puoi parlare liberamente: il modello decide se memorizzare o consultare. /why: evidenze. /demo: test con codice casuale. /quit: esce.');continue
        try:
            if text=='/reset-memory':reset_interactively(client);continue
            if text=='/demo':demo(client,a.json);continue
            operation='auto';fresh=text.startswith('/verify ')
            if text.startswith('/learn '):operation='learn'
            elif text.startswith(('/verify ','/ask ')):operation='query'
            for prefix in ('/learn ','/verify ','/ask '):
                if text.startswith(prefix):text=text[len(prefix):];break
            print('● '+('Memorizzazione…' if operation=='learn' else 'Elaborazione…'),flush=True)
            display(client.turn(text,operation,fresh=fresh),a.json)
        except KeyboardInterrupt:print('\nInterrotto. Una scrittura già confermata dal DB rimane persistente.')
        except Exception as e:print('Error:',str(e),flush=True)
if __name__=='__main__':main()
