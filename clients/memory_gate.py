"""Mandatory read-only MCP retrieval before language generation."""
import asyncio
import os
import json
from pathlib import Path
import subprocess
import sys

# These are stored schema roles, never a parser of the user's language.
CASE_ROLES={'case','case_id','practice','practice_id','project','project_id','thread_id','subject','topic'}
MAX_LINK_QUERIES=4
MAX_PAGES=2
MAX_EVENTS=18
MAX_CONTEXT_CHARS=24000

async def assemble(call,namespace,query):
    seed=await call('semantic_recall',{'namespace':namespace,'query':query[:8000],'limit':6})
    seeds=seed.get('events',[])
    anchors=[]
    for priority in (CASE_ROLES,{'entity'}):
        for event in seeds:
            for binding in event.get('roles',[]):
                key=(binding['role'],binding['value'])
                if key[0] in priority and key not in anchors:
                    anchors.append(key)
    anchors=anchors[:MAX_LINK_QUERIES]
    found={e['event_id']:e for e in seeds}
    searches=[]
    for role,value in anchors:
        offset=0
        for page in range(MAX_PAGES):
            result=await call('query_memory',{'namespace':namespace,
                'filters':[{'role':role,'operator':'eq','value':value}],
                'negated':None,'limit':25,'offset':offset,'include_source':True})
            searches.append({'role':role,'value':value,'total':result['total'],
                             'offset':offset,'has_more':result['has_more']})
            for event in result.get('events',[]):found[event['event_id']]=event
            if not result['has_more']:break
            offset=result['next_offset']
    # Preserve semantic seeds and add later linked records, where follow-up
    # observations often live. IDs are insertion order, not event chronology.
    selected=list(dict.fromkeys([e['event_id'] for e in seeds]+sorted(found,reverse=True)))[:MAX_EVENTS]
    candidates=[];used=0
    for ident in selected:
        e=found[ident];source=e.get('source','')
        excerpt=source if len(source)<=1200 else source[:800]+'\n[excerpt omitted]\n'+source[-400:]
        item=dict(event_id=ident,frame=e['frame'],negated=e['negated'],roles=e['roles'],
                  source=excerpt,source_truncated=len(source)>1200,time=e.get('time'),place=e.get('place'))
        size=len(json.dumps(item,ensure_ascii=False))
        if used+size>MAX_CONTEXT_CHARS-8000:break
        candidates.append(item);used+=size
    operational={'cases':[],'has_more':False}
    if found:
        operational=await call('apfc_context',{'namespace':namespace,'event_ids':selected[:32],'limit':3})
    states=[];state_chars=0;state_omitted=0
    for state in operational.get('cases',[]):
        size=len(json.dumps(state,ensure_ascii=False))
        if used+state_chars+size>MAX_CONTEXT_CHARS:
            state_omitted+=1;continue
        states.append(state);state_chars+=size
    return {'operational_states':states,
            'operational_state_coverage':{'linked_cases_truncated':operational.get('has_more',False),
                'omitted_for_budget':state_omitted,'state_is_derived':True},
            'namespace':namespace,'retrieval':'semantic_seeds_and_exact_linked_evidence',
            'query_truncated':len(query)>8000,'candidates':candidates,'linked_searches':searches,
            'coverage':{'complete_case_state':False,'discovered_events':len(found),
                        'included_events':len(candidates),'omitted_events':len(found)-len(candidates),
                        'bounded_expansion':True},
            'required_checks':['Identify what was requested, already verified, actually sent, answered and still open.',
                'A draft or proposed action is not a sent message or completed action.',
                'Before proposing to repeat an action, find its prior execution and subsequent response; if unavailable, state not verified.',
                'Every quantity needs its source, date, unit and population; never merge counts with different scopes.',
                'Missing evidence in these candidates does not establish that an action or response is absent.',
                'For current external status, inspect the original source and follow-up messages before concluding.',
                'Check experience_summary for pending interventions. On explicit outcome evidence close the matching experience; missing outcomes remain pending. Never treat these retrieved records as new observations.']}

async def recall(root, namespace, query):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    parameters=StdioServerParameters(command=str(root/'.venv/bin/python'),args=[str(root/'mcp_server.py')],env={**os.environ,'CORTEX_LIVE_ORIGIN':'gate'})
    async with stdio_client(parameters) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            async def call(name,args):
                result=await session.call_tool(name,args)
                if result.isError:raise RuntimeError('Memory MCP retrieval failed: '+name)
                return result.structuredContent or json.loads(result.content[0].text)
            assembled=await assemble(call,namespace,query)
            from live_dynamics import notify_memory_selection
            notify_memory_selection(namespace,assembled['candidates'])
            return assembled

def retrieve(root,namespace,text):
    result=subprocess.run([str(root/'.venv/bin/python'),'-m','clients.memory_gate',namespace],
                          cwd=root,input=text,text=True,capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError('Richiamo obbligatorio Cortex fallito: richiesta non inviata al modello. Nessun fallback silenzioso.')
    return json.loads(result.stdout)

if __name__=='__main__':
    print(json.dumps(asyncio.run(recall(Path(__file__).resolve().parents[1],sys.argv[1],sys.stdin.read())),ensure_ascii=False))
