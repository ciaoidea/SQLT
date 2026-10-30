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

async def assemble(call,namespace,query,seed=None):
    expanded=seed is not None
    if seed is None:
        seed=await call('semantic_recall',{'namespace':namespace,'query':query[:8000],'limit':6})
    seeds=seed.get('events',[])
    anchors=[]
    for priority in (CASE_ROLES,{'entity'}):
        for event in seeds:
            for binding in event.get('roles',[]):
                key=(binding['role'],binding['value'])
                if key[0] in priority and key not in anchors:
                    anchors.append(key)
    anchors=[] if expanded else anchors[:MAX_LINK_QUERIES]
    found={e['event_id']:e for e in seeds}
    searches=seed.get('linked_searches',[]) if expanded else []
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
                  source=excerpt,source_truncated=e.get('source_truncated',False) or len(source)>1200,time=e.get('time'),place=e.get('place'))
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
    return {'evidence_contract':seed.get('evidence_contract',{}),
            'retrieval_plan':seed.get('plan'), 'query_outcome':seed.get('outcome'),
            'clarification':seed.get('clarification',''), 'retrieval_coverage':seed.get('coverage',{}),
            'operational_states':states,
            'operational_state_coverage':{'linked_cases_truncated':operational.get('has_more',False),
                'omitted_for_budget':state_omitted,'state_is_derived':True},
            'namespace':namespace,'retrieval':seed.get('retrieval','semantic_seeds_and_exact_linked_evidence'),
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

async def _session_context(session, root, namespace, query, verify_origin=False):
    async def call(name,args):
        result=await session.call_tool(name,args)
        if result.isError:raise RuntimeError('Memory MCP retrieval failed: '+name)
        data=result.structuredContent or json.loads(result.content[0].text)
        if isinstance(data,dict) and isinstance(data.get('result'),dict):data=data['result']
        return data
    config_path=root/'.nlq.json'
    config=json.loads(config_path.read_text()) if config_path.exists() else {}
    nlq_configured=config_path.exists() or bool(os.environ.get('CORTEX_NLQ_SOCKET'))
    nlq_namespace=config.get('namespace',os.environ.get('CORTEX_NLQ_NAMESPACE','main'))
    strategy='natural_memory_context' if nlq_configured and namespace==nlq_namespace else 'memory_context'
    hybrid=await call(strategy,{'namespace':namespace,'query':query[:8000]})
    if verify_origin:
        import hashlib
        expected={'installation':hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:24],'namespace':namespace}
        if hybrid.get('memory_origin')!=expected:
            raise RuntimeError('HTTP MCP origin mismatch; refusing another memory installation')
    assembled=await assemble(call,namespace,query,seed={**hybrid,'events':hybrid['candidates']})
    from live_dynamics import notify_memory_selection
    notify_memory_selection(namespace,assembled['candidates'])
    return assembled


async def recall(root, namespace, query):
    from mcp import ClientSession, StdioServerParameters
    config_path=root/'.recall-http.json'
    if config_path.exists():
        import httpx
        from mcp.client.streamable_http import streamable_http_client
        config=json.loads(config_path.read_text())
        token_path=Path(config['token_file'])
        for path in (config_path,token_path):
            if path.stat().st_mode & 0o077:raise ValueError('MCP client configuration/token must be owner-only')
        token=token_path.read_text().strip()
        if len(token)<32:raise ValueError('Invalid MCP token')
        async with httpx.AsyncClient(headers={'Authorization':'Bearer '+token},timeout=105) as client:
            async with streamable_http_client(config['url'],http_client=client) as (rx,tx,_):
                async with ClientSession(rx,tx) as session:
                    await session.initialize()
                    return await _session_context(session,root,namespace,query,True)
    from mcp.client.stdio import stdio_client
    parameters=StdioServerParameters(command=str(root/'.venv/bin/python'),args=[str(root/'mcp_server.py')],env={**os.environ,'CORTEX_LIVE_ORIGIN':'gate'})
    async with stdio_client(parameters) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            return await _session_context(session,root,namespace,query)


def retrieve(root,namespace,text):
    if (root/'.recall-http.json').exists():
        try:asyncio.get_running_loop()
        except RuntimeError:
            # Native synchronous clients reuse the already-running MCP server;
            # no per-turn Python subprocess or model planner is started.
            try:return asyncio.run(asyncio.wait_for(recall(root,namespace,text),timeout=110))
            except Exception:raise RuntimeError('Richiamo obbligatorio Cortex fallito: richiesta non inviata al modello. Nessun fallback silenzioso.') from None
    result=subprocess.run([str(root/'.venv/bin/python'),'-m','clients.memory_gate',namespace],
                          cwd=root,input=text,text=True,capture_output=True,timeout=110 if (root/'.nlq.json').exists() else 60)
    if result.returncode:raise RuntimeError('Richiamo obbligatorio Cortex fallito: richiesta non inviata al modello. Nessun fallback silenzioso.')
    return json.loads(result.stdout)

if __name__=='__main__':
    print(json.dumps(asyncio.run(recall(Path(__file__).resolve().parents[1],sys.argv[1],sys.stdin.read())),ensure_ascii=False))


def agent_policy(namespace,query_tool="query_memory"):
    """Shared memory work protocol used by the native shell and MCP clients."""
    policy=f'''Persistent knowledge is exclusively Cortex MCP, namespace {namespace}. Never create a second
Markdown, JSON or SQLite knowledge store, or load legacy APFC/MD-OS memory.
For memory questions consult Cortex even if the answer appears in conversation. Use query_memory
for exact filters, counts and exhaustive lists, following pagination; semantic_recall for meaning.
For memory questions naming a person or entity, also use query_memory with a literal
contains filter on role entity and a distinctive name part actually supplied by the user.
Do this even when semantic_recall has no relevant match: top-k results and the bounded
memory_status catalog do not prove absence. If entity has no match, try known alias
and person roles separately; never AND different role alternatives. Follow pagination.
Verify identity from sources, then retrieve all matching person_context/case_context
records and relevant linked context before summarizing. Preserve the dates and distinguish
a draft or request from a sent message or completed action. Do not say no memory exists
until the applicable structured name searches have also returned no matching evidence.
Preserve evidence, identity, negation and dates. Similarity is not proof. Store new explicit user
facts with remember_events, quoting the original evidence; keep related entity fields together.
Do not save generated guesses or ordinary task instructions as knowledge. Forgetting follows
Cortex MCP archival/deletion semantics. Never reset a memory namespace without explicit confirmation.
For ongoing work, reconstruct the case from sourced evidence BEFORE conclusions or next actions:
request, completed checks, actually sent communications, responses and unresolved questions.
Distinguish drafts, proposals and requests from actual execution. Consult original external
sources for current status. Do not propose repeating a completed action without checking its
outcome and subsequent replies. If these cannot be checked, explicitly say state unverified.
For every quantity preserve its population, unit, date and source; different scopes are not
interchangeable. The automatic retrieval is bounded evidence, NOT a complete case history.
Use exact case/entity queries and pagination to fill gaps. Never claim no reply based on top-k.
Use tools to verify actual changes. Keep replies clean: no internal DB IDs or trace JSON unless asked.
For memory lookups, return only the answer. Do not announce searching, consulting
the database or using memory, and do not send interim commentary for these lookups.
'''
    if query_tool not in ('query_memory','query_facts'):raise ValueError('Unknown query interface')
    return policy.replace('query_memory',query_tool)


async def agent_context(call,namespace,query,seed,max_chars=9000):
    context=await assemble(call,namespace,query,seed={**seed,'events':seed['candidates']})
    for event in context['candidates']:
        source=event.get('source','')
        event['source']=source[:150]
        event['source_truncated']=event.get('source_truncated',False) or len(source)>150
    while len(json.dumps(context,ensure_ascii=False))>max_chars:
        if context['operational_states']:
            context['operational_states'].pop()
            context['operational_state_coverage']['omitted_for_budget']+=1
        elif context['candidates']:
            context['candidates'].pop()
            context['coverage']['included_events']-=1
            context['coverage']['omitted_events']+=1
        else:raise ValueError('Agent context metadata exceeds budget')
    context['memory_origin']=seed['memory_origin']
    context['memory_protocol']=agent_policy(namespace)
    return context
