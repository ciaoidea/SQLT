"""Loopback-only, read-only visual query browser using the existing Cortex MCP API."""
import argparse
import asyncio
from concurrent.futures import TimeoutError
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
from planner import plan, PlanCache
from urllib.parse import urlsplit, parse_qs

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from live_dynamics import LiveCollector, running_chat_models
from query_flow import Flow, progress, _sink
ROLES = ('entity', 'person', 'company', 'customer', 'provider', 'agent', 'patient',
         'recipient', 'branch', 'site', 'city', 'contact', 'collaborator', 'alias',
         'project', 'product', 'service', 'team', 'department', 'place', 'società', 'entità')
EXPAND_ROLES = ('entity', 'person', 'company', 'customer', 'provider', 'agent',
                'recipient', 'branch', 'site', 'contact', 'alias', 'project')


def identity(label):
    return 'v-' + hashlib.sha256(label.casefold().strip().encode()).hexdigest()[:24]


def graph(events, semantic_query=None, metadata=None, cap=40):
    nodes, edges = {}, []
    if semantic_query:
        nodes['query'] = {'data': {'id': 'query', 'label': semantic_query, 'kind': 'query'}}
    for event in events[:cap]:
        changed=[];first_edge=len(edges)
        event_id = 'e-' + str(event['event_id'])
        nodes[event_id] = {'data': {'id': event_id, 'label': event['frame'].replace('_', ' '),
                                  'kind': 'event', 'negated': bool(event.get('negated')), 'event': event}}
        changed.append(nodes[event_id])
        if semantic_query:changed.append(nodes['query'])
        for binding in event.get('roles', []):
            if binding['role'] not in ROLES:
                continue
            label = binding['value']
            node_id = identity(label)
            nodes[node_id] = {'data': {'id': node_id, 'label': label, 'kind': 'entity'}}
            changed.append(nodes[node_id])
            edges.append({'data': {'id': event_id + '-' + binding['role'], 'source': event_id,
                                   'target': node_id, 'label': binding['role'], 'kind': 'binding'}})
        if semantic_query:
            similarity = event.get('similarity')
            edges.append({'data': {'id': 's-' + event_id, 'source': 'query', 'target': event_id,
                                   'label': f'{similarity:.2f}' if similarity is not None else 'similarità',
                                   'kind': 'similarity'}})
        progress('bind',changed+edges[first_edge:],event_id=event['event_id'])
    return {'elements': list(nodes.values()) + edges, 'events': len(events[:cap]), **(metadata or {})}


def grouped_graph(events, roles):
    roles = [roles] if isinstance(roles, str) else roles
    if not roles or len(roles) > 4 or len(set(roles)) != len(roles):
        raise ValueError('Use one to four distinct grouping levels')
    nodes, edges, branches, groups = {}, {}, {}, {}
    for event in events:
        bindings = {b['role']: b['value'] for b in event['roles']}
        label = bindings.get('entity') or bindings.get('person') or str(event['event_id'])
        fields = bindings.get('entity', '').split(' | ')
        derived = dict(zip(('company', 'city', 'region', 'address'), fields)) if len(fields) == 4 and event['frame'] == 'branch_connectivity' else {}
        path = tuple(bindings.get(role) or derived.get(role) or 'Not specified' for role in roles)
        duplicate=(path,label) in branches
        branches.setdefault((path, label), event)
        progress('deduplicate' if duplicate else 'classify',event_id=event['event_id'],groups=list(path),group_roles=roles)
        for depth in range(len(roles)):
            groups.setdefault(path[:depth+1], set()).add(label)
    def gid(path):
        return 'group-' + identity(json.dumps(list(zip(roles, path)), ensure_ascii=False))
    shown = 0
    for (path, label), event in branches.items():
        paths = [path[:i+1] for i in range(len(roles))]
        needed = sum(gid(part) not in nodes for part in paths) + 1
        if shown >= 280 or len(nodes) + needed > 360:
            continue
        parent = None
        changed=[];changed_edges=[]
        for depth, part in enumerate(paths):
            ident = gid(part)
            nodes[ident] = {'data': {'id': ident, 'label': f'{part[-1]} ({len(groups[part])})',
                            'kind': 'group', 'group_role': roles[depth], 'group_value': part[-1],
                            'group_level': depth, 'group_path': list(part),
                            'members_count': len(groups[part])}}
            changed.append(nodes[ident])
            if parent:
                edges[ident] = {'data': {'id': 'hierarchy-' + ident, 'source': parent,
                               'target': ident, 'label': roles[depth], 'kind': 'binding'}}
                changed_edges.append(edges[ident])
            parent = ident
        ident = 'member-' + identity(json.dumps([path, label], ensure_ascii=False))
        nodes[ident] = {'data': {'id': ident, 'label': label, 'kind': 'member', 'event': event}}
        edges[ident] = {'data': {'id': 'group-link-' + ident, 'source': parent,
                       'target': ident, 'label': 'member', 'kind': 'binding'}}
        changed.append(nodes[ident]);changed_edges.append(edges[ident])
        progress('group_join',changed+changed_edges,event_id=event['event_id'],group_roles=roles)
        shown += 1
    return {'elements': list(nodes.values()) + list(edges.values()), 'events': len(events),
            'entities': len(branches), 'groups': len(groups), 'group_levels': roles,
            'view_partial': shown < len(branches)}


def object_graph(records,dataset,groups=None):
    """Graph exact stored object links; no inferred semantic relations."""
    nodes={};edges=[];groups=groups or []
    for row in records[:150]:
        changed=[];edge_start=len(edges)
        key=str(row.get('id'));node='object-'+dataset+'-'+key
        label=str(row.get('title') or row.get('experience_key') or row.get('policy_key') or row.get('label') or row.get('case_key') or key)
        nodes[node]={'data':dict(id=node,label=label,kind='object',dataset=dataset,record=row)}
        changed.append(nodes[node])
        parent=None;path=[]
        for level,field in enumerate(groups):
            value=str(row.get(field,'Not recorded'));path.append([field,value]);ident='group-'+identity(json.dumps(path))
            if ident not in nodes:nodes[ident]={'data':dict(id=ident,label=value,kind='group',group_role=field,group_value=value,group_level=level,members_count=0)}
            nodes[ident]['data']['members_count']+=1
            changed.append(nodes[ident])
            target=node if level==len(groups)-1 else None
            if parent:edges.append({'data':dict(id=parent+'-'+ident,source=parent,target=ident,label=field,kind='grouping')})
            parent=ident
        if parent:edges.append({'data':dict(id=parent+'-'+node,source=parent,target=node,label='member',kind='grouping')})
        if dataset=='experiences' and not groups:
            for field,prefix in [('case_key','case'),('policy_key','policy')]:
                value=row[field];ident=prefix+'-'+identity(value)
                linked=dict(case_key=value) if prefix=='case' else dict(policy_key=value)
                nodes[ident]={'data':dict(id=ident,label=value,kind='object',dataset='cases' if prefix=='case' else 'policies',record=linked)}
                changed.append(nodes[ident])
                edges.append({'data':dict(id=node+'-'+prefix,source=node,target=ident,label=prefix,kind='binding')})
        progress('object_join',changed+edges[edge_start:])
    return dict(elements=list(nodes.values())+edges,events=len(records),view_partial=len(records)>150)


def compact_events(events):
    result=[]
    for event in events[:200]:
        record=dict(event);source=record.get('source','')
        record['source_truncated']=len(source)>1200
        if record['source_truncated']:record['source']=source[:1200]+'…'
        result.append(record)
    return result



class MemoryBridge:
    def __init__(self, namespace):
        self.namespace = namespace
        self.traces = []
        self.plan_cache = PlanCache()
        self.ready = threading.Event()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        if not self.ready.wait(30):
            raise RuntimeError('MCP startup timed out')
        if getattr(self, 'error', None):
            raise RuntimeError('Cortex MCP startup failed') from self.error

    def _worker(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect())

    async def _connect(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        try:
            async with stdio_client(StdioServerParameters(command=sys.executable,
                    args=[str(ROOT / 'mcp_server.py')], cwd=str(ROOT), env={**os.environ,'CORTEX_LIVE_ORIGIN':'viewer'})) as io:
                async with ClientSession(*io) as self.client:
                    await self.client.initialize()
                    self.lock = asyncio.Lock()
                    self.planning_lock = asyncio.Semaphore(1)
                    self.stop = asyncio.Event()
                    self.ready.set()
                    await self.stop.wait()
        except BaseException as error:
            self.error = error
            self.ready.set()

    async def _call(self, name, args):
        if name not in ('query_memory','semantic_recall','literal_memory_candidates','agent_memory_context','memory_status','apfc_browse','apfc_case','apfc_status','memory_event'):
            raise ValueError('Read-only tool boundary')
        if name in ('query_memory','semantic_recall','apfc_browse'):
            args = {**args, 'include_sql': True}
        progress('retrieve_start',operation=name)
        result = await self.client.call_tool(name, {**args, 'namespace': self.namespace})
        if result.isError:
            message=next((c.text for c in result.content if c.type=='text'), 'Memory query failed')
            raise RuntimeError(message[:600])
        data = json.loads(next(c.text for c in result.content if c.type == 'text'))
        progress('retrieved',operation=name,returned=len(data.get('events',data.get('records',[]))),matched=data.get('total',data.get('count',0)))
        for index, trace in enumerate(data.get('sql_trace', [])):
            self.traces.append({**trace, 'seconds': data.get('sql_seconds', 0) if index == 0 else 0})
        return data

    async def _catalog(self):
        if getattr(self, 'catalog_time', 0) > time.monotonic() - 120:
            return self.catalog
        frames, roles, frame_roles, examples = set(), set(), {}, {}
        frame_counts, field_counts = {}, {}
        offset = 0
        while True:
            sample = await self._call('query_memory', {'filters': [], 'offset': offset,
                       'limit': 100, 'negated': None, 'include_source': False})
            for event in sample['events']:
                frame = event['frame']; frames.add(frame)
                frame_counts[frame]=frame_counts.get(frame,0)+1
                for present in {b['role'] for b in event['roles']}:
                    key=(frame,present);field_counts[key]=field_counts.get(key,0)+1
                fields = frame_roles.setdefault(frame, set())
                for binding in event['roles']:
                    role, value = binding['role'], binding['value']
                    roles.add(role); fields.add(role)
                    if role not in ROLES and value.isalpha() and len(value) <= 20:
                        values = examples.setdefault(frame, {}).setdefault(role, set())
                        if len(values) < 8: values.add(value)
            if not sample['has_more'] or sample['next_offset'] >= 2000: break
            offset = sample['next_offset']
        self.catalog = {'frames': sorted(frames), 'roles': sorted(roles),
                        'frame_roles': {k: sorted(v) for k, v in frame_roles.items()},
                        'field_coverage':{f:{r:round(field_counts[(f,r)]/frame_counts[f],3) for r in rs}
                                          for f,rs in frame_roles.items()},
                        'catalog_complete':not sample['has_more'],
                        'categorical_examples': {f: {r: sorted(v) for r, v in fields.items()}
                                                 for f, fields in examples.items()}}
        self.catalog_time = time.monotonic()
        return self.catalog

    async def _natural(self, query, specification=None, planning_seconds=None):
        if specification is None:
            catalog = await self._catalog()
            specification, planning_seconds = await asyncio.to_thread(plan, query, catalog)
        self.traces = []  # Display only executed query SQL, not catalog discovery.
        if specification['mode'] == 'clarify':
            return {'elements': [], 'events': 0, 'partial': False, 'plan': specification,
                    'title': specification['title'], 'note': specification['clarification'], 'requested_query': query,
                    'sql_trace': [], 'planning_seconds': planning_seconds}
        dataset=specification.get('dataset','memory')
        if dataset!='memory':
            if specification['mode']!='relational' or len(specification['queries'])>1:
                raise ValueError('APFC object browsing supports one exact relational selection')
            selection=(specification['queries'] or [{}])[0]
            result=await self._table_unlocked(dataset,filters=selection.get('filters',[]),limit=50)
            result.update(object_graph(result['records'],dataset,specification['group_by']),plan=specification,title=specification['title'],planning_seconds=planning_seconds,requested_query=query)
            return result
        events, ids, partial = [], set(), False
        specifications = specification['queries'] or [{'frame': None, 'filters': []}]
        for selection in specifications:
            if specification['mode'] in ('semantic', 'hybrid'):
                result = await self._call('semantic_recall', {**selection,
                         'query': specification['semantic_query'] or query, 'limit': 16, 'negated': False})
                partial = True
                rows = result['events']
            else:
                rows, offset = [], 0
                while True:
                    result = await self._call('query_memory', {**selection, 'negated': False,
                         'limit': 100, 'offset': offset, 'include_source': True})
                    rows.extend(result['events'])
                    if not result['has_more']: break
                    if len(rows) >= 1000:
                        partial = True; break
                    offset = result['next_offset']
            for event in rows:
                if event['event_id'] not in ids:
                    ids.add(event['event_id']); events.append(event)
        role = specification['group_by']
        if role:
            # A company is one visual hub, not one duplicate per outer group.
            visual_roles = (['company'] + [r for r in role if r != 'company']) if 'company' in role else role
            result = grouped_graph(events, visual_roles)
            result['requested_group_levels'] = role
            partial |= result['view_partial']
        else:
            result = graph(events, (specification['semantic_query'] or query)
                           if specification['mode'] in ('semantic', 'hybrid') else None, cap=280)
            partial |= len(events) > 280
        result.update(records=compact_events(events[:50]),records_partial=len(events)>50,dataset='memory',method=specification['mode'], plan=specification, title=specification['title'], requested_query=query,
                      partial=partial, matched_events=len(events), sql_trace=self.traces,
                      planning_seconds=planning_seconds,
                      note=('Risultati parziali: limite di recupero o visualizzazione.' if partial else
                            'Tutti i record corrispondenti recuperati. Conteggi delle entità distinti dai record.'))
        return result

    async def _browse(self, mode, query):
        if mode=='natural':
            progress('planning')
            started=time.perf_counter()
            async with self.lock:
                catalog=await self._catalog()
                hints=await self._call('literal_memory_candidates',{'query':query})
                # Per-question evidence, not a mutation of the shared schema catalog.
                context=await self._call('agent_memory_context',{'query':query})
                catalog={**catalog,'literal_query_evidence':hints.get('events',[]),
                    'agent_context':{k:context[k] for k in ('candidates','operational_states','required_checks') if k in context},
                    'memory_protocol':context.get('memory_protocol','')}
            catalog_seconds=time.perf_counter()-started
            queued=time.perf_counter()
            async with self.planning_lock:
                queue_seconds=time.perf_counter()-queued
                cached=self.plan_cache.get(query,catalog)
                if cached is None:
                    specification,seconds=await asyncio.to_thread(plan,query,catalog)
                    self.plan_cache.put(query,catalog,specification)
                    cache_info={'hit':False,'age_seconds':0,'ttl_seconds':self.plan_cache.ttl}
                else:
                    specification,age=cached;seconds=0.
                    cache_info={'hit':True,'age_seconds':round(age,3),'ttl_seconds':self.plan_cache.ttl}
            progress('plan_ready',group_by=specification.get('group_by',[]),mode=specification.get('mode',''))
            execution_started=time.perf_counter()
            try:
                async with self.lock:result=await self._natural(query,specification,seconds)
            except BaseException:
                self.plan_cache.discard(query,catalog)
                raise
            result['plan_cache']=cache_info
            result['retrieval_timing']={'catalog_seconds':round(catalog_seconds,4),
                                       'planner_queue_seconds':round(queue_seconds,4),
                                       'planning_seconds':round(seconds,4),
                                       'execution_seconds':round(time.perf_counter()-execution_started,4)}
            return result
        async with self.lock:
            self.traces = []
            if mode == 'semantic':
                if not query: raise ValueError('Enter a search phrase')
                result = await self._call('semantic_recall', {'query': query, 'limit': 12})
                return graph(result['events'], query, {'records':compact_events(result['events']),'dataset':'memory','method': 'semantic', 'partial': True,
                            'sql_trace': self.traces, 'note': 'Somiglianza semantica: candidati, non relazioni dimostrate.'})
            if mode == 'initial':
                count = await self._call('query_memory', {'filters': [], 'negated': None,
                                         'count_only': True})
                selected = []
                for frame, limit in [('contact_profile', 16), ('person_context', 8)]:
                    result = await self._call('query_memory', {'filters': [], 'frame': frame,
                                'negated': None, 'limit': limit, 'include_source': True})
                    selected.extend(result['events'])
                if not selected:
                    result = await self._call('query_memory', {'filters': [], 'negated': None,
                                         'limit': 24, 'include_source': True})
                    selected = result['events']
                return graph(selected, metadata={'records':compact_events(selected),'dataset':'memory','total': count['total'],
                            'partial': len(selected) < count['total'], 'method': 'initial', 'sql_trace': self.traces,
                            'note': 'Campione di profili e contesti collegati dai ruoli registrati. Cerca per esplorare il resto.'})
            if not query: raise ValueError('Enter a name')
            found, seen, truncated = [], set(), False
            roles = EXPAND_ROLES if mode == 'expand' else ('entity', 'person', 'company', 'alias')
            for role in roles:
                result = await self._call('query_memory', {'filters': [{'role': role,
                            'operator': 'eq' if mode == 'expand' else 'contains', 'value': query}],
                            'negated': None, 'limit': 40, 'include_source': True})
                truncated |= result['has_more']
                for event in result['events']:
                    if event['event_id'] not in seen:
                        seen.add(event['event_id']); found.append(event)
                if len(found) >= 40:
                    truncated = True
                    break
            return graph(found, metadata={'records':compact_events(found),'dataset':'memory','method': mode, 'partial': truncated, 'sql_trace': self.traces,
                         'note': 'Ruoli documentati nei fatti. Massimo 40 fatti per richiesta.'})

    async def _table_unlocked(self,dataset,frame=None,filters=None,limit=50,offset=0):
        if dataset=='memory':
            result=await self._call('query_memory',dict(filters=filters or [],frame=frame,negated=None,limit=limit,offset=offset,include_source=True))
            records=compact_events(result['events']);result['records']=records
            result.update(graph(result['events'],cap=100),dataset='memory',records=records,title=frame or 'Memory events',partial=result['has_more'],sql_trace=self.traces)
        else:
            result=await self._call('apfc_browse',dict(dataset=dataset,filters=filters or [],limit=limit,offset=offset))
            result.update(object_graph(result['records'],dataset),title=dataset.title(),partial=result['has_more'],sql_trace=self.traces)
        return result

    async def _workbench(self,kind,options):
        async with self.lock:
            self.traces=[]
            if kind=='catalog':
                catalog=await self._catalog()
                status=await self._call('apfc_status',{})
                return dict(**catalog,apfc=status,datasets=['memory','cases','experiences','policies','entities'])
            if kind=='detail':
                if options['dataset']=='memory':return await self._call('memory_event',{'event_id':int(options['key'])})
                if options['dataset']=='cases':return await self._call('apfc_case',{'case_key':options['key']})
                if options['dataset']=='policies':
                    data=await self._call('apfc_browse',dict(dataset='policies',filters=[dict(role='policy_key',value=options['key'])],limit=1))
                    return data['records'][0] if data['records'] else {'status':'unknown'}
                raise ValueError('Unsupported detail dataset')
            return await self._table_unlocked(**options)

    async def _with_flow(self, coroutine, sink):
        token=_sink.set(sink)
        try:return await coroutine
        finally:_sink.reset(token)

    def workbench(self,kind,options=None):
        future=asyncio.run_coroutine_threadsafe(self._with_flow(self._workbench(kind,options or {}),_sink.get()),self.loop)
        try:return future.result(120)
        except TimeoutError:
            future.cancel();raise RuntimeError('Memory request timed out') from None

    def browse(self, mode, query):
        future = asyncio.run_coroutine_threadsafe(self._with_flow(self._browse(mode, query),_sink.get()), self.loop)
        try:
            return future.result(120)
        except TimeoutError:
            future.cancel()
            raise RuntimeError('Memory request timed out') from None

    def close(self):
        if hasattr(self, 'stop'):
            self.loop.call_soon_threadsafe(self.stop.set)
            self.thread.join(5)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Do not log private search terms or returned records.

    def send(self, status, body, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def stream_query(self, operation):
        flow=Flow()
        worker=threading.Thread(target=flow.execute,args=(operation,),daemon=True,name='cortex-query-flow')
        self.send_response(200)
        self.send_header('Content-Type','application/x-ndjson; charset=utf-8')
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Accel-Buffering','no')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Connection','close')
        self.end_headers()
        self.close_connection=True
        worker.start()
        try:
            for item in flow.frames():
                self.wfile.write((json.dumps(item,ensure_ascii=False)+'\n').encode())
                self.wfile.flush()
        except (BrokenPipeError,ConnectionResetError,OSError):
            pass
        finally:flow.closed=True

    def do_GET(self):
        expected = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        if self.headers.get('Host') not in expected:
            return self.send(403, b'Local access only', 'text/plain')
        url = urlsplit(self.path)
        files = {'/': 'map.html', '/map.html': 'map.html', '/map.js': 'map.js',
                 '/map.css': 'map.css', '/live.js': 'live.js', '/vendor/cytoscape.min.js': 'vendor/cytoscape.min.js'}
        if url.path == '/api/model-status':
            return self.send(200,json.dumps(running_chat_models(ROOT)).encode(),'application/json')
        if url.path == '/api/live':
            params=parse_qs(url.query)
            try:
                cursor=int(params.get('cursor',['0'])[0])
                if cursor<0: raise ValueError('Invalid live cursor')
                result=self.server.live.poll(params.get('client',[''])[0],cursor,
                                             params.get('stop',['0'])[0]=='1')
                return self.send(200,json.dumps(result).encode(),'application/json')
            except (ValueError,OSError):
                return self.send(400,b'{"error":"Live activity unavailable"}','application/json')
        if url.path in files:
            name = files[url.path]
            mime = 'text/html' if name.endswith('html') else 'text/css' if name.endswith('css') else 'text/javascript'
            return self.send(200, (HERE / name).read_bytes(), mime + '; charset=utf-8')
        if url.path in ('/api/catalog','/api/browse','/api/detail'):
            params=parse_qs(url.query)
            try:
                if url.path=='/api/catalog':data=self.server.bridge.workbench('catalog')
                elif url.path=='/api/detail':
                    key=params.get('key',[''])[0]
                    if not key or len(key)>160:raise ValueError('Invalid object key')
                    data=self.server.bridge.workbench('detail',dict(dataset=params.get('dataset',[''])[0],key=key))
                else:
                    dataset=params.get('dataset',['memory'])[0]
                    if dataset not in ('memory','cases','experiences','policies','entities'):raise ValueError('Unknown dataset')
                    raw=params.get('filters',['[]'])[0]
                    if len(raw)>3000:raise ValueError('Filters exceed limit')
                    filters=json.loads(raw)
                    if not isinstance(filters,list) or len(filters)>12:raise ValueError('Invalid filters')
                    limit=int(params.get('limit',['50'])[0]);offset=int(params.get('offset',['0'])[0])
                    if not 1<=limit<=100 or offset<0:raise ValueError('Invalid pagination')
                    frame=params.get('frame',[None])[0]
                    if frame is not None and len(frame)>160:raise ValueError('Invalid frame')
                    options=dict(dataset=dataset,frame=frame,filters=filters,limit=limit,offset=offset)
                    if params.get('live',['0'])[0]=='1':
                        return self.stream_query(lambda:self.server.bridge.workbench('browse',options))
                    data=self.server.bridge.workbench('browse',options)
                return self.send(200,json.dumps(data,ensure_ascii=False).encode(),'application/json; charset=utf-8')
            except (ValueError,RuntimeError) as error:
                return self.send(422,json.dumps({'error':str(error)}).encode(),'application/json')
            except Exception:
                return self.send(503,b'{"error":"Memory unavailable; retry after checking MCP."}','application/json')
        if url.path == '/api/graph':
            params = parse_qs(url.query)
            mode, query = params.get('mode', ['initial'])[0], params.get('q', [''])[0].strip()
            if mode not in ('initial', 'search', 'expand', 'semantic', 'natural') or len(query) > 8000:
                return self.send(400, b'Invalid query', 'text/plain')
            try:
                if params.get('live',['0'])[0]=='1':
                    return self.stream_query(lambda:self.server.bridge.browse(mode,query))
                data = self.server.bridge.browse(mode, query)
                data['memory_origin']={'installation':hashlib.sha256(str(ROOT.resolve()).encode()).hexdigest()[:24],
                                       'namespace':self.server.bridge.namespace}
                return self.send(200, json.dumps(data, ensure_ascii=False).encode(), 'application/json; charset=utf-8')
            except (RuntimeError, ValueError) as error:
                return self.send(422, json.dumps({'error': str(error)}).encode(), 'application/json')
            except Exception:
                return self.send(503, json.dumps({'error': 'Memory unavailable. Check the MCP connection and try again.'}).encode(), 'application/json')
        self.send(404, b'Not found', 'text/plain')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8090, help='Local HTTP port (default: 8090)')
    parser.add_argument('--namespace', default='main')
    args = parser.parse_args()
    if not args.namespace.isidentifier() or len(args.namespace) > 32:
        parser.error('Invalid namespace')
    if not 0 <= args.port <= 65535:
        parser.error('Port must be 0..65535')
    bridge = MemoryBridge(args.namespace)
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    except OSError:
        bridge.close()
        raise
    server.daemon_threads = True
    server.bridge = bridge
    server.live = LiveCollector(args.namespace)
    print(f'Open http://127.0.0.1:{server.server_port}/ — Ctrl+C to stop', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.live.close(); server.server_close(); bridge.close()


if __name__ == '__main__':
    main()
