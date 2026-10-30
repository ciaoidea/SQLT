"""Read-only adapter to the existing validated SQL Tensor NLQ pipeline."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
from urllib.parse import urlencode, urlsplit

ROOT=Path(__file__).resolve().parent


def context_from_result(data, namespace, root=ROOT):
    expected=hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:24]
    if data.get('memory_origin')!={'installation':expected,'namespace':namespace}:
        raise ValueError('NLQ origin/namespace mismatch; refusing another memory installation')
    plan=data.get('plan')
    if not isinstance(plan,dict):raise ValueError('NLQ response is missing its validated plan')
    if plan.get('dataset','memory')!='memory':
        raise ValueError('Automatic episodic recall does not flatten APFC objects into facts')
    candidates=[];used=0
    records=[] if plan['mode']=='clarify' else data.get('records',[])
    for event in records:
        item={k:event.get(k) for k in ('event_id','frame','negated','roles','time','place')}
        source=event.get('source','')
        item.update(source=source[:600],source_truncated=event.get('source_truncated',False) or len(source)>600)
        size=len(json.dumps(item,ensure_ascii=False))
        if used+size>6500:continue
        candidates.append(item);used+=size
        if len(candidates)>=8:break
    return {'namespace':namespace,'retrieval':'validated_natural_language_query','plan':plan,
            'candidates':candidates,'clarification':plan.get('clarification',''),
            'outcome':'clarification_required' if plan['mode']=='clarify' else ('matching_records' if candidates else 'no_matching_records_for_this_plan'),
            'coverage':{'complete_database_search':False,'query_partial':data.get('partial',True),
                        'records_omitted':max(0,len(records)-len(candidates)),
                        'scope':'Only the validated plan; not proof of absence in external sources or other frames.'},
            'evidence_contract':{'live_source_checks':[], 'current_operational_status':'not_verified',
                                 'empty_result_means':'no_support_for_this_query_not_service_up_or_down'},
            'planning_seconds':data.get('planning_seconds')}


def natural_context(query,namespace):
    if not isinstance(query,str) or not 1<=len(query)<=8000:
        raise ValueError('Natural query supports 1..8000 characters; nothing was silently truncated')
    path=os.environ.get('CORTEX_NLQ_SOCKET')
    config_path=ROOT/'.nlq.json'
    config=json.loads(config_path.read_text()) if config_path.exists() else {}
    path=path or config.get('socket')
    if path:
        class LocalConnection(http.client.HTTPConnection):
            def connect(self):
                self.sock=socket.socket(socket.AF_UNIX);self.sock.settimeout(95);self.sock.connect(path)
        c=LocalConnection('cortex.internal',timeout=95);host='cortex.internal'
    else:
        url=urlsplit(config.get('url',''))
        if url.scheme!='http' or url.hostname not in ('127.0.0.1','localhost') or not url.port or url.path not in ('','/') or url.query or url.fragment or url.username or url.password:
            raise RuntimeError('Configure an installation-local UNIX socket or loopback HTTP NLQ backend')
        c=http.client.HTTPConnection(url.hostname,url.port,timeout=95);host=f'{url.hostname}:{url.port}'
    try:
        c.request('GET','/api/graph?'+urlencode({'mode':'natural','q':query}),headers={'Host':host})
        response=c.getresponse();raw=response.read(2000001)
        if len(raw)>2000000:raise ValueError('NLQ response exceeds budget')
        if response.status!=200:raise RuntimeError('Validated NLQ retrieval failed; no semantic fallback')
        return context_from_result(json.loads(raw),namespace)
    finally:c.close()
