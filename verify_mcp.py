"""Real STDIO MCP write/read/restart checks without any language model."""
import asyncio
import json
import os
from pathlib import Path
import sys
import time
import uuid
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from episodic_memory import EpisodicMemory

ROOT=Path(__file__).resolve().parent

def data(result):
    if result.isError:raise RuntimeError(str(result.content))
    return result.structuredContent or json.loads(result.content[0].text)

def roles(a,b):return [dict(role='agent',value=a),dict(role='patient',value='libro'),dict(role='recipient',value=b)]

async def main():
    started=time.perf_counter();ns='check_'+uuid.uuid4().hex[:12];report={};params=StdioServerParameters(command=sys.executable,args=[str(ROOT/'mcp_server.py')])
    try:
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as client:
                await client.initialize()
                report['tools']=[t.name for t in (await client.list_tools()).tools]
                status=data(await client.call_tool('memory_status',{'namespace':ns}))
                report['no_local_llm']=not status['local_llm_loaded'];report['server_rss_mib']=status['rss_mib']
                ident=None
                for a,b,negated in [('Marco','Luca',False),('Luca','Marco',False),('Sara','Paolo',True)]:
                    result=data(await client.call_tool('remember_event',dict(namespace=ns,frame='give',roles=roles(a,b),source=f'Test event: {a} gives book to {b}, negated={negated}',negated=negated)))
                    if ident is None:ident=result['event_id']
                query=dict(namespace=ns,frame='give',roles=[dict(role='agent',value='Marco')])
                found=data(await client.call_tool('recall_events',query))
                report['role_direction_pass']=len(found['events'])==1 and found['events'][0]['roles'][2]['value']=='luca'
                negative=data(await client.call_tool('recall_events',dict(namespace=ns,frame='give',roles=[dict(role='agent',value='Sara')],negated=False)))
                report['negative_event_not_positive']=negative['count']==0
                trained=data(await client.call_tool('observe_outcome',dict(namespace=ns,event_id=ident,relation='holder',value='Luca',evidence='User explicitly supplies observed holder Luca')))
                prediction=data(await client.call_tool('predict_outcome',dict(namespace=ns,frame='give',roles=roles('Marco','Luca'),relation='holder')))
                report['tensor_operator_available']=trained['examples']==1 and prediction['status']=='prediction'
                invalid=await client.call_tool('memory_status',{'namespace':'../SQLT'})
                report['invalid_namespace_rejected']=bool(invalid.isError)
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as client:
                await client.initialize()
                found=data(await client.call_tool('recall_events',query))
                report['new_server_process_recall_pass']=len(found['events'])==1 and found['events'][0]['event_id']==ident
                prediction=data(await client.call_tool('predict_outcome',dict(namespace=ns,frame='give',roles=roles('Marco','Luca'),relation='holder')))
                report['new_server_process_operator_pass']=prediction['status']=='prediction' and prediction['examples']==1
                removed=data(await client.call_tool('deactivate_events',dict(namespace=ns,event_ids=[ident])))
                report['mcp_deactivate_pass']=removed['deactivated_event_ids']==[ident]
                report['deleted_operator_not_reused']=data(await client.call_tool('predict_outcome',dict(namespace=ns,frame='give',roles=roles('Marco','Luca'),relation='holder')))['status']=='unknown'
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as client:
                await client.initialize()
                report['forget_survives_restart']=data(await client.call_tool('recall_events',query))['count']==0
                report['other_events_preserved']=data(await client.call_tool('memory_status',dict(namespace=ns)))['events']==2
                archived=data(await client.call_tool('recall_inactive_events',dict(namespace=ns,frame='give',roles=[dict(role='agent',value='Marco')])))
                report['reversible_archive_preserved']=archived['count']==1
                data(await client.call_tool('restore_events',dict(namespace=ns,event_ids=[ident])))
                report['restore_pass']=data(await client.call_tool('recall_events',query))['count']==1
        report['passed']=all(report[k] for k in ('no_local_llm','role_direction_pass','negative_event_not_positive','tensor_operator_available','invalid_namespace_rejected','new_server_process_recall_pass','new_server_process_operator_pass','mcp_deactivate_pass','deleted_operator_not_reused','forget_survives_restart','other_events_preserved','reversible_archive_preserved','restore_pass'))
    finally:
        memory=EpisodicMemory(ns)
        try:memory.delete_namespace()
        finally:memory.close()
        report['elapsed_seconds']=time.perf_counter()-started
        (ROOT/'verification-mcp.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    if not report.get('passed'):raise SystemExit(1)
if __name__=='__main__':asyncio.run(main())
