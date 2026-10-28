"""Bounded read-only APFC object browsing for visual and natural query clients."""
import json
import time

CATALOG={
 'cases':['case_key','title','revision'],
 'experiences':['case_key','experience_key','policy_key','action','status','reward','outcome_definition'],
 'policies':['policy_key'],
 'entities':['entity_key','label','scope'],
}


def browse(memory,dataset,filters=None,limit=50,offset=0,include_sql=False):
    if dataset not in CATALOG:raise ValueError('Unknown APFC dataset')
    if type(limit)is not int or not 1<=limit<=100 or type(offset)is not int or offset<0:raise ValueError('Invalid page')
    if not isinstance(filters or [],list) or len(filters or [])>12:raise ValueError('At most 12 filters')
    definitions={
      'cases':('apfc_cases c','c.namespace',{'case_key':'c.case_key','title':'c.title','revision':'c.revision'},
               ['c.id','c.case_key','c.title','c.revision'],['id','case_key','title','revision'],'c.id',''),
      'entities':('apfc_entities n JOIN episodic_events e ON e.id=n.evidence_event_id','n.namespace',
               {'entity_key':'n.entity_key','label':'n.label','scope':'n.scope'},
               ['n.id','n.entity_key','n.label','n.scope','n.evidence_event_id'],['id','entity_key','label','scope','evidence_event_id'],'n.id','e.active=1 AND e.namespace=n.namespace'),
      'policies':('apfc_policies p','p.namespace',{'policy_key':'p.policy_key'},
               ['p.id','p.policy_key','p.features','p.actions'],['id','policy_key','features','actions'],'p.id',''),
      'experiences':('apfc_experiences x JOIN apfc_cases c ON c.id=x.case_id JOIN apfc_policies p ON p.id=x.policy_id JOIN episodic_events b ON b.id=x.before_event_id LEFT JOIN episodic_events o ON o.id=x.outcome_event_id','c.namespace',
               {'case_key':'c.case_key','experience_key':'x.experience_key','policy_key':'p.policy_key','action':'x.action','status':'x.status','reward':'x.reward','outcome_definition':'x.outcome_definition'},
               ['x.id','c.case_key','x.experience_key','p.policy_key','x.action','x.status','x.reward','x.features','x.before_event_id','x.outcome_event_id','x.outcome_definition','x.observed_quote'],
               ['id','case_key','experience_key','policy_key','action','status','reward','features','before_event_id','outcome_event_id','outcome_definition','observed_quote'],'x.id',"b.active=1 AND b.namespace=c.namespace AND (x.status='pending' OR (o.active=1 AND o.namespace=c.namespace))"),
    }
    joins,namespace,columns,select,names,order,active=definitions[dataset]
    clauses=[namespace+'=%s'];params=[memory.namespace]
    if active:clauses.append(active)
    for item in filters or []:
        if not isinstance(item,dict) or set(item)-{'role','operator','value'}:raise ValueError('Invalid filter')
        role=item.get('role');op=item.get('operator','eq');value=item.get('value')
        if role not in columns:raise ValueError('Field unavailable in selected APFC dataset')
        col=columns[role]
        if op in ('exists','missing'):
            if value is not None:raise ValueError('Existence filter takes no value')
            clauses.append(col+(' IS NOT NULL' if op=='exists' else ' IS NULL'))
        elif op in ('eq','ne','contains','prefix'):
            if not isinstance(value,(str,int,float)) or isinstance(value,bool) or len(str(value))>1000:raise ValueError('Invalid filter value')
            if op in ('eq','ne'):clauses.append(col+('=%s' if op=='eq' else '<>%s'));params.append(value)
            elif op=='contains':clauses.append('LOCATE(%s,'+col+')>0');params.append(str(value))
            else:clauses.append('LEFT('+col+',CHAR_LENGTH(%s))=%s');params.extend([str(value),str(value)])
        elif op in ('number_eq','gt','gte','lt','lte'):
            if role not in ('revision','reward'):raise ValueError('Field is not numeric')
            import math
            try:number=float(value)
            except (ValueError,TypeError):raise ValueError('Invalid numeric operand')
            if not math.isfinite(number):raise ValueError('Invalid numeric operand')
            clauses.append(col+{'number_eq':'=','gt':'>','gte':'>=','lt':'<','lte':'<='}[op]+'%s');params.append(number)
        else:raise ValueError('Unsupported operator')
    where=' AND '.join(clauses);traces=[];start=time.perf_counter();memory.c.commit()
    try:
        with memory.c.cursor() as q:
            q.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
            memory.c.begin()
            sql='SELECT COUNT(*) FROM '+joins+' WHERE '+where
            t=time.perf_counter();q.execute(sql,params);total=q.fetchone()[0]
            traces.append(dict(sql=sql,parameters=params.copy(),seconds=time.perf_counter()-t))
            sql='SELECT '+','.join(select)+' FROM '+joins+' WHERE '+where+' ORDER BY '+order+' LIMIT %s OFFSET %s'
            t=time.perf_counter();q.execute(sql,params+[limit,offset]);rows=q.fetchall()
            traces.append(dict(sql=sql,parameters=params+[limit,offset],seconds=time.perf_counter()-t))
        memory.c.commit()
    except BaseException:memory.c.rollback();raise
    records=[]
    for row in rows:
        record=dict(zip(names,row))
        for key in ('features','actions'):
            if key in record:record[key]=json.loads(record[key])
        records.append(record)
    result=dict(dataset=dataset,records=records,total=total,has_more=offset+len(rows)<total,next_offset=offset+len(rows) if offset+len(rows)<total else None,sql_seconds=time.perf_counter()-start,
                scope='Active evidence only; metadata/observations, not uncalibrated predictions or verified causal relations')
    if include_sql:result['sql_trace']=traces
    return result
