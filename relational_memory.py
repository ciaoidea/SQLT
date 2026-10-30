"""Parameterized relational queries over active Cortex facts, without embeddings."""
import json
import time
from decimal import Decimal, InvalidOperation
from episodic_memory import canonical

OPERATORS={'eq','ne','in','contains','prefix','gt','gte','lt','lte','exists','missing'}
NUMERIC={'gt':'>','gte':'>=','lt':'<','lte':'<=','number_eq':'='}
NUMBER_PATTERN=r'^[+-]?([0-9]+([.][0-9]+)?|[.][0-9]+)$'


def where(namespace,frame,filters,negated):
    if not isinstance(filters,list) or len(filters)>16:raise ValueError('Use at most 16 filters')
    clauses=['e.namespace=%s','e.active=TRUE'];params=[namespace]
    if frame is not None:clauses.append('e.frame=%s');params.append(canonical(frame))
    if negated is not None:clauses.append('e.negated=%s');params.append(bool(negated))
    for item in filters:
        if not isinstance(item,dict) or set(item)-{'role','operator','value'}:
            raise ValueError('Invalid filter')
        role=canonical(item.get('role'));op=item.get('operator','eq');value=item.get('value')
        if op not in OPERATORS|{'number_eq'}:raise ValueError('Unsupported filter operator')
        predicate='b.event_id=e.id AND b.role=%s';args=[role]
        if op in ('exists','missing'):
            if value is not None:raise ValueError('Existence filters have no value')
        elif op in NUMERIC:
            if isinstance(value,bool):raise ValueError('A decimal number is required')
            try:number=Decimal(str(value))
            except InvalidOperation:raise ValueError('A decimal number is required') from None
            if not number.is_finite() or abs(number)>=Decimal('1e20') or number.as_tuple().exponent < -10:
                raise ValueError('Number exceeds DECIMAL(30,10) precision')
            predicate+=' AND b.label REGEXP %s AND CAST(b.label AS DECIMAL(30,10)) '+NUMERIC[op]+' %s'
            args.extend([NUMBER_PATTERN,str(number)])
        elif op=='in':
            if not isinstance(value,list) or not 1<=len(value)<=32:raise ValueError('IN requires 1..32 labels')
            labels=[canonical(v) for v in value]
            predicate+=' AND b.label IN ('+','.join(['%s']*len(labels))+')';args.extend(labels)
        else:
            value=canonical(value)
            if op in ('eq','ne'):
                predicate+=' AND b.label '+('=' if op=='eq' else '<>')+' %s';args.append(value)
            elif op=='contains':predicate+=' AND LOCATE(%s,b.label)>0';args.append(value)
            elif op=='prefix':predicate+=' AND LEFT(b.label,CHAR_LENGTH(%s))=%s';args.extend([value,value])
        clauses.append(('NOT ' if op=='missing' else '')+'EXISTS(SELECT 1 FROM episodic_bindings b WHERE '+predicate+')')
        params.extend(args)
    return ' AND '.join(clauses),params


def query(memory,filters,frame=None,negated=False,limit=100,offset=0,count_only=False,
          select_roles=None,include_source=True,include_sql=False):
    """AND filters; total counts all matches, never just a nearest-neighbor subset."""
    if type(limit) is not int or not 1<=limit<=100:raise ValueError('Page size must be 1..100')
    if type(offset) is not int or not 0<=offset<=10000:raise ValueError('Offset must be 0..10000')
    clause,params=where(memory.namespace,frame,filters,negated)
    if select_roles is not None:
        if not isinstance(select_roles,list) or not 1<=len(select_roles)<=12:raise ValueError('Select 1..12 roles')
        select_roles={canonical(role) for role in select_roles}
    started=time.perf_counter();memory.c.commit();trace=[]
    try:
        with memory.c.cursor() as cursor:
            cursor.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
            cursor.execute('START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY')
            if select_roles is not None:
                cursor.execute('SELECT DISTINCT b.role FROM episodic_bindings b '
                               'JOIN episodic_events e ON e.id=b.event_id '
                               'WHERE e.namespace=%s AND e.active=TRUE',(memory.namespace,))
                unknown=select_roles-{row[0] for row in cursor.fetchall()}
                if unknown:
                    raise ValueError('Unknown stored attributes in select_roles: '+', '.join(sorted(unknown))+'. Omit select_roles to inspect complete stored attributes.')
            count_sql='SELECT COUNT(*) FROM episodic_events e WHERE '+clause
            cursor.execute(count_sql,params)
            if include_sql:trace.append(dict(sql=count_sql,parameters=list(params)))
            total=cursor.fetchone()[0];events=[]
            if not count_only:
                rows_sql='SELECT e.id,e.frame,e.negated,e.roles,e.source,e.event_time,e.place FROM episodic_events e WHERE '+clause+' ORDER BY e.id LIMIT %s OFFSET %s'
                cursor.execute(rows_sql,params+[limit,offset])
                if include_sql:trace.append(dict(sql=rows_sql,parameters=params+[limit,offset]))
                events=[dict(event_id=r[0],frame=r[1],negated=bool(r[2]),roles=json.loads(r[3]),
                             source=r[4],time=r[5],place=r[6]) for r in cursor.fetchall()]
                for event in events:
                    if select_roles is not None:event['roles']=[b for b in event['roles'] if b['role'] in select_roles]
                    if not include_source:event.pop('source')
        memory.c.commit()
    except BaseException:
        memory.c.rollback();raise
    more=not count_only and offset+len(events)<total
    result=dict(events=events,count=len(events),total=total,offset=offset,limit=limit,
                has_more=more,next_offset=offset+len(events) if more else None,
                complete=not count_only and offset==0 and not more,count_only=count_only,
                sql_seconds=time.perf_counter()-started,search='relational_sql',
                scope='Exact AND filters over active user statements, not independently verified facts. '
                      'Total covers all matches. Follow next_offset for remaining pages; concurrent changes can affect later pages.')
    if include_sql:result['sql_trace']=trace
    return result


def subject_facts(events):
    """Logical fact view: exact subject labels, all attributes, no inferred joins."""
    subjects={}
    for event in events:
        labels=list(dict.fromkeys(b['value'] for b in event['roles'] if b['role']=='entity'))
        if len(labels)!=1:continue  # A relation with several subjects is not a property of each.
        scope=tuple(sorted((b['role'],b['value']) for b in event['roles'] if b['role'] in ('scope','company','customer','tenant','organization')))
        for label in labels:
            subject=subjects.setdefault((label,scope),{'entity':label,'scope':[{'role':k,'value':v} for k,v in scope],'attributes':{}})
            for binding in event['roles']:
                if binding['role']=='entity':continue
                entries=subject['attributes'].setdefault(binding['role'],[])
                entries.append({'value':binding['value'],'event_id':event['event_id'],
                                'negated':event['negated'],'time':event.get('time','')})
    return list(subjects.values())
