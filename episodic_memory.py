"""Role-bound episodic tensors and supervised local outcome prediction.

Language interpretation is external. Hash-derived codes encode identities, not
semantic similarity. Learned readouts predict outcomes; predictions never become
observations automatically. No source LLM parameters are changed.
"""
import hashlib
import json
import time
import numpy as np
from storage import connect,retry_transaction
import semantic_memory

ENTITY_DIM, ROLE_DIM = 16, 8

def canonical(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        raise ValueError('Use a nonempty label of at most 160 characters')
    return ' '.join(value.casefold().split())

def code(label, dimension, kind):
    seed = int.from_bytes(hashlib.sha256((kind + ':' + canonical(label)).encode()).digest()[:8], 'little')
    a = np.random.default_rng(seed).standard_normal(dimension).astype('<f4')
    return a / np.linalg.norm(a)

def bindings(roles):
    if not isinstance(roles, list) or not 1 <= len(roles) <= 12:
        raise ValueError('An event needs 1..12 role bindings')
    result = []
    for item in roles:
        if set(item) != {'role', 'value'}: raise ValueError('Invalid role binding')
        result.append({'role': canonical(item['role']), 'value': canonical(item['value'])})
    if len({x['role'] for x in result}) != len(result): raise ValueError('Duplicate role')
    return result

def tensor(roles):
    return sum(np.outer(code(x['role'], ROLE_DIM, 'role'), code(x['value'], ENTITY_DIM, 'entity')) for x in roles).astype('<f4')

class OutcomeOperator:
    """Shared role-axis readout: y = event_tensor.T @ w.

    The role-binding architecture is supplied. Which combination predicts an
    observed outcome is learned by small recursive least squares updates.
    """
    def __init__(self, weights=None, precision=None, count=0):
        self.weights = np.zeros(ROLE_DIM, dtype='<f4') if weights is None else weights
        self.precision = np.eye(ROLE_DIM, dtype='<f4') * 100 if precision is None else precision
        self.count = count
    def predict(self, event): return event.T @ self.weights
    def observe(self, event, target):
        for row, y in zip(event.T, target):
            px = self.precision @ row
            gain = px / (1 + row @ px)
            self.weights += gain * (y - row @ self.weights)
            self.precision -= np.outer(gain, px)
            self.precision[:] = (self.precision + self.precision.T) * .5
        if not np.isfinite(self.weights).all() or not np.isfinite(self.precision).all():
            raise ArithmeticError('Nonfinite operator')
        self.count += 1

class EpisodicMemory:
    def __init__(self, namespace='main'):
        if not namespace.isidentifier() or len(namespace) > 32: raise ValueError('Invalid namespace')
        self.namespace = namespace
        self.c = connect()
        # Avoid no-op DDL on every MCP call: ALTER TABLE can invalidate concurrent
        # InnoDB/vector-index transactions even when the column already exists.
        with self.c.cursor() as q:
            q.execute('''SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema=DATABASE() AND table_name IN
                ('episodic_events','episodic_bindings','episodic_entities',
                 'episodic_operators','episodic_outcomes','semantic_encoders',
                 'episodic_semantic_vectors')''')
            complete=q.fetchone()[0]==7
        self.c.commit()
        if complete:return
        with self.c.cursor() as q:
            q.execute('''CREATE TABLE IF NOT EXISTS episodic_events(
                id BIGINT AUTO_INCREMENT PRIMARY KEY, namespace VARCHAR(32) NOT NULL,
                frame VARCHAR(160) NOT NULL, negated BOOLEAN NOT NULL, active BOOLEAN NOT NULL DEFAULT TRUE,
                event_time VARCHAR(160) NOT NULL, place VARCHAR(160) NOT NULL,
                roles JSON NOT NULL, source TEXT NOT NULL, provenance VARCHAR(32) NOT NULL,
                representation VECTOR(128) NOT NULL, created TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6),
                INDEX lookup(namespace,frame,negated)) ENGINE=InnoDB''')
            q.execute('''CREATE TABLE IF NOT EXISTS episodic_bindings(
                event_id BIGINT NOT NULL, role VARCHAR(160) NOT NULL, label VARCHAR(160) NOT NULL,
                PRIMARY KEY(event_id,role), INDEX binding(role,label),
                FOREIGN KEY(event_id) REFERENCES episodic_events(id) ON DELETE CASCADE) ENGINE=InnoDB''')
            q.execute('''CREATE TABLE IF NOT EXISTS episodic_entities(
                namespace VARCHAR(32), label VARCHAR(160), representation VECTOR(16) NOT NULL,
                PRIMARY KEY(namespace,label)) ENGINE=InnoDB''')
            q.execute('''CREATE TABLE IF NOT EXISTS episodic_operators(
                namespace VARCHAR(32), frame VARCHAR(160), relation_name VARCHAR(160),
                revision BIGINT NOT NULL, examples BIGINT NOT NULL, weights VECTOR(8) NOT NULL,
                precision_rows VECTOR(64) NOT NULL, checksum CHAR(64) NOT NULL,
                PRIMARY KEY(namespace,frame,relation_name)) ENGINE=InnoDB''')
            q.execute('''CREATE TABLE IF NOT EXISTS episodic_outcomes(
                event_id BIGINT, relation_name VARCHAR(160), value VARCHAR(160), evidence TEXT NOT NULL,
                PRIMARY KEY(event_id,relation_name),
                FOREIGN KEY(event_id) REFERENCES episodic_events(id) ON DELETE CASCADE) ENGINE=InnoDB''')
        with self.c.cursor() as q:
            q.execute('ALTER TABLE episodic_events ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE')
        semantic_memory.ensure_schema(self.c)
        self.c.commit()
    def close(self): self.c.close()
    def vocabulary(self,active=True):
        self.c.commit()
        with self.c.cursor() as q:
            q.execute('SELECT DISTINCT frame FROM episodic_events WHERE namespace=%s AND active=%s ORDER BY frame LIMIT 32', (self.namespace,active))
            frames = [r[0] for r in q.fetchall()]
            q.execute('SELECT DISTINCT b.role FROM episodic_bindings b JOIN episodic_events e ON e.id=b.event_id WHERE e.namespace=%s AND e.active=%s ORDER BY b.role LIMIT 32', (self.namespace,active))
            roles = [r[0] for r in q.fetchall()]
            q.execute('SELECT DISTINCT e.frame,b.role,b.label FROM episodic_bindings b JOIN episodic_events e ON e.id=b.event_id WHERE e.namespace=%s AND e.active=%s ORDER BY e.frame,b.role,b.label LIMIT 65', (self.namespace,active))
            labels=q.fetchall()
        return dict(frames=frames, roles=roles,
                    participants=[dict(frame=r[0],role=r[1],value=r[2]) for r in labels[:64]],
                    participants_truncated=len(labels)>64)
    def remember(self, frame, roles, source, negated=False, event_time='', place='', provenance='user_statement'):
        result=self.remember_many([dict(frame=frame,roles=roles,source=source,negated=negated,
                                       event_time=event_time,place=place)],provenance)
        return dict(event_id=result['event_ids'][0],write_ms=result['write_ms'],
                    semantic_encoding_seconds=result['semantic_encoding_seconds'],
                    semantic_chunks=result['semantic_chunks'])

    @retry_transaction
    def remember_many(self, events, provenance='user_statement'):
        if not isinstance(events,list) or not 1<=len(events)<=64:
            raise ValueError('Use 1..64 events per batch')
        if provenance not in ('user_statement','supervised_example'):
            raise ValueError('Generated content cannot teach itself')
        normalized=[]
        for event in events:
            if not isinstance(event,dict) or set(event)-{'frame','roles','source','negated','event_time','place'}:
                raise ValueError('Invalid event fields')
            frame=canonical(event.get('frame'));roles=bindings(event.get('roles'))
            source=event.get('source');negated=event.get('negated',False)
            event_time=event.get('event_time','');place=event.get('place','')
            if not isinstance(negated,bool):raise ValueError('Negation must be boolean')
            if not isinstance(source,str) or not 1<=len(source)<=8192:raise ValueError('Invalid source text')
            if not isinstance(event_time,str) or not isinstance(place,str) or max(len(event_time),len(place))>160:
                raise ValueError('Context too long')
            normalized.append(dict(frame=frame,roles=roles,source=source,negated=negated,event_time=event_time,place=place))
        if sum(len(e['source']) for e in normalized)>64000:
            raise ValueError('Batch source budget exceeded (64000 characters)')
        encoding_started=time.perf_counter()
        encoded=semantic_memory.embed_passages([e['source'] for e in normalized])
        encoding_seconds=time.perf_counter()-encoding_started
        started=time.perf_counter();self.c.begin();ids=[];binding_rows=[];entities={};vector_entries=[]
        try:
            with self.c.cursor() as q:
                q.execute('SELECT COUNT(*) FROM episodic_events WHERE namespace=%s',(self.namespace,))
                if q.fetchone()[0]+len(normalized)>10000:raise ValueError('Namespace event budget reached')
                for event,vectors in zip(normalized,encoded):
                    q.execute('INSERT INTO episodic_events(namespace,frame,negated,event_time,place,roles,source,provenance,representation) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                              (self.namespace,event['frame'],event['negated'],event['event_time'],event['place'],
                               json.dumps(event['roles']),event['source'],provenance,tensor(event['roles']).reshape(-1).tobytes()))
                    ident=q.lastrowid;ids.append(ident)
                    vector_entries.append((ident,event['source'],vectors))
                    for binding in event['roles']:
                        binding_rows.append((ident,binding['role'],binding['value']))
                        entities[binding['value']]=(self.namespace,binding['value'],code(binding['value'],ENTITY_DIM,'entity').tobytes())
                semantic_memory.insert_vector_batch(q,vector_entries)
                q.executemany('INSERT INTO episodic_bindings VALUES(%s,%s,%s)',binding_rows)
                q.executemany('INSERT IGNORE INTO episodic_entities VALUES(%s,%s,%s)',list(entities.values()))
            self.c.commit()
        except BaseException:self.c.rollback();raise
        return dict(event_ids=ids,stored=len(ids),write_ms=(time.perf_counter()-started)*1000,
                    semantic_encoding_seconds=encoding_seconds,semantic_chunks=sum(map(len,encoded)))
    def find(self, frame, roles, negated=False, event_time='', place='', limit=8, active=True):
        frame=canonical(frame)
        if not 1 <= limit <= 16: raise ValueError('Retrieval budget exceeded')
        roles=bindings(roles) if roles else []
        query=tensor(roles).reshape(-1) if roles else np.zeros(128,dtype='<f4')
        score='VEC_DISTANCE_COSINE(e.representation,%s)' if roles else '0'
        params=[query.tobytes()] if roles else []
        clauses=['e.namespace=%s','e.frame=%s','e.negated=%s','e.active=%s']; params += [self.namespace,frame,bool(negated),bool(active)]
        for b in roles:
            clauses.append('EXISTS(SELECT 1 FROM episodic_bindings b WHERE b.event_id=e.id AND b.role=%s AND b.label=%s)')
            params += [b['role'],b['value']]
        for column,value in (('event_time',event_time),('place',place)):
            if value: clauses.append('e.'+column+'=%s');params.append(value)
        params.append(limit)
        self.c.commit()
        with self.c.cursor() as q:
            q.execute('SELECT e.id,e.roles,e.source,e.event_time,e.place,'+score+' AS distance FROM episodic_events e WHERE '+' AND '.join(clauses)+' ORDER BY distance,e.id DESC LIMIT %s',params)
            return [dict(event_id=r[0],roles=json.loads(r[1]),source=r[2],time=r[3],place=r[4],distance=r[5]) for r in q.fetchall()]
    @retry_transaction
    def observe_outcome(self, event_id, relation, value, evidence):
        """Explicit supervision only; not an effect inferred from the verb."""
        relation=canonical(relation);value=canonical(value)
        if not isinstance(evidence,str) or not evidence.strip() or len(evidence)>8192:raise ValueError('Observed evidence required')
        self.c.begin()
        try:
            with self.c.cursor() as q:
                q.execute('SELECT frame,representation,negated,roles FROM episodic_events WHERE id=%s AND namespace=%s AND active=1 FOR UPDATE',(event_id,self.namespace));event=q.fetchone()
                if not event:raise ValueError('Unknown event')
                event_roles=json.loads(event[3])
                if len(event_roles)>ROLE_DIM:raise ValueError('Outcome learner supports at most eight roles')
                if value not in {b['value'] for b in event_roles}:raise ValueError('This readout supports participant-valued outcomes only')
                # Positive and negative events have separate learned operators.
                frame=event[0]+(' [negated]' if event[2] else '')
                q.execute('SELECT examples,weights,precision_rows,checksum FROM episodic_operators WHERE namespace=%s AND frame=%s AND relation_name=%s FOR UPDATE',(self.namespace,frame,relation));row=q.fetchone()
                if row:
                    count,raw,precision,sha=row
                    if hashlib.sha256(raw+precision).hexdigest()!=sha:raise ValueError('Operator checksum mismatch')
                    op=OutcomeOperator(np.frombuffer(raw,dtype='<f4').copy(),np.frombuffer(precision,dtype='<f4').copy().reshape(8,8),count)
                else:op=OutcomeOperator()
                q.execute('SELECT value FROM episodic_outcomes WHERE event_id=%s AND relation_name=%s',(event_id,relation))
                if q.fetchone():raise ValueError('Outcome already supervised; do not silently overwrite evidence')
                op.observe(np.frombuffer(event[1],dtype='<f4').reshape(8,16),code(value,16,'entity'))
                raw=op.weights.tobytes();precision=op.precision.tobytes()
                q.execute('INSERT INTO episodic_operators VALUES(%s,%s,%s,1,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE revision=revision+1,examples=VALUES(examples),weights=VALUES(weights),precision_rows=VALUES(precision_rows),checksum=VALUES(checksum)',(self.namespace,frame,relation,op.count,raw,precision,hashlib.sha256(raw+precision).hexdigest()))
                q.execute('INSERT INTO episodic_outcomes VALUES(%s,%s,%s,%s)',(event_id,relation,value,evidence))
                q.execute('INSERT IGNORE INTO episodic_entities VALUES(%s,%s,%s)',(self.namespace,value,code(value,16,'entity').tobytes()))
            self.c.commit()
        except BaseException:self.c.rollback();raise
        return dict(examples=op.count,learned_coefficients=8)
    def predict(self, frame, roles, relation, negated=False):
        frame=canonical(frame)+(' [negated]' if negated else '');roles=bindings(roles);relation=canonical(relation)
        if len(roles)>ROLE_DIM:return dict(status='unknown',reason='Outcome readout supports at most eight roles')
        self.c.commit()
        with self.c.cursor() as q:
            q.execute('SELECT examples,weights,precision_rows,checksum FROM episodic_operators WHERE namespace=%s AND frame=%s AND relation_name=%s',(self.namespace,frame,relation));row=q.fetchone()
        if not row:return dict(status='unknown',reason='No observed outcomes for this frame and relation')
        count,raw,precision,sha=row
        if hashlib.sha256(raw+precision).hexdigest()!=sha:raise ValueError('Operator checksum mismatch')
        op=OutcomeOperator(np.frombuffer(raw,dtype='<f4'),np.frombuffer(precision,dtype='<f4').reshape(8,8),count)
        y=op.predict(tensor(roles));norm=float(np.linalg.norm(y))
        if norm<1e-6:return dict(status='unknown',reason='No usable prediction')
        # Open identity codes allow decoding previously unseen participants.
        candidates=sorted({b['value'] for b in roles})
        scores=[float(y@code(v,16,'entity')/norm) for v in candidates]
        order=np.argsort(scores)[::-1];best=int(order[0]);margin=scores[best]-(scores[int(order[1])] if len(order)>1 else 0)
        return dict(status='prediction',value=candidates[best],similarity=scores[best],margin=margin,examples=count,
                    scope='Hypothesis among supplied participants; no calibrated confidence or truth guarantee')
    def forget(self,event_ids):return self._change_events(event_ids)
    def deactivate(self,event_ids):return self._change_events(event_ids,False)
    def restore(self,event_ids):return self._change_events(event_ids,True)
    @retry_transaction
    def _change_events(self, event_ids, active=None):
        """Atomically remove selected episodes and rebuild affected readouts.

        Only operators trained from those episodes are rebuilt. No source LLM
        weights or other namespaces are reachable through this method.
        """
        if not isinstance(event_ids,list) or not 1<=len(event_ids)<=32 or any(type(i) is not int or i<=0 for i in event_ids):
            raise ValueError('Supply 1..32 positive event IDs')
        ids=sorted(set(event_ids));marks=','.join(['%s']*len(ids));started=time.perf_counter()
        self.c.begin()
        try:
            with self.c.cursor() as q:
                q.execute('SELECT id FROM episodic_events WHERE namespace=%s AND id IN ('+marks+') ORDER BY id FOR UPDATE',[self.namespace]+ids)
                if [row[0] for row in q.fetchall()]!=ids:
                    raise ValueError('Every event must exist in this namespace; nothing deleted')
                q.execute('SELECT DISTINCT e.frame,e.negated,o.relation_name FROM episodic_outcomes o JOIN episodic_events e ON e.id=o.event_id WHERE e.namespace=%s AND e.id IN ('+marks+') ORDER BY e.frame,e.negated,o.relation_name',[self.namespace]+ids)
                affected=q.fetchall()
                if len(affected)>32:raise ValueError('Operator rebuild budget exceeded; nothing deleted')
                replays=[];examples=0
                for frame,negative,relation in affected:
                    key=frame+(' [negated]' if negative else '')
                    q.execute('SELECT revision FROM episodic_operators WHERE namespace=%s AND frame=%s AND relation_name=%s FOR UPDATE',(self.namespace,key,relation))
                    previous=q.fetchone();revision=previous[0] if previous else 0
                    eligibility=('(e.active=1 OR e.id IN ('+marks+'))' if active is True else '(e.active=1 AND e.id NOT IN ('+marks+'))')
                    q.execute('SELECT e.representation,o.value FROM episodic_outcomes o JOIN episodic_events e ON e.id=o.event_id WHERE e.namespace=%s AND e.frame=%s AND e.negated=%s AND o.relation_name=%s AND '+eligibility+' ORDER BY e.id LIMIT %s',[self.namespace,frame,negative,relation]+ids+[10001-examples])
                    rows=q.fetchall();examples+=len(rows)
                    if examples>10000:raise ValueError('Replay budget exceeded; nothing deleted')
                    replays.append((key,relation,revision,rows))
                if active is None:q.execute('DELETE FROM episodic_events WHERE namespace=%s AND id IN ('+marks+')',[self.namespace]+ids)
                else:q.execute('UPDATE episodic_events SET active=%s WHERE namespace=%s AND id IN ('+marks+')',[active,self.namespace]+ids)
                for key,relation,revision,rows in replays:
                    if not rows:
                        q.execute('DELETE FROM episodic_operators WHERE namespace=%s AND frame=%s AND relation_name=%s',(self.namespace,key,relation))
                        continue
                    op=OutcomeOperator()
                    for raw,value in rows:op.observe(np.frombuffer(raw,dtype='<f4').reshape(8,16),code(value,16,'entity'))
                    raw=op.weights.tobytes();precision=op.precision.tobytes()
                    q.execute('INSERT INTO episodic_operators VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE revision=VALUES(revision),examples=VALUES(examples),weights=VALUES(weights),precision_rows=VALUES(precision_rows),checksum=VALUES(checksum)',(self.namespace,key,relation,revision+1,op.count,raw,precision,hashlib.sha256(raw+precision).hexdigest()))
                q.execute('DELETE x FROM episodic_entities x WHERE x.namespace=%s AND NOT EXISTS(SELECT 1 FROM episodic_bindings b JOIN episodic_events e ON e.id=b.event_id WHERE e.namespace=x.namespace AND b.label=x.label) AND NOT EXISTS(SELECT 1 FROM episodic_outcomes o JOIN episodic_events e ON e.id=o.event_id WHERE e.namespace=x.namespace AND o.value=x.label)',(self.namespace,))
            self.c.commit()
        except BaseException:self.c.rollback();raise
        key='deleted_event_ids' if active is None else ('restored_event_ids' if active else 'deactivated_event_ids')
        return dict(**{key:ids},operators_rebuilt=len(replays),replayed_observations=examples,seconds=time.perf_counter()-started,reversible=(active is not None))

    def status(self):
        self.c.commit()
        with self.c.cursor() as q:
            q.execute('SELECT COALESCE(SUM(active=1),0),COALESCE(SUM(active=0),0) FROM episodic_events WHERE namespace=%s',(self.namespace,));events,archived=q.fetchone()
            q.execute('SELECT COUNT(*) FROM episodic_operators WHERE namespace=%s',(self.namespace,));operators=q.fetchone()[0]
        return dict(namespace=self.namespace,events=int(events),archived_events=int(archived),learned_operators=operators)
    @retry_transaction
    def delete_namespace(self):
        """Used only by isolated verification namespaces."""
        if not self.namespace.startswith('check_'):raise ValueError('Refusing operational namespace deletion')
        self.c.rollback();self.c.begin()
        with self.c.cursor() as q:
            q.execute('DELETE FROM episodic_events WHERE namespace=%s',(self.namespace,))
            q.execute('DELETE FROM episodic_operators WHERE namespace=%s',(self.namespace,))
            q.execute('DELETE FROM episodic_entities WHERE namespace=%s',(self.namespace,))
        self.c.commit()
