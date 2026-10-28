"""APFC: evidence-bound identities, revisioned work state and local outcome models.

No language parsing, generative model, raw SQL interface or autonomous executor.
All tables are additive; existing episodic memory remains the source evidence.
"""
from collections import OrderedDict
import hashlib
import json
import math
import time
import threading
import numpy as np
from storage import retry_transaction
from episodic_memory import canonical
from apfc_experience import ExperienceCoordinator, EXPERIENCE_SCHEMA

SECTIONS = {'goals','confirmed','completed','waiting','questions','next_steps','contradictions'}
DIM = 17
MAX_SAMPLES = 2048
_CACHE = OrderedDict()
_CACHE_LOCK = threading.RLock()
TABLES = ('apfc_entities','apfc_aliases','apfc_entity_events','apfc_cases',
          'apfc_case_events','apfc_case_revisions','apfc_policies','apfc_transitions','apfc_models','apfc_experiences')


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def ids(values):
    if not isinstance(values,list) or not 1 <= len(values) <= 32:
        raise ValueError('Provide 1..32 evidence event IDs')
    if any(type(x) is not int or x < 1 for x in values):
        raise ValueError('Invalid evidence event ID')
    return sorted(set(values))


def text(value, limit=1000):
    if not isinstance(value,str) or not value.strip() or len(value)>limit:
        raise ValueError('Invalid or oversized text')
    return value.strip()


def ensure_schema(c):
    # Explicit migration only: not run on each tool call.
    with c.cursor() as q:
        q.execute('SELECT GET_LOCK(%s,10)', ('apfc.migration',))
        if q.fetchone()[0]!=1:raise TimeoutError('APFC migration busy')
        try:
            for sql in SCHEMA:q.execute(sql)
            q.execute("ALTER TABLE apfc_experiences ADD COLUMN IF NOT EXISTS before_quote VARCHAR(1000) NOT NULL DEFAULT ''")
        finally:q.execute('SELECT RELEASE_LOCK(%s)', ('apfc.migration',))
    c.commit()


SCHEMA = [
'''CREATE TABLE IF NOT EXISTS apfc_entities(
 id BIGINT AUTO_INCREMENT PRIMARY KEY, namespace VARCHAR(32) NOT NULL,
 entity_key VARCHAR(160) NOT NULL, label VARCHAR(160) NOT NULL, scope VARCHAR(160) NOT NULL,
 evidence_event_id BIGINT NOT NULL, UNIQUE KEY identity_key(namespace,entity_key),
 FOREIGN KEY(evidence_event_id) REFERENCES episodic_events(id) ON DELETE CASCADE) ENGINE=InnoDB''',
'''CREATE TABLE IF NOT EXISTS apfc_aliases(
 namespace VARCHAR(32) NOT NULL, scope VARCHAR(160) NOT NULL, alias VARCHAR(160) NOT NULL,
 entity_id BIGINT NOT NULL, evidence_event_id BIGINT NOT NULL,
 PRIMARY KEY(namespace,scope,alias), INDEX(entity_id),
 FOREIGN KEY(entity_id) REFERENCES apfc_entities(id) ON DELETE CASCADE,
 FOREIGN KEY(evidence_event_id) REFERENCES episodic_events(id) ON DELETE CASCADE) ENGINE=InnoDB''',
'''CREATE TABLE IF NOT EXISTS apfc_entity_events(
 entity_id BIGINT NOT NULL,event_id BIGINT NOT NULL, role VARCHAR(160) NOT NULL,
 PRIMARY KEY(event_id,role), INDEX(entity_id,event_id),
 FOREIGN KEY(entity_id) REFERENCES apfc_entities(id) ON DELETE CASCADE,
 FOREIGN KEY(event_id) REFERENCES episodic_events(id) ON DELETE CASCADE) ENGINE=InnoDB''',
'''CREATE TABLE IF NOT EXISTS apfc_cases(
 id BIGINT AUTO_INCREMENT PRIMARY KEY, namespace VARCHAR(32) NOT NULL,
 case_key VARCHAR(160) NOT NULL,title VARCHAR(240) NOT NULL,revision BIGINT NOT NULL DEFAULT 0,
 state JSON NOT NULL,updated TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
 UNIQUE KEY case_key(namespace,case_key)) ENGINE=InnoDB''',
'''CREATE TABLE IF NOT EXISTS apfc_case_events(
 case_id BIGINT NOT NULL,event_id BIGINT NOT NULL,PRIMARY KEY(case_id,event_id), INDEX(event_id),
 FOREIGN KEY(case_id) REFERENCES apfc_cases(id) ON DELETE CASCADE,
 FOREIGN KEY(event_id) REFERENCES episodic_events(id) ON DELETE CASCADE) ENGINE=InnoDB''',
'''CREATE TABLE IF NOT EXISTS apfc_case_revisions(
 case_id BIGINT NOT NULL,revision BIGINT NOT NULL,state JSON NOT NULL,
 request_id VARCHAR(160) NOT NULL,payload_hash CHAR(64) NOT NULL,
 created TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6),PRIMARY KEY(case_id,revision),
 UNIQUE KEY request_key(case_id,request_id),
 FOREIGN KEY(case_id) REFERENCES apfc_cases(id) ON DELETE CASCADE) ENGINE=InnoDB''',
'''CREATE TABLE IF NOT EXISTS apfc_policies(
 id BIGINT AUTO_INCREMENT PRIMARY KEY,namespace VARCHAR(32) NOT NULL,
 policy_key VARCHAR(160) NOT NULL,features JSON NOT NULL,actions JSON NOT NULL,
 UNIQUE KEY policy_key(namespace,policy_key)) ENGINE=InnoDB''',
'''CREATE TABLE IF NOT EXISTS apfc_transitions(
 id BIGINT AUTO_INCREMENT PRIMARY KEY,policy_id BIGINT NOT NULL,
 request_id VARCHAR(160) NOT NULL,payload_hash CHAR(64) NOT NULL,action VARCHAR(160) NOT NULL,
 features JSON NOT NULL,reward DOUBLE NOT NULL,before_event_id BIGINT NOT NULL,outcome_event_id BIGINT NOT NULL,
 created TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6),
 UNIQUE KEY request_key(policy_id,request_id),UNIQUE KEY observed_once(policy_id,outcome_event_id),
 INDEX(policy_id,action,id),
 FOREIGN KEY(policy_id) REFERENCES apfc_policies(id) ON DELETE CASCADE,
 FOREIGN KEY(before_event_id) REFERENCES episodic_events(id) ON DELETE CASCADE,
 FOREIGN KEY(outcome_event_id) REFERENCES episodic_events(id) ON DELETE CASCADE) ENGINE=InnoDB''',
'''CREATE TABLE IF NOT EXISTS apfc_models(
 policy_id BIGINT NOT NULL,action VARCHAR(160) NOT NULL,revision BIGINT NOT NULL,
 examples BIGINT NOT NULL,sample_signature CHAR(64) NOT NULL,
 weights BLOB NOT NULL,precision_matrix BLOB NOT NULL,checksum CHAR(64) NOT NULL,
 PRIMARY KEY(policy_id,action),
 FOREIGN KEY(policy_id) REFERENCES apfc_policies(id) ON DELETE CASCADE) ENGINE=InnoDB'''
]

SCHEMA.append(EXPERIENCE_SCHEMA)


def has_schema(c):
    with c.cursor() as q:
        q.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name='apfc_models'")
        return q.fetchone()[0]==1


def clear_namespace(c,namespace):
    if not has_schema(c):return
    with c.cursor() as q:
        for table in ('apfc_entities','apfc_cases','apfc_policies'):
            q.execute('DELETE FROM '+table+' WHERE namespace=%s',(namespace,))
    with _CACHE_LOCK:_CACHE.clear()


def invalidate_evidence(c,namespace,event_ids,permanent=False):
    if not has_schema(c):return
    marks=','.join(['%s']*len(event_ids));removed=set(event_ids)
    with c.cursor() as q:
        q.execute('''DELETE m FROM apfc_models m JOIN apfc_policies p ON p.id=m.policy_id
         JOIN apfc_transitions t ON t.policy_id=p.id AND t.action=m.action
         WHERE p.namespace=%s AND (t.before_event_id IN ('''+marks+') OR t.outcome_event_id IN ('+marks+'))',[namespace]+event_ids+event_ids)
        if permanent:
            q.execute('''SELECT DISTINCT c.id,c.revision,c.state FROM apfc_cases c
             JOIN apfc_case_events b ON b.case_id=c.id WHERE c.namespace=%s AND b.event_id IN ('''+marks+')',[namespace]+event_ids)
            affected=q.fetchall()
            def scrub(raw):
                state=json.loads(raw)
                return {section:[item for item in items if not (set(item['evidence_event_ids']) & removed)] for section,items in state.items()}
            for ident,revision,raw in affected:
                q.execute('SELECT revision,state FROM apfc_case_revisions WHERE case_id=%s',(ident,))
                for version,old in q.fetchall():
                    q.execute('UPDATE apfc_case_revisions SET state=%s WHERE case_id=%s AND revision=%s',(encode(scrub(old)),ident,version))
                state=encode(scrub(raw));revision+=1
                q.execute('UPDATE apfc_cases SET state=%s,revision=%s WHERE id=%s',(state,revision,ident))
                request='forget_'+hashlib.sha256(encode([event_ids,revision]).encode()).hexdigest()
                q.execute('INSERT INTO apfc_case_revisions(case_id,revision,state,request_id,payload_hash) VALUES(%s,%s,%s,%s,%s)',(ident,revision,state,request,hashlib.sha256(state.encode()).hexdigest()))
    with _CACHE_LOCK:_CACHE.clear()


class LocalModel:
    """Ridge/RLS predictor with an intercept; fixed small memory per action."""
    def __init__(self, weights=None, precision=None):
        self.w=np.zeros(DIM) if weights is None else weights.copy()
        self.p=np.eye(DIM)*10 if precision is None else precision.copy()
    def predict(self,x):return float(x@self.w)
    def observe(self,x,y):
        px=self.p@x;k=px/(1+float(x@px))
        self.w+=k*(y-self.predict(x));self.p-=np.outer(k,px)
        self.p=(self.p+self.p.T)*.5
        if not np.isfinite(self.w).all() or not np.isfinite(self.p).all():
            raise ArithmeticError('Nonfinite APFC operator')


class APFC(ExperienceCoordinator):
    def __init__(self,memory):
        self.c=memory.c;self.namespace=memory.namespace;self._in_atomic=False
    def _begin(self):
        if not self._in_atomic:self.c.begin()
    def _commit(self):
        if not self._in_atomic:self.c.commit()
    def _rollback(self):
        if not self._in_atomic:self.c.rollback()
    def evidence(self, event_ids):
        event_ids=ids(event_ids)
        with self.c.cursor() as q:
            q.execute('SELECT id FROM episodic_events WHERE namespace=%s AND active=1 AND id IN ('+','.join(['%s']*len(event_ids))+')', [self.namespace]+event_ids)
            if {r[0] for r in q.fetchall()}!=set(event_ids):
                raise ValueError('Evidence missing, inactive or outside namespace')
        return event_ids
    def _entity(self,q,key):
        q.execute('SELECT id,label,scope,evidence_event_id FROM apfc_entities WHERE namespace=%s AND entity_key=%s',(self.namespace,canonical(key)))
        row=q.fetchone()
        if row is None:raise ValueError('Unknown entity key')
        self.evidence([row[3]])
        return row
    @retry_transaction
    def bind_entity(self,entity_key,label,scope,aliases,event_ids):
        key=canonical(entity_key);label=canonical(label);scope=canonical(scope)
        if not isinstance(aliases,list) or len(aliases)>16:raise ValueError('At most 16 aliases')
        aliases=sorted(set([label]+[canonical(a) for a in aliases]))
        try:
            self._begin();ev=self.evidence(event_ids)
            with self.c.cursor() as q:
                q.execute('SELECT id,label,scope FROM apfc_entities WHERE namespace=%s AND entity_key=%s FOR UPDATE',(self.namespace,key))
                old=q.fetchone()
                if old and (old[1]!=label or old[2]!=scope):raise ValueError('Stable identity cannot be silently reassigned')
                if not old:
                    q.execute('INSERT INTO apfc_entities(namespace,entity_key,label,scope,evidence_event_id) VALUES(%s,%s,%s,%s,%s)',(self.namespace,key,label,scope,ev[0]));ident=q.lastrowid
                else:ident=old[0]
                for alias in aliases:
                    q.execute('SELECT entity_id FROM apfc_aliases WHERE namespace=%s AND scope=%s AND alias=%s FOR UPDATE',(self.namespace,scope,alias));r=q.fetchone()
                    if r and r[0]!=ident:raise ValueError('Alias collision: explicit identity resolution required')
                    q.execute('INSERT INTO apfc_aliases VALUES(%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE evidence_event_id=VALUES(evidence_event_id)',(self.namespace,scope,alias,ident,ev[0]))
            self._commit();return dict(entity_key=key,entity_id=ident,scope=scope,aliases=aliases)
        except BaseException:self._rollback();raise
    @retry_transaction
    def link_entity(self,entity_key,event_id,role):
        role=canonical(role)
        try:
            self._begin();self.evidence([event_id])
            with self.c.cursor() as q:
                entity=self._entity(q,entity_key)
                q.execute('SELECT 1 FROM episodic_bindings WHERE event_id=%s AND role=%s',(event_id,role))
                if q.fetchone() is None:raise ValueError('Role not present in event')
                q.execute('SELECT entity_id FROM apfc_entity_events WHERE event_id=%s AND role=%s FOR UPDATE',(event_id,role));old=q.fetchone()
                if old and old[0]!=entity[0]:raise ValueError('Role already bound to another identity')
                q.execute('INSERT IGNORE INTO apfc_entity_events VALUES(%s,%s,%s)',(entity[0],event_id,role))
            self._commit();return dict(entity_key=canonical(entity_key),event_id=event_id,role=role)
        except BaseException:self._rollback();raise
    def resolve(self,alias,scope=None,limit=20):
        alias=canonical(alias)
        if not 1<=limit<=50:raise ValueError('limit must be 1..50')
        self._commit()
        with self.c.cursor() as q:
            sql='''SELECT DISTINCT n.entity_key,n.label,n.scope FROM apfc_aliases a
             JOIN apfc_entities n ON n.id=a.entity_id
             JOIN episodic_events e ON e.id=a.evidence_event_id AND e.active=1
             JOIN episodic_events origin ON origin.id=n.evidence_event_id AND origin.active=1
             WHERE a.namespace=%s AND a.alias=%s'''
            params=[self.namespace,alias]
            if scope is not None:sql+=' AND a.scope=%s';params.append(canonical(scope))
            q.execute(sql+' ORDER BY n.id LIMIT %s',params+[limit+1]);rows=q.fetchall()
        return dict(entities=[dict(entity_key=r[0],label=r[1],scope=r[2]) for r in rows[:limit]],has_more=len(rows)>limit,ambiguous=len(rows)>1)
    def entity_context(self,entity_key,limit=20):
        if not 1<=limit<=50:raise ValueError('limit must be 1..50')
        self._commit()
        with self.c.cursor() as q:
            n=self._entity(q,entity_key)
            q.execute('''SELECT e.id,e.frame,e.roles,e.source,e.negated FROM apfc_entity_events b
             JOIN episodic_events e ON e.id=b.event_id WHERE b.entity_id=%s AND e.namespace=%s AND e.active=1
             ORDER BY e.id DESC LIMIT %s''',(n[0],self.namespace,limit+1));rows=q.fetchall()
        return dict(entity_key=canonical(entity_key),events=[dict(event_id=r[0],frame=r[1],roles=json.loads(r[2]),source=r[3],negated=bool(r[4])) for r in rows[:limit]],has_more=len(rows)>limit)
    @retry_transaction
    def update_case(self,case_key,title,patch,expected_revision,request_id):
        return self._update_case(case_key,title,patch,expected_revision,request_id)
    def _update_case(self,case_key,title,patch,expected_revision,request_id):
        key=canonical(case_key);title=text(title,240);request_id=canonical(request_id)
        if type(expected_revision) is not int or expected_revision<0:raise ValueError('Invalid revision')
        if not isinstance(patch,dict) or not patch or not set(patch)<=SECTIONS:raise ValueError('Unknown case state section')
        for section,items in patch.items():
            if not isinstance(items,list) or len(items)>16:raise ValueError('At most 16 items per section')
            for item in items:
                if not isinstance(item,dict) or set(item)!={'text','evidence_event_ids'}:raise ValueError('Each state item needs text and evidence_event_ids')
                text(item['text']);ids(item['evidence_event_ids'])
        digest=hashlib.sha256(encode(dict(title=title,patch=patch,expected_revision=expected_revision)).encode()).hexdigest()
        try:
            self._begin()
            with self.c.cursor() as q:
                q.execute('SELECT id,revision,state FROM apfc_cases WHERE namespace=%s AND case_key=%s FOR UPDATE',(self.namespace,key));row=q.fetchone()
                if not row:
                    if expected_revision!=0:raise ValueError('Revision conflict')
                    q.execute('INSERT INTO apfc_cases(namespace,case_key,title,state) VALUES(%s,%s,%s,%s)',(self.namespace,key,title,'{}'));row=(q.lastrowid,0,'{}')
                ident,revision,raw=row
                q.execute('SELECT revision,payload_hash FROM apfc_case_revisions WHERE case_id=%s AND request_id=%s',(ident,request_id));previous=q.fetchone()
                if previous:
                    if previous[1]!=digest:raise ValueError('Idempotency key reused with different payload')
                    self._rollback();return dict(case_key=key,revision=previous[0],duplicate=True)
                if revision!=expected_revision:raise ValueError('Revision conflict; re-read case before updating')
                state=json.loads(raw);state.update(patch)
                if len(encode(state).encode())>24000:raise ValueError('Case state exceeds 24000 UTF-8 bytes; use linked evidence')
                refs=sorted({i for items in state.values() for item in items for i in item['evidence_event_ids']})
                # Check references in batches, allowing a larger linked state than one write input.
                for offset in range(0,len(refs),32):self.evidence(refs[offset:offset+32])
                revision+=1
                q.execute('UPDATE apfc_cases SET title=%s,state=%s,revision=%s WHERE id=%s',(title,encode(state),revision,ident))
                q.execute('INSERT INTO apfc_case_revisions(case_id,revision,state,request_id,payload_hash) VALUES(%s,%s,%s,%s,%s)',(ident,revision,encode(state),request_id,digest))
                for event_id in refs:q.execute('INSERT IGNORE INTO apfc_case_events VALUES(%s,%s)',(ident,event_id))
            self._commit();return dict(case_key=key,revision=revision,duplicate=False)
        except BaseException:self._rollback();raise
    def get_case(self,case_key):
        self._commit()
        with self.c.cursor() as q:
            q.execute('SELECT id,title,revision,state FROM apfc_cases WHERE namespace=%s AND case_key=%s',(self.namespace,canonical(case_key)));row=q.fetchone()
            if not row:return dict(status='unknown',case_key=canonical(case_key))
            q.execute('SELECT e.id FROM apfc_case_events b JOIN episodic_events e ON e.id=b.event_id WHERE b.case_id=%s AND e.namespace=%s AND e.active=1',(row[0],self.namespace));active={r[0] for r in q.fetchall()}
        state=json.loads(row[3]);omitted=0
        for key,items in state.items():
            kept=[x for x in items if set(x['evidence_event_ids'])<=active];omitted+=len(items)-len(kept);state[key]=kept
        return dict(status='needs_review' if omitted else 'available',case_key=canonical(case_key),title=row[1],revision=row[2],state=state,unsupported_items_omitted=omitted,experience_summary=self.experiences(case_key,limit=3),state_is_bounded_working_set=True,scope='Evidence-bound operational state; language interpretation remains external')
    def related_cases(self,event_ids,limit=3):
        if not isinstance(event_ids,list) or len(event_ids)>32 or any(type(i)is not int or i<1 for i in event_ids):raise ValueError('Invalid event IDs')
        if not 1<=limit<=8:raise ValueError('limit must be 1..8')
        if not event_ids:return dict(cases=[],has_more=False)
        self._commit()
        with self.c.cursor() as q:
            q.execute('''SELECT c.case_key,COUNT(*) AS matches FROM apfc_cases c
             JOIN apfc_case_events b ON b.case_id=c.id JOIN episodic_events e ON e.id=b.event_id
             WHERE c.namespace=%s AND e.active=1 AND b.event_id IN ('''+','.join(['%s']*len(event_ids))+') GROUP BY c.id,c.case_key ORDER BY matches DESC,c.id DESC LIMIT %s',[self.namespace]+event_ids+[limit+1]);rows=q.fetchall()
        return dict(cases=[self.get_case(r[0]) for r in rows[:limit]],has_more=len(rows)>limit)
    @retry_transaction
    def register_policy(self,key,features,actions):
        key=canonical(key)
        if not isinstance(features,list) or not 1<=len(features)<=16 or not isinstance(actions,list) or not 1<=len(actions)<=16:raise ValueError('Use 1..16 features and actions')
        features=[canonical(f) for f in features];actions=[canonical(a) for a in actions]
        if len(set(features))!=len(features) or len(set(actions))!=len(actions):raise ValueError('Duplicate feature or action')
        try:
            self._begin()
            with self.c.cursor() as q:
                q.execute('SELECT features,actions FROM apfc_policies WHERE namespace=%s AND policy_key=%s FOR UPDATE',(self.namespace,key));old=q.fetchone()
                if old and (json.loads(old[0])!=features or json.loads(old[1])!=actions):raise ValueError('Policy schema immutable; use a new policy key for new units or features')
                if not old:q.execute('INSERT INTO apfc_policies(namespace,policy_key,features,actions) VALUES(%s,%s,%s,%s)',(self.namespace,key,encode(features),encode(actions)))
            self._commit();return dict(policy_key=key,features=features,actions=actions)
        except BaseException:self._rollback();raise
    def policy(self,key):
        with self.c.cursor() as q:
            q.execute('SELECT id,features,actions FROM apfc_policies WHERE namespace=%s AND policy_key=%s',(self.namespace,canonical(key)));row=q.fetchone()
        if not row:raise ValueError('Unknown policy')
        return row[0],json.loads(row[1]),json.loads(row[2])
    def vector(self,features,names):
        if not isinstance(features,dict) or set(features)!=set(names):raise ValueError('Feature names must exactly match policy schema')
        if any(type(v)not in (int,float) or not math.isfinite(v) or not -1<=v<=1 for v in features.values()):raise ValueError('Features must be finite numbers normalized to [-1,1]')
        x=np.zeros(DIM);x[0]=1;x[1:len(names)+1]=[features[n] for n in names];return x
    def samples(self,policy_id,action):
        with self.c.cursor() as q:
            q.execute('''SELECT t.id,t.features,t.reward FROM apfc_transitions t
             JOIN episodic_events b ON b.id=t.before_event_id AND b.active=1
             JOIN episodic_events o ON o.id=t.outcome_event_id AND o.active=1
             WHERE t.policy_id=%s AND t.action=%s AND b.namespace=%s AND o.namespace=%s
             ORDER BY t.id LIMIT %s''',(policy_id,action,self.namespace,self.namespace,MAX_SAMPLES+1));rows=q.fetchall()
        if len(rows)>MAX_SAMPLES:raise ValueError('Policy observation budget exceeded; consolidate explicitly before more training')
        signature=hashlib.sha256(encode(rows).encode()).hexdigest();return rows,signature
    def model(self,policy_id,action,names):
        rows,signature=self.samples(policy_id,action)
        key=(self.c.db,self.namespace,policy_id,action,tuple(names),signature)
        model=None
        with self.c.cursor() as q:
            q.execute('SELECT sample_signature,weights,precision_matrix,checksum FROM apfc_models WHERE policy_id=%s AND action=%s',(policy_id,action));stored=q.fetchone()
        if stored and stored[0]==signature:
            if hashlib.sha256(stored[1]+stored[2]).hexdigest()!=stored[3]:raise ValueError('APFC operator checksum mismatch')
            w=np.frombuffer(stored[1],dtype='<f8');p=np.frombuffer(stored[2],dtype='<f8')
            if w.size!=DIM or p.size!=DIM*DIM or not np.isfinite(w).all() or not np.isfinite(p).all():raise ValueError('Invalid APFC operator')
            with _CACHE_LOCK:
                if key in _CACHE:
                    cached=_CACHE.pop(key);_CACHE[key]=cached;return cached[0],len(rows),signature
            model=LocalModel(w,p.reshape(DIM,DIM))
        if model is None:
            model=LocalModel()
            for _,raw,y in rows:model.observe(self.vector(json.loads(raw),names),y)
        with _CACHE_LOCK:
            _CACHE[key]=(model,len(rows))
            while len(_CACHE)>32:_CACHE.popitem(last=False)
        return model,len(rows),signature
    @retry_transaction
    def observe(self,key,action,features,reward,before_event_id,outcome_event_id,request_id):
        return self._observe(key,action,features,reward,before_event_id,outcome_event_id,request_id)
    def _observe(self,key,action,features,reward,before_event_id,outcome_event_id,request_id,compound_evidence=False):
        action=canonical(action);request_id=canonical(request_id)
        if type(reward)not in (int,float) or not math.isfinite(reward) or not 0<=reward<=1:raise ValueError('Observed reward must be finite in [0,1]')
        if before_event_id==outcome_event_id and not compound_evidence:raise ValueError('Use distinct before and outcome evidence')
        digest=hashlib.sha256(encode([key,action,features,reward,before_event_id,outcome_event_id]).encode()).hexdigest()
        try:
            self._begin();self.evidence([before_event_id,outcome_event_id]);policy_id,names,actions=self.policy(key)
            if action not in actions:raise ValueError('Unknown action')
            x=self.vector(features,names)
            with self.c.cursor() as q:
                q.execute('SELECT id,payload_hash FROM apfc_transitions WHERE policy_id=%s AND request_id=%s',(policy_id,request_id));old=q.fetchone()
                if old:
                    if old[1]!=digest:raise ValueError('Idempotency key reused with different observation')
                    self._rollback();return dict(transition_id=old[0],duplicate=True)
                model,count,_=self.model(policy_id,action,names)
                if count>=MAX_SAMPLES:raise ValueError('Policy observation budget exceeded')
                before=model.predict(x);model=LocalModel(model.w,model.p);model.observe(x,reward)
                q.execute('INSERT INTO apfc_transitions(policy_id,request_id,payload_hash,action,features,reward,before_event_id,outcome_event_id) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)',(policy_id,request_id,digest,action,encode(features),reward,before_event_id,outcome_event_id));ident=q.lastrowid
                _,signature=self.samples(policy_id,action)
                w=model.w.astype('<f8').tobytes();p=model.p.astype('<f8').tobytes()
                q.execute('''INSERT INTO apfc_models VALUES(%s,%s,1,%s,%s,%s,%s,%s)
                 ON DUPLICATE KEY UPDATE revision=revision+1,examples=VALUES(examples),sample_signature=VALUES(sample_signature),weights=VALUES(weights),precision_matrix=VALUES(precision_matrix),checksum=VALUES(checksum)''',(policy_id,action,count+1,signature,w,p,hashlib.sha256(w+p).hexdigest()))
            self._commit();return dict(transition_id=ident,duplicate=False,examples=count+1,prediction_before=before,prediction_after=model.predict(x),observed_reward=reward,scope='Local numerical readout updated; LLM weights unchanged')
        except BaseException:self._rollback();raise
    def predict(self,key,features):
        start=time.perf_counter();self._commit();policy_id,names,actions=self.policy(key);x=self.vector(features,names);results=[]
        for action in actions:
            model,n,_=self.model(policy_id,action,names)
            results.append(dict(action=action,examples=n,estimated_reward=None if not n else float(np.clip(model.predict(x),0,1)),novelty=float(x@model.p@x),status='untrained' if not n else 'hypothesis'))
        return dict(policy_key=canonical(key),actions=results,seconds=time.perf_counter()-start,automatic_execution=False,scope='Uncalibrated action-outcome estimates, not truth or permission to act; evaluate on unseen observations')
    def status(self):
        self._commit();counts={}
        with self.c.cursor() as q:
            for table in ('apfc_entities','apfc_cases','apfc_policies'):
                q.execute('SELECT COUNT(*) FROM '+table+' WHERE namespace=%s',(self.namespace,));counts[table]=q.fetchone()[0]
        with self.c.cursor() as q:
            q.execute("SELECT x.status,COUNT(*) FROM apfc_experiences x JOIN apfc_cases c ON c.id=x.case_id JOIN episodic_events b ON b.id=x.before_event_id AND b.active=1 LEFT JOIN episodic_events o ON o.id=x.outcome_event_id WHERE c.namespace=%s AND (x.status='pending' OR o.active=1) GROUP BY x.status",(self.namespace,))
            experience_counts=dict(q.fetchall())
            q.execute('SELECT COUNT(*) FROM apfc_transitions t JOIN apfc_policies p ON p.id=t.policy_id JOIN episodic_events b ON b.id=t.before_event_id AND b.active=1 JOIN episodic_events o ON o.id=t.outcome_event_id AND o.active=1 WHERE p.namespace=%s',(self.namespace,));observations=q.fetchone()[0]
        return dict(namespace=self.namespace,counts=counts,experiences=experience_counts,learned_observations=observations,feature_dimension_max=16,cached_models=len(_CACHE),cache_limit=32)
    def delete_test_namespace(self):
        if not self.namespace.startswith('check_'):raise ValueError('Refusing operational deletion')
        with self.c.cursor() as q:
            for table in ('apfc_entities','apfc_cases','apfc_policies'):q.execute('DELETE FROM '+table+' WHERE namespace=%s',(self.namespace,))
        self._commit()


if __name__ == '__main__':
    import argparse
    from episodic_memory import EpisodicMemory
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',default='status',choices=['migrate','status','benchmark'])
    args=parser.parse_args()
    if args.command=='benchmark':
        from verify_apfc import benchmark
        print(json.dumps(benchmark(),indent=2));raise SystemExit(0)
    m=EpisodicMemory()
    try:
        if args.command=='migrate':ensure_schema(m.c);print('APFC additive schema ready; existing memories unchanged.')
        else:print(encode(APFC(m).status()))
    finally:m.close()
