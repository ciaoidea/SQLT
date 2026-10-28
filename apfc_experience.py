"""Atomic bridge between recorded interventions, observed outcomes and learning.

Language clients supply structured meaning and exact evidence quotations.
This module neither parses sentences nor executes external actions.
"""
from contextlib import contextmanager
import hashlib
import json
from episodic_memory import canonical
from storage import retry_transaction

EXPERIENCE_SCHEMA = '''CREATE TABLE IF NOT EXISTS apfc_experiences(
 id BIGINT AUTO_INCREMENT PRIMARY KEY,case_id BIGINT NOT NULL,
 experience_key VARCHAR(160) NOT NULL,policy_id BIGINT NOT NULL,action VARCHAR(160) NOT NULL,
 features JSON NOT NULL,before_event_id BIGINT NOT NULL,before_quote VARCHAR(1000) NOT NULL DEFAULT '',
 outcome_definition VARCHAR(1000) NOT NULL,pending_item JSON NOT NULL,
 status VARCHAR(16) NOT NULL DEFAULT 'pending',begin_revision BIGINT NOT NULL,
 begin_hash CHAR(64) NOT NULL,outcome_event_id BIGINT NULL,
 observed_quote TEXT NULL,evidence_kind VARCHAR(32) NULL,reward DOUBLE NULL,
 transition_id BIGINT NULL,completion_hash CHAR(64) NULL,completion_revision BIGINT NULL,
 created TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6),completed_at TIMESTAMP(6) NULL,
 UNIQUE KEY experience_key(case_id,experience_key),INDEX(case_id,status,id),INDEX(policy_id),
 FOREIGN KEY(case_id) REFERENCES apfc_cases(id) ON DELETE CASCADE,
 FOREIGN KEY(policy_id) REFERENCES apfc_policies(id) ON DELETE CASCADE,
 FOREIGN KEY(before_event_id) REFERENCES episodic_events(id) ON DELETE CASCADE,
 FOREIGN KEY(outcome_event_id) REFERENCES episodic_events(id) ON DELETE CASCADE,
 FOREIGN KEY(transition_id) REFERENCES apfc_transitions(id) ON DELETE SET NULL
) ENGINE=InnoDB'''


class ExperienceCoordinator:
    @contextmanager
    def atomic(self):
        if self._in_atomic:raise RuntimeError('Nested APFC coordinator transaction')
        self.c.begin();self._in_atomic=True
        try:
            yield
            self.c.commit()
        except BaseException:
            self.c.rollback();raise
        finally:self._in_atomic=False

    def _case_row(self,key):
        with self.c.cursor() as q:
            q.execute('SELECT id,title,revision,state FROM apfc_cases WHERE namespace=%s AND case_key=%s FOR UPDATE',(self.namespace,canonical(key)))
            return q.fetchone()

    def _experience_row(self,case_key,experience_key):
        with self.c.cursor() as q:
            q.execute('''SELECT x.id,x.case_id,x.policy_id,x.action,x.features,x.before_event_id,
                x.status,x.pending_item,x.begin_revision,x.begin_hash,x.outcome_event_id,
                x.completion_hash,x.completion_revision,x.transition_id,c.title,c.revision,c.state,p.policy_key,x.before_quote
                FROM apfc_experiences x JOIN apfc_cases c ON c.id=x.case_id
                JOIN apfc_policies p ON p.id=x.policy_id
                WHERE c.namespace=%s AND c.case_key=%s AND x.experience_key=%s FOR UPDATE''',
                (self.namespace,canonical(case_key),canonical(experience_key)))
            return q.fetchone()

    @retry_transaction
    def start_experience(self,case_key,title,experience_key,policy_key,action,features,
                         before_event_id,outcome_definition,expected_revision,actions=None,before_quote=''):
        from apfc import encode,text
        key=canonical(case_key);experience_key=canonical(experience_key)
        policy_key=canonical(policy_key);action=canonical(action);title=text(title,240)
        outcome_definition=text(outcome_definition)
        if before_quote:before_quote=text(before_quote)
        if not isinstance(features,dict) or not features:raise ValueError('Explicit pre-action numerical features required')
        if any(canonical(n)!=n for n in features):raise ValueError('Use canonical feature names from the policy')
        digest=hashlib.sha256(encode([title,policy_key,action,features,before_event_id,outcome_definition,expected_revision,actions,before_quote]).encode()).hexdigest()
        with self.atomic():
            self.evidence([before_event_id])
            if before_quote:
                with self.c.cursor() as q:
                    q.execute('SELECT source FROM episodic_events WHERE namespace=%s AND id=%s',(self.namespace,before_event_id))
                    if before_quote not in q.fetchone()[0]:raise ValueError('Before quote must be an exact excerpt of the source')
            row=self._experience_row(key,experience_key)
            if row:
                if row[9]!=digest:raise ValueError('Experience key reused with different intervention; use a new key')
                return dict(experience_key=experience_key,status=row[6],case_revision=row[8],duplicate=True,learning_updated=False)
            case=self._case_row(key)
            if (case[2] if case else 0)!=expected_revision:raise ValueError('Revision conflict; re-read case')
            with self.c.cursor() as q:
                q.execute('SELECT id,features,actions FROM apfc_policies WHERE namespace=%s AND policy_key=%s',(self.namespace,policy_key))
                policy=q.fetchone()
                if policy:
                    policy_id,names,known_actions=policy[0],json.loads(policy[1]),json.loads(policy[2])
                    if actions is not None and [canonical(a) for a in actions]!=known_actions:
                        raise ValueError('Existing action schema differs; use a new policy key')
                else:
                    names=sorted(features);known_actions=[canonical(a) for a in (actions or [action])]
                    # Use the same validation as existing policies without an inner transaction.
                    if not 1<=len(names)<=16 or not 1<=len(known_actions)<=16 or len(set(known_actions))!=len(known_actions):
                        raise ValueError('Use 1..16 unique features/actions')
                    q.execute('INSERT INTO apfc_policies(namespace,policy_key,features,actions) VALUES(%s,%s,%s,%s)',(self.namespace,policy_key,encode(names),encode(known_actions)))
                    policy_id=q.lastrowid
                self.vector(features,names)
                if action not in known_actions:raise ValueError('Unknown action for policy')
                q.execute('SELECT outcome_definition FROM apfc_experiences WHERE policy_id=%s LIMIT 1',(policy_id,))
                contract=q.fetchone()
                if contract and contract[0]!=outcome_definition:raise ValueError('Outcome criterion differs; use a new policy version')
            item=dict(text='Outcome pending for '+action+': '+outcome_definition,evidence_event_ids=[before_event_id])
            state=json.loads(case[3]) if case else {}
            waiting=state.get('waiting',[])
            patch={'waiting':(waiting+[item])[-16:]}
            result=self._update_case(key,title,patch,expected_revision,'experience_start_'+hashlib.sha256(experience_key.encode()).hexdigest())
            case=self._case_row(key)
            with self.c.cursor() as q:
                q.execute('''INSERT INTO apfc_experiences(case_id,experience_key,policy_id,action,features,
                 before_event_id,outcome_definition,pending_item,begin_revision,begin_hash,before_quote)
                 VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',(case[0],experience_key,policy_id,action,encode(features),before_event_id,outcome_definition,encode(item),result['revision'],digest,before_quote))
        return dict(experience_key=experience_key,status='pending',case_revision=result['revision'],duplicate=False,
                    learning_updated=False,waiting_for='Explicit observed outcome',state_windowed=len(waiting)>=16)

    @retry_transaction
    def complete_experience(self,case_key,experience_key,outcome_event_id,observed_quote,
                            reward,evidence_kind,expected_revision):
        from apfc import encode,text
        key=canonical(case_key);experience_key=canonical(experience_key);observed_quote=text(observed_quote)
        if evidence_kind not in ('user_report','tool_observation'):raise ValueError('An explicit reported or tool-observed outcome is required')
        digest=hashlib.sha256(encode([outcome_event_id,observed_quote,reward,evidence_kind]).encode()).hexdigest()
        with self.atomic():
            row=self._experience_row(key,experience_key)
            if not row:raise ValueError('Unknown intervention; record the pre-action context first')
            self.evidence([row[5],outcome_event_id])
            if row[6]=='completed':
                if row[11]!=digest:raise ValueError('Contradictory completion; preserve new evidence and review, do not overwrite')
                return dict(experience_key=experience_key,status='completed',case_revision=row[12],transition_id=row[13],duplicate=True,learning_updated=False)
            if row[15]!=expected_revision:raise ValueError('Revision conflict; re-read case before resolving outcome')
            with self.c.cursor() as q:
                q.execute('SELECT source FROM episodic_events WHERE namespace=%s AND id=%s AND active=1',(self.namespace,outcome_event_id))
                source=q.fetchone()[0]
                if observed_quote not in source:raise ValueError('Observed quote must be an exact excerpt of the outcome source')
                if row[5]==outcome_event_id:
                    initial=row[18]
                    if not initial or initial not in source:raise ValueError('A compound source requires an anchored before quote')
                    a=source.index(initial);b=source.index(observed_quote)
                    if max(a,b)<min(a+len(initial),b+len(observed_quote)):
                        raise ValueError('Before and outcome quotes must be distinct non-overlapping evidence')
            observation=self._observe(row[17],row[3],json.loads(row[4]),reward,row[5],outcome_event_id,
                                      'experience_'+str(row[0]),compound_evidence=(row[5]==outcome_event_id))
            state=json.loads(row[16]);pending=json.loads(row[7])
            waiting=[item for item in state.get('waiting',[]) if item!=pending]
            completed=state.get('completed',[])+[dict(text='Observed outcome recorded for '+row[3],evidence_event_ids=sorted({row[5],outcome_event_id}))]
            confirmed=state.get('confirmed',[])+[dict(text=observed_quote,evidence_event_ids=[outcome_event_id])]
            patch=dict(waiting=waiting,completed=completed[-16:],confirmed=confirmed[-16:])
            result=self._update_case(key,row[14],patch,expected_revision,'experience_finish_'+str(row[0]))
            with self.c.cursor() as q:
                q.execute('''UPDATE apfc_experiences SET status='completed',outcome_event_id=%s,
                 observed_quote=%s,evidence_kind=%s,reward=%s,transition_id=%s,completion_hash=%s,
                 completion_revision=%s,completed_at=CURRENT_TIMESTAMP(6) WHERE id=%s''',
                 (outcome_event_id,observed_quote,evidence_kind,reward,observation['transition_id'],digest,result['revision'],row[0]))
        return dict(experience_key=experience_key,status='completed',case_revision=result['revision'],transition_id=observation['transition_id'],duplicate=False,learning_updated=not observation['duplicate'],examples=observation.get('examples'),prediction_before=observation.get('prediction_before'),prediction_after=observation.get('prediction_after'),state_windowed=len(completed)>16 or len(confirmed)>16,scope='Observed outcome updated case state and local numerical parameters in one transaction; no external action executed')

    def experiences(self,case_key,status=None,limit=8,offset=0):
        from apfc import text
        key=canonical(case_key)
        if status not in (None,'pending','completed'):raise ValueError('Invalid experience status')
        if type(limit)is not int or not 1<=limit<=32 or type(offset)is not int or offset<0:raise ValueError('Invalid pagination')
        self._commit()
        with self.c.cursor() as q:
            where='c.namespace=%s AND c.case_key=%s AND b.active=1';params=[self.namespace,key]
            if status:where+=' AND x.status=%s';params.append(status)
            joins=''' FROM apfc_experiences x JOIN apfc_cases c ON c.id=x.case_id
             JOIN apfc_policies p ON p.id=x.policy_id JOIN episodic_events b ON b.id=x.before_event_id
             LEFT JOIN episodic_events o ON o.id=x.outcome_event_id'''
            q.execute('SELECT COUNT(*)'+joins+' WHERE '+where,params);total=q.fetchone()[0]
            q.execute('''SELECT x.experience_key,x.action,x.features,x.before_event_id,x.status,
             x.outcome_definition,x.outcome_event_id,x.observed_quote,x.reward,p.policy_key,o.active'''+joins+
             ' WHERE '+where+" ORDER BY (x.status='pending') DESC,x.id DESC LIMIT %s OFFSET %s",params+[limit,offset])
            rows=q.fetchall()
        results=[]
        for r in rows:
            valid=r[4]=='pending' or r[10]==1
            results.append(dict(experience_key=r[0],action=r[1],features=json.loads(r[2]),before_event_id=r[3],
                 status=r[4] if valid else 'needs_review',outcome_definition=r[5],outcome_event_id=r[6],
                 observed_quote=r[7] if valid else None,reward=r[8] if valid else None,policy_key=r[9]))
        more=offset+len(results)<total
        return dict(experiences=results,total=total,has_more=more,next_offset=offset+len(results) if more else None)
