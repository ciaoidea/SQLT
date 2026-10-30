"""Local MCP access to Cortex memory; never loads an LLM or accepts raw SQL."""
import os
import sys
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
from contextlib import contextmanager
from typing import Annotated
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from episodic_memory import EpisodicMemory
import semantic_memory
import relational_memory
from live_dynamics import observed

mcp=FastMCP('cortex',instructions='Cortex is persistent semantic vector memory. Prepare with prepare_semantic_memory on connection, Use query_memory for complete lists, counts, numeric filters and exact field lookups; semantic_recall for meaning-based questions. A local non-generative multilingual encoder supplies semantic vectors; you evaluate source evidence and generate language. Preserve agent/patient/recipient roles and negation. Quote original user evidence, distinguish observations from predictions. Do not fabricate observations. For ongoing practices use evidence-bound apfc_case/apfc_update_case state. For performed interventions use apfc_start_experience, then apfc_complete_experience on explicit observed outcomes to update state and numerical learning atomically. Missing outcomes remain pending. APFC predictions are local hypotheses and never execute actions.')

class Binding(BaseModel):
    role: str=Field(min_length=1,max_length=160,description='Stable semantic role, such as agent, patient or recipient; not a grammatical word position')
    value: str=Field(min_length=1,max_length=160,description='Participant or object label. Preserve identity across paraphrases.')

class MemoryEvent(BaseModel):
    frame: str=Field(min_length=1,max_length=160)
    roles: list[Binding]=Field(min_length=1,max_length=12)
    source: str=Field(min_length=1,max_length=60000,description='Original user evidence for this complete event or entity, not generated prose')
    negated: bool=False
    event_time: str=Field(default='',max_length=160)
    place: str=Field(default='',max_length=160)

class MemoryFilter(BaseModel):
    role: str=Field(min_length=1,max_length=160,description='Stored role name; discover with memory_status if unknown')
    operator: Literal['eq','ne','in','contains','prefix','number_eq','gt','gte','lt','lte','exists','missing']='eq'
    value: str | int | float | list[str] | None=None

@contextmanager
def memory(namespace):
    m=EpisodicMemory(namespace)
    try:yield m
    finally:m.close()

READ=ToolAnnotations(readOnlyHint=True,destructiveHint=False,openWorldHint=False)
WRITE=ToolAnnotations(readOnlyHint=False,destructiveHint=False,openWorldHint=False)

@mcp.tool(annotations=WRITE)
def prepare_semantic_memory(namespace: str='main') -> dict:
    """Warm the local multilingual semantic encoder and index up to 32 old episodes.

    This only adds derived semantic vectors; no observations are created or
    modified. Re-run if remaining is nonzero. New memories are indexed atomically
    on remember_event. The frozen encoder does not load a generative local LLM.
    """
    with memory(namespace) as m:
        indexed=semantic_memory.backfill(m)
        return dict(**indexed,**m.status(),encoder=semantic_memory.MODEL,
                    dimension=semantic_memory.DIM,local_generative_llm_loaded=False)

@mcp.tool(annotations=READ)
@observed
def semantic_recall(query: str, limit: int=8, negated: bool | None=None,
                    namespace: str='main',include_sql: bool=False,
                    filters: Annotated[list[MemoryFilter],Field(max_length=16)] | None=None,
                    frame: str | None=None) -> dict:
    """Search active experiences by MEANING from the original natural-language query.

    Use for meaning-based questions, not exhaustive SQL lists/counts (query_memory). No exact frame, role or entity
    spelling is required. A local learned multilingual encoder maps query and
    original source text to 384D vectors; MariaDB ranks cosine distance.
    Results are candidates: inspect source evidence, functional roles, negation,
    time and identity. Similarity is not logical entailment or calibrated truth.
    By default both positive and negative experiences are returned and explicitly
    marked. Never assert a fact based on mere topic similarity. Use recall_events
    for exact role/context refinement when needed. Archived episodes are excluded.
    Optional filters/frame constrain candidates in SQL BEFORE vector ranking.
    include_sql returns the executed template and bound parameters (the query
    vector is represented as base64 float32 with a checksum).
    """
    with memory(namespace) as m:return semantic_memory.search(m,query,limit,negated,include_sql,
                                                            [f.model_dump() for f in filters] if filters else None,frame)

@mcp.tool(annotations=READ)
@observed
def query_memory(filters: Annotated[list[MemoryFilter],Field(max_length=16)],
                 frame: str | None=None,negated: bool | None=False,
                 limit: int=100,offset: int=0,count_only: bool=False,
                 select_roles: list[str] | None=None,include_source: bool=True,namespace: str='main',include_sql: bool=False) -> dict:
    """SELECT/WHERE over ALL active facts; lists, counts, numeric filters and pagination.

    Use directly for exhaustive lists/counts and exact field lookups, rather than
    semantic_recall top-k. All filters are ANDed on the SAME event. Discover role
    names with memory_status only if unknown. Example FTTH at 1000 Mbps:
    filters=[{"role":"line_type","value":"ftth"},
             {"role":"download_mbps","operator":"number_eq","value":1000}].
    For a branch name inside an entity label, use contains on role entity and
    inspect identity/source. eq/ne/in match canonical labels; contains/prefix are
    literal substring operations, not SQL wildcards. Numeric operators accept
    only strictly numeric stored labels, not strings like '1000 300'. Missing
    fields do not match ne or numeric comparisons; use missing explicitly.
    Positive facts only by default; negated=None includes both, marked separately.
    total is the exact matched event count (not distinct entities). count_only
    avoids fetching rows. Read next_offset while has_more before claiming a full
    list. For compact lists select_roles=["entity","line_type","download_mbps"]
    and include_source=False return only needed fields; omit these options when
    checking original evidence. Each call uses a consistent read snapshot.
    include_sql returns executed SELECT templates and their parameters for local
    inspection. No SQL text is accepted from the caller.
    """
    with memory(namespace) as m:
        return relational_memory.query(m,[f.model_dump() for f in filters],frame,negated,limit,offset,count_only,
                                       select_roles,include_source,include_sql)

@mcp.tool(annotations=READ)
def focus_memory(request_id: str,
                 event_ids: Annotated[list[Annotated[int, Field(strict=True,gt=0)]],Field(max_length=128)],
                 namespace: str='main') -> dict:
    """Select only retrieved facts that actually support the CURRENT user request.

    Use the live_graph_request_id supplied by the Cortex client. Before the final
    answer, select supporting event IDs after inspecting their source meaning,
    identity, scope, time and negation. Similarity or mere co-retrieval is not
    relevance. Exclude unrelated personal/work contexts. Select [] if none help.
    No database read/write or extra language-model call. This controls an optional
    live graph only. The viewer accepts the active request ID and known records;
    this is a model selection, not independent verification or a memory update.
    """
    from live_dynamics import request_focus
    return request_focus(namespace,request_id,event_ids)

@mcp.tool(annotations=READ)
def memory_status(namespace: str='main') -> dict:
    """Get counts, known frames/roles and a bounded catalog of participant labels.

    Interpret paraphrases using these stored labels, not invented exact filters.
    The catalog may be truncated; its absence is not evidence of missing memory.
    No model loading.
    """
    with memory(namespace) as m:return dict(**m.status(),vocabulary=m.vocabulary(),archived_vocabulary=m.vocabulary(active=False),language_backend='external MCP client',local_llm_loaded=('torch' in sys.modules or 'sql_language_channel' in sys.modules),rss_mib=int(next(line for line in open('/proc/self/status') if line.startswith('VmRSS:')).split()[1])/1024)

@mcp.tool(annotations=WRITE)
@observed
def remember_events(events: Annotated[list[MemoryEvent],Field(min_length=1,max_length=1024)],
                    namespace: str='main') -> dict:
    """Store user-supplied events in one call, using bounded atomic batches.

    Use for pasted lists, tables or several facts. Interpret the entire message
    first. Group all properties of one entity/branch (company, city, address,
    phone, email) in ONE coherent event with role bindings, not separate calls
    for each line or field. Distinguish branches by company AND location; keep
    shared header context in the bindings and quote original evidence in source.
    Do not invent missing fields. Up to 1024 events per request; the server splits
    into transactions of at most 64 events and 64000 UTF-8 source bytes. Each
    source is at most 60000 UTF-8 bytes. Larger documents need multiple sourced
    events. Return one overall confirmation. A partial result lists committed
    event IDs and next_offset: never retry already committed events.
    """
    values=[event.model_dump() for event in events]
    # Validate all labels/contexts and byte bounds before the first write.
    from episodic_memory import canonical, bindings
    for event in values:
        canonical(event['frame']);bindings(event['roles'])
        if not event['source'].strip() or len(event['source'].encode('utf-8'))>60000:
            raise ValueError('Source exceeds 60000 UTF-8 bytes; split the document into sourced events')
    ids=[];batch=[];size=0;write_ms=0;encoding_seconds=0;chunks_count=0
    with memory(namespace) as m:
        batches=[]
        for event in values:
            length=len(event['source'].encode('utf-8'))
            if batch and (len(batch)>=64 or size+length>64000):
                batches.append(batch);batch=[];size=0
            batch.append(event);size+=length
        if batch:batches.append(batch)
        for batch in batches:
            try: result=m.remember_many(batch)
            except Exception as error:
                if not ids:raise
                return dict(status='partial',event_ids=ids,stored=len(ids),next_offset=len(ids),
                            error_type=type(error).__name__,retry='Inspect committed IDs; resume only uncommitted events. Do not resend the full request.')
            ids.extend(result['event_ids']);write_ms+=result['write_ms']
            encoding_seconds+=result['semantic_encoding_seconds'];chunks_count+=result['semantic_chunks']
    return dict(event_ids=ids,stored=len(ids),write_ms=write_ms,
                semantic_encoding_seconds=encoding_seconds,semantic_chunks=chunks_count,
                status='stored_user_statements',meaning_verified_by_database=False)

@mcp.tool(annotations=WRITE)
@observed
def remember_event(frame: str, roles: list[Binding], source: str, negated: bool=False,
                   event_time: str='', place: str='', namespace: str='main') -> dict:
    """Persist an event interpreted from USER evidence, including negated events.

    You, the calling LLM, interpret the language. Normalize equivalent descriptions
    to the same frame and semantic roles: for give, agent=giver, recipient=receiver,
    patient=object, even if the sentence says 'receive'. Preserve explicit negation.
    Do not store questions, generated hypotheses, or predictions as observations.
    source must quote the original user statement. Do not invent missing context.
    """
    with memory(namespace) as m:
        return dict(**m.remember(frame,[r.model_dump() for r in roles],source,negated,event_time,place),
                    status='stored_user_statement',meaning_verified_by_database=False)

@mcp.tool(annotations=READ)
@observed
def recall_events(frame: str, roles: list[Binding], negated: bool=False,
                  event_time: str='', place: str='', limit: int=8, namespace: str='main') -> dict:
    """Find episodes using only known role constraints; omit the role being asked.

    Interpret the question before calling. Giving and receiving the same event
    must use the same canonical frame and functional roles. Positive queries do
    not match negated episodes. Returns source evidence and vector distances.
    No results means unknown, not proof that an event did not happen. Multiple
    conflicting episodes must remain ambiguous; do not silently choose the latest.
    If an exact query is empty, candidate_events contains a bounded broader search
    in the same frame, polarity and active namespace. These are NOT exact matches.
    Inspect their original evidence, then retry with the actual stored labels if
    they express the intended meaning. Never treat unrelated candidates as facts
    answering the question. If the frame is wrong, discover vocabulary and retry.
    """
    with memory(namespace) as m:
        events=m.find(frame,[r.model_dump() for r in roles],negated,event_time,place,limit)
        candidates=m.find(frame,[],negated,limit=limit) if not events and (roles or event_time or place) else []
        return dict(events=events,count=len(events),candidate_events=candidates,
                    candidates_are_exact_matches=False,
                    scope='Retrieved user statements, not independently verified facts')

@mcp.tool(annotations=WRITE)
def observe_outcome(event_id: int, relation: str, value: str, evidence: str, namespace: str='main') -> dict:
    """Train the small tensor readout from an explicitly OBSERVED outcome.

    Use only when the user supplies the outcome, not when you infer or predict it.
    This changes local memory operator weights, not GPT or SQLT weights. Current
    decoder supports outcomes identifying one event participant. No silent
    overwrite: duplicate feedback for an event/relation is rejected.
    """
    with memory(namespace) as m:return m.observe_outcome(event_id,relation,value,evidence)

@mcp.tool(annotations=READ)
def predict_outcome(frame: str, roles: list[Binding], relation: str,
                    negated: bool=False, namespace: str='main') -> dict:
    """Apply an already learned role-axis tensor operator to new participants.

    Returns a hypothesis, never an observed fact. Similarity/margin are not
    calibrated confidence. Unknown when no operator has observed supervision.
    Never feed this prediction back as an observation automatically.
    """
    with memory(namespace) as m:return m.predict(frame,[r.model_dump() for r in roles],relation,negated)

@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False,destructiveHint=True,openWorldHint=False))
@observed
def forget_events(event_ids: list[int], namespace: str='main') -> dict:
    """Forget selected memories ONLY when the user explicitly requests removal.

    Retrieve candidate episodes first, check their source and scope, then supply
    their IDs. Do not delete merely because an episode seems unimportant to you.
    Unknown/cross-namespace IDs reject the whole request. Removes episode data,
    bindings and outcomes, and rebuilds affected learned readouts from remaining
    observations. GPT/pretrained weights and other namespaces are untouched.
    Maximum 32 events and 10000 replayed observations per transaction.
    """
    with memory(namespace) as m:return m.forget(event_ids)

@mcp.tool(annotations=WRITE)
@observed
def deactivate_events(event_ids: list[int], namespace: str='main') -> dict:
    """Reversibly remove selected episodes from ACTIVE memory on user request.

    Retrieve matching episodes first. This is not permanent deletion: original
    episodes/outcomes remain archived and can be restored. They are excluded
    from ordinary recall and affected readouts are rebuilt without them. Tell
    the user that deactivation is reversible. Do not call this as a substitute
    if the user explicitly insists on permanent physical erasure.
    """
    with memory(namespace) as m:return m.deactivate(event_ids)

@mcp.tool(annotations=READ)
def recall_inactive_events(frame: str, roles: list[Binding], negated: bool=False,
                           limit: int=8, namespace: str='main') -> dict:
    """Inspect archived episodes only for explicit inspection/restoration requests.

    Never use these inactive episodes to answer ordinary memory questions. Use
    memory_status archived_vocabulary for frame/role discovery when needed.
    """
    with memory(namespace) as m:
        events=m.find(frame,[r.model_dump() for r in roles],negated,limit=limit,active=False)
        return dict(archived_events=events,count=len(events),status='inactive_not_ordinary_memory')

@mcp.tool(annotations=WRITE)
@observed
def restore_events(event_ids: list[int], namespace: str='main') -> dict:
    """Restore archived episodes only at the user's request; rebuild readouts."""
    with memory(namespace) as m:return m.restore(event_ids)


# APFC uses explicit evidence and bounded numerical operators; no executor.
from apfc import APFC

@mcp.tool(annotations=WRITE)
def apfc_bind_entity(entity_key: str, label: str, scope: str,
                     aliases: Annotated[list[str],Field(max_length=16)],
                     evidence_event_ids: Annotated[list[int],Field(min_length=1,max_length=32)],
                     namespace: str='main') -> dict:
    """Register a stable identity and explicitly evidenced aliases within a scope.

    Interpret meaning externally. Similarity alone never authorizes merging.
    Different companies/offices must use distinct keys and scopes. Cite active
    source event IDs proving the identity/equivalence. Alias collisions reject
    the transaction; an existing identity cannot be silently reassigned.
    """
    with memory(namespace) as m:return APFC(m).bind_entity(entity_key,label,scope,aliases,evidence_event_ids)

@mcp.tool(annotations=WRITE)
def apfc_link_entity(entity_key: str,event_id: int,role: str,namespace: str='main') -> dict:
    """Link an existing event's role to an established identity after checking its source.
    The role must exist; linking cannot silently replace a different identity.
    """
    with memory(namespace) as m:return APFC(m).link_entity(entity_key,event_id,role)

@mcp.tool(annotations=READ)
def apfc_identity(alias: str | None=None,scope: str | None=None,
                  entity_key: str | None=None,limit: int=20,namespace: str='main') -> dict:
    """Resolve exact aliases by scope OR retrieve an identity's linked active events.
    Omitted scope may return multiple identities; never conflate them. Retrieve
    semantic candidates first for unknown wording, then verify before binding.
    """
    if bool(alias)==bool(entity_key):raise ValueError('Provide alias OR entity_key')
    with memory(namespace) as m:
        a=APFC(m)
        return a.entity_context(entity_key,limit) if entity_key else a.resolve(alias,scope,limit)

class APFCStateItem(BaseModel):
    model_config=ConfigDict(extra="forbid")
    text: str=Field(min_length=1,max_length=1000)
    evidence_event_ids: list[int]=Field(min_length=1,max_length=32)

class APFCStatePatch(BaseModel):
    model_config=ConfigDict(extra="forbid")
    goals: list[APFCStateItem] | None=Field(default=None,max_length=16)
    confirmed: list[APFCStateItem] | None=Field(default=None,max_length=16)
    completed: list[APFCStateItem] | None=Field(default=None,max_length=16)
    waiting: list[APFCStateItem] | None=Field(default=None,max_length=16)
    questions: list[APFCStateItem] | None=Field(default=None,max_length=16)
    next_steps: list[APFCStateItem] | None=Field(default=None,max_length=16)
    contradictions: list[APFCStateItem] | None=Field(default=None,max_length=16)

@mcp.tool(annotations=WRITE)
@observed
def apfc_update_case(case_key: str,title: str,patch: APFCStatePatch,
                      expected_revision: int,request_id: str,namespace: str='main') -> dict:
    """Persist an evidence-bound operational state with optimistic concurrency.

    Read apfc_case first (new case revision=0). Replace ONLY the supplied state
    sections; others survive. Empty lists clear sections. Every item needs active
    event evidence. Save observations with remember_event(s) first. Distinguish
    goals, completed actions, waiting replies, questions and proposed next steps.
    Source support and valid temporal meaning must be checked by the language
    client; this tool validates references, not semantic truth. On conflicts read
    the new revision before merging. Reuse request_id only for identical retries.
    No action is executed; no LLM weights are trained.
    """
    with memory(namespace) as m:return APFC(m).update_case(case_key,title,patch.model_dump(exclude_none=True),expected_revision,request_id)

@mcp.tool(annotations=READ)
def apfc_case(case_key: str,namespace: str='main') -> dict:
    """Read current revisioned case state; hide items with missing/archived evidence.
    A needs_review result is incomplete. This is a derived state, not the full
    source archive. Read original events when checking external current status.
    """
    with memory(namespace) as m:return APFC(m).get_case(case_key)

@mcp.tool(annotations=READ)
def apfc_context(event_ids: Annotated[list[int],Field(max_length=32)],
                  limit: int=3,namespace: str='main') -> dict:
    """Find bounded operational states explicitly linked to recalled active events.
    Used automatically by the mandatory memory gate; no new observations created.
    An empty result means no linked APFC state, not absence of episodic memory.
    """
    with memory(namespace) as m:return APFC(m).related_cases(event_ids,limit)

@mcp.tool(annotations=WRITE)
def apfc_register_policy(policy_key: str,features: list[str],actions: list[str],namespace: str='main') -> dict:
    """Declare 1..16 named numeric features and candidate actions for a local predictor.
    Feature values must use a consistent documented normalization to [-1,1];
    include units/normalization in feature names or supporting case evidence.
    Schemas are immutable: new units/features require a new policy key. This
    creates no observations and grants no execution permission.
    """
    with memory(namespace) as m:return APFC(m).register_policy(policy_key,features,actions)

@mcp.tool(annotations=WRITE)
def apfc_observe(policy_key: str,action: str,features: dict[str,float],reward: float,
                  before_event_id: int,outcome_event_id: int,request_id: str,namespace: str='main') -> dict:
    """Learn a small numerical action-outcome predictor from verified observations.

    Use actual pre-action features and observed reward in [0,1] with consistent
    meaning (e.g. success within a specified deadline). Both active source event
    IDs must exist and be distinct. Never train on predictions, generated answers,
    missing objections or unverified self-reports. Idempotent request_id; an
    outcome may supervise a policy only once. Updates local RLS tensors, not the
    LLM. Maximum 2048 retained active observations per action; explicit migration
    needed beyond that bounded prototype budget. Does not execute an action.
    """
    with memory(namespace) as m:return APFC(m).observe(policy_key,action,features,reward,before_event_id,outcome_event_id,request_id)

@mcp.tool(annotations=READ)
@observed
def apfc_predict(policy_key: str,features: dict[str,float],namespace: str='main') -> dict:
    """Estimate outcomes per candidate action with small persisted local operators.
    Returns uncalibrated hypotheses, sample counts and novelty, not probabilities
    or permission to act. Untrained actions stay unknown. Rebuilds in RAM when
    referenced evidence is archived/deleted, excluding it from predictions.
    Compare on held-out real observations before trusting a policy operationally.
    """
    with memory(namespace) as m:return APFC(m).predict(policy_key,features)

@mcp.tool(annotations=WRITE)
@observed
def apfc_start_experience(case_key: str,title: str,experience_key: str,policy_key: str,
                           action: str,features: dict[str,float],before_event_id: int,
                           outcome_definition: str,expected_revision: int,
                           actions: list[str] | None=None,before_quote: str='',namespace: str='main') -> dict:
    """Track a performed intervention and await its observed outcome; no training yet.

    Use naturally during ongoing work, not only when user names APFC. Save actual
    evidence with remember_event(s) first. Read case revision (new case=0), choose
    a stable experience key for this single intervention, and supply ONLY known
    pre-action features normalized to [-1,1]. Unknown values must not be invented.
    The criterion defines the stable meaning of reward (0=failure, 1=success or
    an explicitly defined continuous measure); reuse exactly for this policy.
    Automatically creates a missing policy/case and updates waiting state in one
    transaction. No parser or external action execution. Identical retries are
    idempotent. A new intervention needs a new experience key. Do not record
    drafts, conditional plans or proposed actions as performed interventions.
    An intervention and its already observed result in one message may be tracked
    retrospectively: start first, then complete from a distinct outcome event.
    """
    with memory(namespace) as m:
        return APFC(m).start_experience(case_key,title,experience_key,policy_key,action,features,
            before_event_id,outcome_definition,expected_revision,actions,before_quote)

@mcp.tool(annotations=WRITE)
@observed
def apfc_complete_experience(case_key: str,experience_key: str,outcome_event_id: int,
                              observed_quote: str,reward: float,
                              evidence_kind: Literal['user_report','tool_observation'],
                              expected_revision: int,namespace: str='main') -> dict:
    """Close a tracked intervention and learn from its explicit observed result.

    Save the new user/tool outcome as evidence first; observed_quote must be an
    exact excerpt of its source. Read case revision and pending experience. Reward
    must follow that experience's criterion, never an invented confidence score.
    Case state, completion marker and numerical model update commit atomically.
    Duplicate completions never train twice; contradictory feedback rejects the
    overwrite and must be retained as separate evidence for review. Unknown,
    predicted, hypothetical or absent results must remain pending. Interpretation
    of whether the quote proves the outcome remains the language client's task;
    the server verifies source references and quote, not semantic truth.
    """
    with memory(namespace) as m:
        return APFC(m).complete_experience(case_key,experience_key,outcome_event_id,
            observed_quote,reward,evidence_kind,expected_revision)

@mcp.tool(annotations=READ)
def apfc_experiences(case_key: str,status: Literal['pending','completed'] | None=None,
                      limit: int=8,offset: int=0,namespace: str='main') -> dict:
    """List tracked interventions, pending outcomes and learning criteria by case.
    Paginate until has_more=false for a complete list. Pending experiences are
    included in automatic case recall; do not ask to repeat already performed
    actions when only an outcome is missing. Archived evidence is excluded or
    marked needs_review. No training or action execution occurs on reading.
    """
    with memory(namespace) as m:return APFC(m).experiences(case_key,status,limit,offset)

@mcp.tool(annotations=READ)
def apfc_browse(dataset: Literal['cases','experiences','policies','entities'],
                 filters: list[MemoryFilter] | None=None,limit: int=50,offset: int=0,
                 namespace: str='main',include_sql: bool=False) -> dict:
    """Read-only paginated APFC objects with parameterized exact field filters.

    cases: case_key,title,revision; experiences: case_key,experience_key,policy_key,
    action,status,reward,outcome_definition; policies: policy_key; entities:
    entity_key,label,scope. Numeric comparisons only on revision/reward. Source
    archival excludes experiences and identities. Case rows expose metadata only;
    use apfc_case to read evidence-filtered state. Never execute supplied SQL.
    """
    from apfc_browser import browse
    with memory(namespace) as m:
        return browse(m,dataset,[f.model_dump() for f in filters] if filters else [],limit,offset,include_sql)

@mcp.tool(annotations=READ)
@observed
def memory_event(event_id: int,namespace: str='main') -> dict:
    """Read one active source event by exact ID within the selected namespace.
    For source verification after following an APFC evidence reference. Archived
    or foreign namespace events are never returned by this ordinary reader.
    """
    if type(event_id)is not int or event_id<1:raise ValueError('Invalid event ID')
    import json
    with memory(namespace) as m:
        with m.c.cursor() as q:
            q.execute('SELECT id,frame,roles,source,negated,event_time,place FROM episodic_events WHERE namespace=%s AND id=%s AND active=1',(namespace,event_id))
            row=q.fetchone()
        return dict(event=None if row is None else dict(event_id=row[0],frame=row[1],roles=json.loads(row[2]),source=row[3],negated=bool(row[4]),time=row[5],place=row[6]))

@mcp.tool(annotations=READ)
def apfc_status(namespace: str='main') -> dict:
    """APFC identity/case/policy counts and fixed operator/cache resource limits."""
    with memory(namespace) as m:return APFC(m).status()

@mcp.tool(annotations=READ)
async def memory_context(query: str, namespace: str='main') -> dict:
    """Automatic hybrid memory lookup before generation, for any topic or model.

    Combines literal stored values, semantic candidates and query_memory expansion.
    Read-only and bounded; not exhaustive or a proof of relevance.
    """
    if not isinstance(query,str) or not 1<=len(query)<=8000:
        raise ValueError('Query must contain 1..8000 characters')
    from memory_context import lexical_candidates, assemble_context
    import asyncio
    def lexical():
        with memory(namespace) as m:return lexical_candidates(m,query)
    matches=await asyncio.to_thread(lexical)
    async def call(name,args):
        if name=='query_memory':
            args={**args,'filters':[MemoryFilter(**f) for f in args['filters']]}
        return await asyncio.to_thread(globals()[name],**args)
    return await assemble_context(call,namespace,query,matches)

@mcp.tool(annotations=READ)
@observed
def natural_memory_context(query: str, namespace: str='main') -> dict:
    """Run the existing validated Natural Language Query before generation.

    Uses the configured SQL Tensor NLQ backend/planner. No broad semantic fallback
    on empty or unsupported plans; only matched source records are returned.
    """
    from nlq_context import natural_context
    return natural_context(query,namespace)

if __name__=='__main__':mcp.run(transport='stdio')
