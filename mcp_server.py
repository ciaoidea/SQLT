"""Local MCP access to Cortex memory; never loads an LLM or accepts raw SQL."""
import os
import sys
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
from contextlib import contextmanager
from typing import Annotated
from typing import Literal
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from episodic_memory import EpisodicMemory
import semantic_memory
import relational_memory

mcp=FastMCP('cortex',instructions='Cortex is persistent semantic vector memory. Prepare with prepare_semantic_memory on connection, Use query_memory for complete lists, counts, numeric filters and exact field lookups; semantic_recall for meaning-based questions. A local non-generative multilingual encoder supplies semantic vectors; you evaluate source evidence and generate language. Preserve agent/patient/recipient roles and negation. Quote original user evidence, distinguish observations from predictions. Do not fabricate observations.')

class Binding(BaseModel):
    role: str=Field(min_length=1,max_length=160,description='Stable semantic role, such as agent, patient or recipient; not a grammatical word position')
    value: str=Field(min_length=1,max_length=160,description='Participant or object label. Preserve identity across paraphrases.')

class MemoryEvent(BaseModel):
    frame: str=Field(min_length=1,max_length=160)
    roles: list[Binding]=Field(min_length=1,max_length=12)
    source: str=Field(min_length=1,max_length=8192,description='Original user evidence for this complete event or entity, not generated prose')
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
def semantic_recall(query: str, limit: int=8, negated: bool | None=None,
                    namespace: str='main') -> dict:
    """Search active experiences by MEANING from the original natural-language query.

    Use for meaning-based questions, not exhaustive SQL lists/counts (query_memory). No exact frame, role or entity
    spelling is required. A local learned multilingual encoder maps query and
    original source text to 384D vectors; MariaDB ranks cosine distance.
    Results are candidates: inspect source evidence, functional roles, negation,
    time and identity. Similarity is not logical entailment or calibrated truth.
    By default both positive and negative experiences are returned and explicitly
    marked. Never assert a fact based on mere topic similarity. Use recall_events
    for exact role/context refinement when needed. Archived episodes are excluded.
    """
    with memory(namespace) as m:return semantic_memory.search(m,query,limit,negated)

@mcp.tool(annotations=READ)
def query_memory(filters: Annotated[list[MemoryFilter],Field(max_length=16)],
                 frame: str | None=None,negated: bool | None=False,
                 limit: int=100,offset: int=0,count_only: bool=False,
                 select_roles: list[str] | None=None,include_source: bool=True,namespace: str='main') -> dict:
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
    No SQL text is accepted.
    """
    with memory(namespace) as m:
        return relational_memory.query(m,[f.model_dump() for f in filters],frame,negated,limit,offset,count_only,
                                       select_roles,include_source)

@mcp.tool(annotations=READ)
def memory_status(namespace: str='main') -> dict:
    """Get counts, known frames/roles and a bounded catalog of participant labels.

    Interpret paraphrases using these stored labels, not invented exact filters.
    The catalog may be truncated; its absence is not evidence of missing memory.
    No model loading.
    """
    with memory(namespace) as m:return dict(**m.status(),vocabulary=m.vocabulary(),archived_vocabulary=m.vocabulary(active=False),language_backend='external MCP client',local_llm_loaded=('torch' in sys.modules or 'sql_language_channel' in sys.modules),rss_mib=int(next(line for line in open('/proc/self/status') if line.startswith('VmRSS:')).split()[1])/1024)

@mcp.tool(annotations=WRITE)
def remember_events(events: Annotated[list[MemoryEvent],Field(min_length=1,max_length=64)],
                    namespace: str='main') -> dict:
    """Store multiple user-supplied events in ONE call and atomic transaction.

    Use for pasted lists, tables or several facts. Interpret the entire message
    first. Group all properties of one entity/branch (company, city, address,
    phone, email) in ONE coherent event with role bindings, not separate calls
    for each line or field. Distinguish branches by company AND location; keep
    shared header context in the bindings and quote original evidence in source.
    Do not invent missing fields. Up to 64 events and 64000 source characters;
    use further batches only if necessary. Return a concise overall confirmation,
    not a narration for every row. Entire batch validates before any write.
    """
    with memory(namespace) as m:
        return dict(**m.remember_many([event.model_dump() for event in events]),
                    status='stored_user_statements',meaning_verified_by_database=False)

@mcp.tool(annotations=WRITE)
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
def restore_events(event_ids: list[int], namespace: str='main') -> dict:
    """Restore archived episodes only at the user's request; rebuild readouts."""
    with memory(namespace) as m:return m.restore(event_ids)

if __name__=='__main__':mcp.run(transport='stdio')
