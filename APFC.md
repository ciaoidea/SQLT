# APFC — Artificial Prefrontal Cortex

APFC is an evidence-bound operational coordinator, not a biological cortex,
consciousness claim, new general LLM, or autonomous action executor. Language
interpretation stays with the external client. No keyword parser is introduced.
Existing episodic/semantic/relational retrieval remains available.

## Three cooperating components

1. Stable identities: `apfc_entities`, scoped `apfc_aliases`, and
   `apfc_entity_events`. The language client verifies equivalence from active
   original events before registering an alias or linking an event role.
   Similarity proposes candidates; it never silently merges identities.
2. Operational state: `apfc_cases`, `apfc_case_events`, and
   `apfc_case_revisions`. State sections are goals, confirmed, completed, waiting,
   questions, next_steps and contradictions. Each item cites active event IDs.
   A patch replaces only named sections. Expected revisions prevent lost writes;
   request IDs make identical retries idempotent. State has a 24 KB UTF-8 budget.
3. Local learning: `apfc_policies`, `apfc_transitions`, `apfc_models`.
   Each policy defines up to 16 normalized numerical features and 16 actions.
   Per-action recursive least-squares operators predict observed reward in [0,1].
   Feature normalization, units and reward meaning must be fixed by the task.
   Source before/outcome events must be active. The low-level observer requires
   distinct events. The experience coordinator also supports a single compound
   source when before/outcome quotes are exact, separate and non-overlapping. An outcome is used
   only once per policy. No generated answer or prediction is valid feedback.

Numerical operator: x=[1, features, zero padding], predicted reward=x^T w;
P initially=10I; k=Px/(1+x^T Px); w+=k(y-x^T w); P-=k x^T P.
This is a small linear predictor, not a learned universal rule system.
Each action stores 2,448 bytes of float64 tensor coefficients plus metadata.
A bounded LRU caches at most 32 reconstructed operators (about 77 KiB of raw
arrays). Persistent matrices have checksums. Model reads validate evidence and
integrity, so a cache cannot silently continue using withdrawn observations.

## MCP workflow

- Save source observations with existing `remember_event(s)`.
- `apfc_bind_entity` registers evidenced aliases in a company/project scope.
- `apfc_link_entity` connects an existing event role to that identity.
- `apfc_identity` resolves aliases or fetches explicitly linked evidence.
- `apfc_case` reads state/revision; `apfc_update_case` applies evidenced patches.
- `apfc_context` retrieves states linked to recalled event IDs.
- `apfc_register_policy` fixes a feature/action schema.
- `apfc_predict` returns hypotheses, counts and novelty; it executes nothing.
- `apfc_observe` updates parameters from actual externally verified feedback.
- `apfc_status` reports resource limits and counts.

The Codex persistent client's mandatory memory gate automatically calls
`apfc_context` after existing semantic/relational retrieval, within its total
24,000-character context budget. State discovery is linked and bounded, not an
exhaustive scan. Other MCP clients can use the same tools; automatic tool use
still depends on their client orchestration. Existing facts are not automatically
reclassified or assigned identities without verification. Empty APFC tables are
expected before the first real case is explicitly established.

## Persistence and withdrawal

SQL namespace locks serialize writes, including state/model updates. Updating
state uses optimistic revisions as well. APFC updates and dependent deletes are
transactional. No DDL runs per tool request.
Archived evidence hides dependent state items and aliases; outcome models are
invalidated and predictions are rebuilt from remaining active observations.
Permanent event deletion also scrubs directly cited text items from current and
historical case state and removes affected saved operators. It cannot identify
uncited semantic dependencies, third-party chat histories or backups. Case titles
and identity labels are metadata: removing an episode is not deletion of an
entire identity or case. A namespace reset clears APFC as well as episodic data.

## Installation and verification

The installer now initializes both schemas. Quick status: `./cortex apfc`;
repeatable isolated numerical demonstration: `./cortex apfc benchmark`.
Existing installations:

```sh
.venv/bin/python apfc.py migrate
.venv/bin/python apfc.py status
.venv/bin/python -m unittest test_apfc test_memory_gate
.venv/bin/python verify_apfc.py --output outputs/apfc-benchmark.json
```

Migration only adds tables. Existing memory/configuration is preserved. Restart
or reopen MCP clients to discover the new tools; do not kill an active writer.
No database data or credentials should be copied between cortex and CORTEX.

The benchmark uses clearly isolated check_ namespaces and removes its fixtures.
It tests an artificial linear numerical task, unseen feature inputs, reload
consistency and timings. Passing it does not validate general reasoning,
natural-language understanding, causal inference or operational reliability.
Real-case benchmarks are still required. Low latency of this local component
also does not remove network/LLM/encoder latency from the whole chat.

Prototype bounds: at most 2,048 active observations per action for exact replay
and evidence checking. Exceeding the bound fails explicitly; nothing is silently
truncated. A larger deployment needs incremental sufficient statistics with
an explicit withdrawal/replay strategy. There is no claim of infinite memory.


## Conversation-to-learning coordinator

`apfc_experiences` is the persisted bridge from case state to action-outcome
learning. `apfc_start_experience` records an actually performed intervention,
known normalized pre-action features, a fixed reward criterion, and a pending
outcome. It creates a missing policy/case transactionally and trains nothing.

On explicit user/tool feedback, `apfc_complete_experience` verifies the exact
source excerpt, records the observation, updates the numerical operator, removes
the pending item and writes the derived case state in ONE SQL transaction. If
any part fails, the experience, case and learner all roll back. Duplicate
completion is idempotent; contradictory feedback is rejected for review.
Before/after numerical estimates are exposed for inspection, not calibrated
probabilities or proof of improved general skill.

The same source event can contain both an intervention and its result. Supply
`before_quote` at recording and a distinct `observed_quote` at completion. The
coordinator validates both against the original and rejects overlapping excerpts.
This avoids fabricating duplicate source events just to obtain two IDs. Source
interpretation and the truth of reported outcomes still depend on the external
client and evidence quality; substring checks do not prove semantic truth.

`apfc_case` and the mandatory memory gate include a compact experience summary
with pending interventions, criteria, initial features and learned outcomes.
`apfc_experiences` provides status filtering and pagination for full retrieval.
Case text is a bounded working set; old pending experiences remain in the table
even if their working-state item falls outside the latest 16-item window.
`./cortex apfc` now distinguishes pending/completed experiences and counts actual
active learning observations, rather than equating saved practices with training.

The shell prompt, memory client and MCP instructions direct the language model
to use this lifecycle naturally. This is LLM-directed structured tool use, not
a keyword parser or a guarantee that every model/client always follows it.
Unknown measurements/outcomes must stay unknown. Existing records are not
arbitrarily transformed into training examples. A retrospective experience must
still have explicit source evidence and a well-defined criterion.

Tests:

```sh
.venv/bin/python -m unittest test_apfc_experience test_apfc_experience_mcp
.venv/bin/python verify_apfc_chat.py
```

The last command makes two real external-language-model calls in separate
sessions, using an isolated synthetic narrative and real MCP/SQL. It verifies
that a natural-language intervention becomes pending without training, then an
explicit observed outcome produces exactly one local learning update. It never
controls a router and removes its test namespace. No subscription-free local
language model or general learning capability is implied.
