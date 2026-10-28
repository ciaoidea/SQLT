# Cortex

Read AGENTS.md for memory operations and evidence rules. Use the Cortex MCP
server for persistent memory; consult it before answering memory questions.


## APFC operational coordination

Use APFC tools for multi-step practices when objectives, completed actions or
waiting replies must survive the chat. Read `apfc_case` before patching; cite
original active event IDs in every state item. Keep completed work separate
from proposed next steps; update only changed sections with expected_revision
and a unique request_id. The mandatory Codex memory gate retrieves linked case
states automatically, but does not infer or write updates by itself.

Resolve established scoped identities with `apfc_identity`; register aliases
only from evidence, never from cosine similarity alone. Link event roles after
checking identity. Existing memories are not silently merged.

`apfc_predict` produces uncalibrated hypotheses for declared numeric features and
actions. `apfc_observe` requires verified before/outcome evidence and consistent
reward semantics. Never treat a prediction, generated answer or lack of objection
as feedback. Predictions grant no execution permission. See APFC.md for resource
bounds, mathematical model and limitations.


### Experience-to-learning lifecycle

For an explicitly performed intervention in a tracked practice, use
`apfc_start_experience` after saving the source observations. It creates the
policy if missing, records known pre-action features, and marks the outcome
pending in the case. Do not require the user to know tool names or numeric
commands. Derive boolean features only from explicitly stated/observed facts;
never fill missing conditions or invent numerical measurements. Reuse an existing
policy's exact feature normalization, action labels and outcome criterion.

When the user/tool supplies the actual result, retrieve the pending experience
with `apfc_case`/`apfc_experiences`, store the exact outcome source, and call
`apfc_complete_experience` with a verbatim source quote and criterion-consistent
reward. This atomically updates state and the local model. Missing, conditional,
predicted or hypothetical outcomes remain pending. A retrospective message that
explicitly includes both intervention and observed result can start then complete
the same experience; use distinct before/outcome events, or anchored non-overlapping before_quote
and observed_quote excerpts when one stored source explicitly contains both.
Never duplicate facts merely to manufacture evidence IDs. Repeated
confirmation of the same episode is not a new training example. New interventions
need new experience keys. If required information is missing, preserve it as an
open question rather than fabricate data or silently claim learning.
