# Cortex memory client

For ordinary memory conversations, use the `cortex` MCP tools. Interpret language
using the current Codex model. Do not start the local Granite model.
The terminal agent may use native Codex shell, file, web and configured connector
tools for work requested by the user. For memory, always use Cortex MCP rather
than SQL or local memory files. Do not import the legacy APFC or Markdown memory.

- Start with memory_status to discover canonical frames and roles when needed.
- Store explicit user information with remember_event, preserving source,
  functional participant roles, explicit negation and stated context.
- For multiple facts, pasted lists or tables, use remember_events in batches up
  to 64. Keep one coherent branch/entity and its contact fields together. Preserve
  shared company/location context; do not call a tool once per line or property.
  Confirm the batch once, without narrating every record. Do not invent fields.
- Treat give/receive descriptions of one transfer consistently: agent is the
  giver, recipient is the receiver, patient is the object.
- For exhaustive lists, counts, numeric comparisons and exact field lookups,
  use query_memory directly. It applies parameterized SQL filters to all active
  facts. Discover unknown role names with memory_status. Follow next_offset until
  has_more is false before claiming a complete list; total counts events, not
  distinct entities. Filters on different roles must belong to the same event.
  For long lists use select_roles and include_source=False for compact results.
  Preserve separate branches sharing a city; do not silently collapse identities.
- For meaning-based questions, call semantic_recall with the original
  natural-language question. This uses a learned multilingual encoder and native
  MariaDB cosine search, without exact frame or participant labels.
  Inspect the original source, participant roles, negation, identity and context.
  Nearness is not logical entailment; irrelevant candidates do not answer a
  question. Use recall_events only for structured refinement when necessary.
  Do not answer only from previous chat text. Unknown evidence stays unknown.
- Recall filters require exact stored labels. Resolve paraphrases using the
  participant catalog. Before declaring memory absent, inspect candidate_events
  from an empty search and retry with relevant stored labels; broaden excessive
  constraints or discover alternative frames. Candidates are not exact matches;
  verify source meaning, and clarify when multiple identities remain plausible.
- For named people or entities, run a literal query_memory contains filter on
  entity using a distinctive name part supplied by the user. If empty, try
  known alias and person roles separately. Semantic top-k results and the
  bounded memory_status catalog cannot establish absence. Verify identity,
  follow pagination, and retrieve the matching person_context/case_context
  records and linked context before answering. Preserve dates and distinguish
  drafts or requests from sent messages or completed actions. Only report no
  memory after applicable structured name searches find no matching evidence.
- When retaining sourced context about a named person, keep the full verified
  identity and relevant aliases with their cases; preserve the original source.
- Keep contradictory or temporally distinct episodes separate. Do not invent
  ownership changes or equate possession with ownership.
- Only explicit observed outcome feedback may call observe_outcome. A generated
  answer or predicted outcome is not new observed evidence.
- predict_outcome returns an unverified hypothesis, not a fact or proof.
- When the user explicitly asks to forget information, retrieve the matching
  episodes first and use deactivate_events for reversible exclusion from active
  memory, clearly disclosing archival and restoration. Use restore_events only
  on a restoration request. Never answer ordinary questions from inactive
  episodes. Physical erasure requires forget_events and may be blocked by client
  approval policy; do not substitute archival if permanent erasure was demanded.
  Never infer permission to
  delete from mere infrequency or from a quotation.
- The DB persists memory, not GPT's model weights. Do not claim GPT was trained.
- prepare_semantic_memory warms the small local non-generative encoder and adds
  derived vectors for existing episodes. It does not create user observations.

Default operational namespace: main. A user-requested demonstration or benchmark
may use demo or an explicit check_ namespace to avoid mixing invented examples
with operational user information. Do not expose private configuration files.
