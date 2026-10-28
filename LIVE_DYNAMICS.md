# Live dynamics

Live dynamics overlays observed memory-tool activity on the existing relation
graph. Static/table/natural-language/SQL behavior is unchanged. It is not a
visualization of private LLM reasoning, neural activations, or inferred causality.
A retrieval is a candidate/result, not proof that an LLM used it in its answer.

## Use

Open the Memory Workbench, select the single **Graph** button. Live dynamics is enabled inside that view; uncheck its in-canvas option for static rendering.
Make a normal request in a Cortex chat connected to this installation's MCP
server. Blue pulses show retrieved records, amber writes, red tool errors.
Existing event nodes are highlighted by exact event ID; out-of-view records
appear as temporary labeled halos. The overlay does not modify graph records.
The panel separates viewer queries from other MCP-client processes.

Table mode is the default. Graph mode enables Live by default. Table view, a hidden tab, or switching Live off stops
collection. Closing the page sends a best-effort stop; abrupt loss expires the
viewer lease in at most 4.5 seconds. Returning to a visible Live graph resumes.
Other visible Live viewers keep their own subscriptions. Up to 32 viewers share
one collector per installation and memory namespace.

Restart already-running Cortex/MCP clients once after installing this release;
Python processes do not reload instrumentation automatically. No restart is
needed between normal chats afterward. The viewer cannot observe unrelated
models or external tools that never call the instrumented Cortex MCP server.
Chat session IDs are not exposed: the panel identifies MCP producer processes.

## Implementation and resource bounds

- live_dynamics.py wraps selected real MCP memory operations. No tools/schema or
  results are changed. Idle calls only consult a 250-ms cached socket-presence
  check; there is no telemetry socket, serializer or receiver thread when idle.
- A visible viewer polls authenticated same-origin `/api/live` once per second.
  A four-second lease activates a private UNIX datagram receiver, shared across
  MCP processes running as the same local OS user and installation/namespace.
- Producers send nonblocking datagrams. Congestion/errors drop visual events;
  tools never wait for the viewer and no DB transaction depends on telemetry.
- 256 events maximum in RAM, 8192 bytes maximum per packet, 100 accepted events
  per second, at most 12 returned record IDs per event, 128 events per poll.
  Only operation/stage, process/origin, call ID, duration, count, record IDs,
  frame labels, negation and up to four selected graph-role labels per returned
  record are transported. No prompts, source bodies, SQL, cookies or vectors
  are copied. Credential-specific roles are excluded by an explicit allowlist. Counts may exceed displayed IDs.
- Events are cleared when the last viewer leaves. No event files, MariaDB
  writes, extra LLM calls, embedding calls or database queries are introduced.
- Rendering caps 48 recent events and 80 record highlights, fading over 6.5 s.
  Data arrives in batches at ~1 Hz; animation is smooth but this is not a
  real-time robot controller. Stop/idle cannot block the chat.
- UNIX socket names include the resolved installation path and namespace;
  Cortex and CORTEX are isolated. Directory 0700, socket 0600. Existing web Host
  and Access checks cover the live endpoint too. CORTEX stays loopback-only.

## Validation

Synthetic tests cover idle behavior, bounded capture, source/query exclusion,
error preservation, namespace/root isolation, multi-viewer leases, disconnect,
and stale socket handling. Existing viewer tests remain required.

Real local browser test: external MCP query produces two observed events;
Table and hidden-tab transitions stop the socket; closing expires/stops it;
no JavaScript errors. Only read-only memory queries are needed for this check.

On the development machine, a synthetic microbenchmark measured observer p50
~0.6 microseconds idle and ~0.15 milliseconds active before further metadata
reduction. These figures exclude tool execution and browser rendering and are
not an end-to-end latency or hardware energy claim.

## Query construction flows

The same Graph now supports request-local NDJSON streaming for live queries.
The optional checkbox is inside the graph; there is no second Graph/Live button.
The query executes once. Instrumented stages are planning, retrieval, assignment
of stored grouping values, duplicate identity consolidation, recorded role
bindings and group/object joins. ContextVar scopes isolate concurrent requests.
Group convergence is a structural aggregation, not evidence of a new concept.
Every animated join comes from a real builder edge whose endpoints already
exist. Node flares use the exact graph IDs. The final result replaces intermediate
patches with the canonical static result; static queries use the same builders.

A 32-packet queue (maximum 64 KiB per progress packet) is nonblocking. Metadata
notices are limited to leave room for graph mutations; the newest mutation
patches take precedence on overflow. The final result is always delivered
separately, even when visual updates are skipped. Disconnection stops progress
capture without cancelling an authorized read or changing its result. Query
streams are scoped to the requesting authenticated HTTP response, not broadcast.
No intermediate sources or full object records are duplicated into progress.

There is no artificial delay to make computation look slower: stages that finish
within one browser frame may appear together. Smooth flares visualize recorded
transitions after arrival; they do not claim the computation is still running.
Natural-language planning may still take seconds. No additional model calls or
DB queries are introduced by streaming. The graph retains its static topology
when flares fade. Existing external MCP activity observation remains available.

Entering Graph from Table starts with an empty graph and no selected table. Table rows remain cached for returning to Table. A new query builds its own result. Dynamic joins have bounded tracer tails; canonical static edges remain visible after the transient flares fade.

## Stable graph from external chat memory access

When Graph starts empty, returned records from external MCP clients now become
stable event nodes and entity/role links, using actual record IDs and canonical
entity hashes. Transient flares fade, while those nodes and links remain until
the Graph workspace is cleared/reopened. Records accumulate within the view;
there is no invented conversation-turn boundary. The title explicitly identifies
retrieved memory, not the generated answer or a claim that every candidate was
used. All MCP clients for this installation/namespace remain in scope.

When an explicit viewer query owns the graph, external chat activity does not
modify its canonical result. The graph remains exactly the static query result.
Return to Table then Graph for a fresh chat-observation workspace.

The live snapshot includes at most 12 records, four allowlisted participant roles
per record, and negation. Results are sent in two-record bounded datagrams.
Ingest batches visual graph updates once per poll, up to 96 records, with the
existing 380-node cap. No source text is transported or background DB lookup
introduced. Clicking Read source explicitly performs the existing read-only
memory detail request. View limits and partial role snapshots are not exhaustive
query answers. Restart existing MCP/Cortex processes once for the richer record
metadata; old clients can only contribute their previously available metadata.

Read-only existing-record test: one returned record, three stable nodes and two links
still present after eight seconds; no JavaScript errors. Reset opens an empty
graph again. The test does not claim to reproduce or verify the LLM answer.

The Cortex shell and memory-chat client emit an explicit chat_turn start event before retrieval. A visible Live graph clears once on that boundary, then accumulates the new question's MCP results. Individual tool calls do not reset it. Boundaries and results arriving in one poll are processed in order; only events after the newest boundary build the new view. No prompt text, extra MCP call or database write is needed. Third-party MCP-only clients need their own user-turn integration; timing is not guessed. Existing running Cortex clients must be restarted once to load this hook.


Automatic-gate exploration and prepared context are now separate stages. Gate
MCP processes identify themselves with origin=gate. Their broad search results
are provisional gray pulses, not permanently added to the graph. After context
assembly, notify_memory_selection reports the actual selected event IDs and
bounded metadata. The stable chat graph keeps this prepared selection. Green
means included in prepared context, not proven relevant or used in the answer.
Later direct model tool calls remain identifiable as retrieved evidence, not
selection judgments. This change does not alter retrieval, scores, expansion,
context budgets, or what the LLM receives. The viewer still observes all local
MCP clients in its namespace; it is not a per-chat authenticated session filter.

Read-only reproduction: a person lookup initially retrieved related semantic
seeds, while a broader linked expansion also explored unrelated contact records.
Those records were absent from the prepared context. Browser regression retained
only the selected facts, with no unrelated contact consolidated and no JavaScript
errors. Relevance is not inferred from name matching in the viewer. Existing
shell processes invoke a fresh memory-gate subprocess each turn, which loads
this stage distinction automatically.


## Request-scoped semantic focus

Automatic context assembly is not relevance verification. Prepared-context IDs
no longer automatically become the stable final chat graph. Exploration stays
provisional. Cortex supplies a live_graph_request_id only when a Live viewer is
active and instructs the existing model turn to call focus_memory before its
final answer with source-backed event IDs supporting the current user request.
The tool adds one local MCP step, not a separate classifier/model service or a
standalone LLM request. It reads/writes no database data. Selection remains a
fallible model judgment, not a formal proof of relevance or truth.

The viewer accepts focus only for its active request ID and builds nodes from
already observed records in a bounded 192-record cache. Unselected provisional
activity contracts/fades; selected recorded bindings remain stable. Unknown IDs
never create invented facts. No focus means no claimed relevant final graph.
An explicit empty selection leaves no supporting-memory nodes. Viewer queries
retain their own canonical result and are not filtered by this chat mechanism.
Third-party clients must provide turn IDs and the focus step to offer the same
behavior. Restart pre-existing Cortex/MCP clients once to load the new tool.
