# Automatic memory retrieval

A successful MCP connection does not mean a model will choose a memory tool.
Small models can search the tool catalog for a requested fact, find no dedicated
tool, and incorrectly report that the fact is unavailable.

`memory_context(query, namespace)` is a read-only MCP tool for host-driven
retrieval before generation. It combines:

1. Bounded literal matching against active stored values, using Unicode tokens,
   parameterized SQL, namespace isolation and exact-value anchors. There is no
   natural-language grammar, branch list or topic-specific lookup rule.
2. Semantic retrieval through the existing multilingual encoder.
3. `query_memory` expansion of actual stored identities.

Exact-value anchors take precedence over incidental generic words. Without a
usable literal anchor, semantic candidates remain available. Returned records
include identities, field roles, negation, dates and source excerpts. Context is
bounded; omitted records and incomplete coverage are explicit. Retrieved material
is evidence, never executable instructions or new user observations.

This guarantees the retrieval step when the host enables the gate, not the truth
of arbitrary model answers. Similarity and literal overlap do not prove relevance.
Conflicting identities still require interpretation. Complete lists, aggregates,
and counts still need exact filters and `query_memory` pagination. Source excerpts
are truncated explicitly; inspect original sources for full context.

## Cortex terminal

The existing mandatory Codex gate now obtains `memory_context`, preserves linked
APFC case-state retrieval, and stops before a model request on retrieval failure.
Restart an already-open terminal client to load changed Python code. Persistent
chat sessions and memory data are not reset. No database schema migration or
training is required.

## Unsloth Studio (optional integration)

The files in `integrations/unsloth/` install an opt-in pre-generation hook in an
existing Unsloth Studio backend. They do not install Unsloth or alter model weights.

First register an authenticated Cortex HTTP MCP server in Unsloth. Then run the
installer as the same OS user that runs Unsloth:

```sh
python integrations/unsloth/install.py \
  --backend-root /path/to/site-packages/studio/backend \
  --server-id YOUR_UNSLOTH_MCP_SERVER_ID
```

Restart Unsloth Studio. Enable MCP in the chat and keep that server enabled.
The hook reads the existing server configuration through Unsloth's own storage
helpers; no token is embedded in the integration or passed on command lines.
It injects bounded, explicitly untrusted evidence into the server-side request,
without adding it to saved chat history. A new request replaces the context block.
Disabling MCP, the configured server, or the integration disables this hook.

The hook runs at Unsloth's common chat-completion generation entry point and is
independent of model tool selection. With retrieval opted in, an unavailable or
invalid memory response stops generation with HTTP 503 instead of silently
answering without memory. It does not automatically save messages, run write
operations, alter permissions, or force a factual answer when evidence is missing.

A request exceeding 8000 text characters is refused rather than silently
truncated. Retrieval has a 25-second overall timeout and a context budget; this
can increase time before the first generated token. Other routes that do not
pass through this entry point are outside the integration's tested scope.

Installation checks the exact insertion point, backs up the prior source under
`~/.local/state/cortex-unsloth-backups/`, and refuses unknown source layouts.
An Unsloth upgrade can replace the hook; rerun the installer and verify afterward.
To remove it:

```sh
python integrations/unsloth/install.py \
  --backend-root /path/to/site-packages/studio/backend --uninstall
```

Private deployment configuration lives in
`~/.config/unsloth-cortex-memory.json`; do not publish it, MCP tokens, Unsloth's
SQLite database, chat histories, or actual retrieved records.

## LAN transport

`mcp_lan.py` optionally exposes the same server using Streamable HTTP. Its
owner-only JSON configuration is selected with `CORTEX_MCP_LAN_CONFIG` or defaults
to `~/.config/cortex/mcp-lan.json`:

```json
{
  "host": "LAN_ADDRESS_OF_THIS_MACHINE",
  "port": 8093,
  "allowed_networks": ["YOUR_TRUSTED_LAN_CIDR"],
  "allowed_origins": [],
  "token_file": "/path/to/private/token-file"
}
```

Use a randomly generated token of at least 32 characters and a matching
`Authorization: Bearer ...` request header. Run the listener under the installation
owner; binding it to the LAN IP does not expose MariaDB. All authenticated clients
have the existing MCP tool privileges, including mutation tools. Source networks
are enforced without trusting forwarded headers; Host/Origin protections remain
enabled. A browser frontend should connect through its backend.

Plain HTTP does not encrypt tokens or memory traffic. Use it only on a trusted
LAN; use HTTPS or a VPN for untrusted networks. Network changes may require
updating the explicit bind address and network allowlist.

## Validation

`test_memory_context.py` covers forced pre-generation retrieval, fail-closed
behavior, explicit truncation, negation, context replacement, exact expansion,
namespace propagation, and installer compatibility/idempotency/removal. Existing
memory-gate tests cover APFC case-state retention and prevention of generation
when retrieval fails. Live read-only tests used the remote Unsloth MCP client and
a running small local model; private questions and results are not distributed.

### Live Graph with an isolated HTTP service

The viewer uses an owner-only UNIX datagram directory under
`/tmp/cortex-live-<UID>`. A systemd MCP service with `PrivateTmp=true` cannot see
that directory by default: memory queries succeed but Graph receives no events.
Share only that directory using `BindPaths=/tmp/cortex-live-<UID>` and create it
at boot with a systemd-tmpfiles rule owned by the same installation user, mode
0700. Keep the rest of the service's private temporary filesystem isolated.
The viewer and MCP service must use the same installation and OS UID.

This restores observed tool bursts while Graph/Live is visible. It does not by
itself provide a remote application's turn boundary or model-selected final
`focus_memory` event. Those are separate from the transport and are necessary
for a trustworthy stable graph of the evidence actually used in an answer.

## Contextual NLQ strategy

`natural_memory_context` reuses the existing SQL Tensor Natural Language Query
planner and executor. It applies a schema-validated plan and returns only its
matches. Empty/unsupported plans never fall back to general contact profiles or
broad semantic neighbors. The planner receives the server date and the sampled
coverage of each field: a field in the catalog is not necessarily populated in
every record. Questions can concern identity, attributes, relations, quantities,
locations or dates; this is not a date-specific filter.

This optional strategy uses the existing planner provider (currently Codex as
configured in `sqlt.net/planner.py`), independently of the model generating the
answer. It adds a planning round trip and is not a fully local or free retrieval
path. It does not normalize existing identities or make model-generated plans
infallible. Explicit SQL filters constrain matches; a semantic ranking phrase
alone is not an enforced condition. Completeness is scoped to the chosen plan,
not all external sources or all possible representations of an entity.

Configure a running viewer backend through `CORTEX_NLQ_SOCKET` or an owner-only
`.nlq.json` in the installation, containing `{"socket":"/path/to/http.sock"}`.
The backend returns an installation fingerprint and namespace; the adapter
refuses responses from another Cortex installation or namespace. No schema or
user-data migration is performed. Cortex's terminal gate chooses this strategy
when this installation-specific configuration exists, otherwise the original
hybrid strategy is used.

For the optional Unsloth adapter, set `retrieval_tool` to
`natural_memory_context` in its private configuration and restart Studio. Its
maximum retrieval wait is 105 seconds (including the existing planner timeout),
instead of 25 seconds for the fast hybrid strategy. No generation follows a
retrieval failure. The query budget is 8000 characters; evidence remains bounded.
No current service-health conclusion can be drawn from an empty plan result.

For the portable loopback HTTP viewer, `.nlq.json` may instead contain
`{"url":"http://127.0.0.1:PORT"}`. Only a loopback HTTP destination or an explicit
local UNIX socket is accepted. The source/namespace fingerprint must still match.

For read-only NLQ answer generation, the optional Unsloth `read_answer_only: true`
setting suppresses further tool discovery after a validated relational, semantic
or hybrid read plan has executed. The model receives those results for language
output instead of re-searching tool names for the requested data. This affects
that request only; it is not a database permission boundary. Additional tools are
not available during this final read-answer phase, so partial coverage must be
reported rather than completed silently. This mode is opt-in; the normal Cortex
terminal stays on its existing retrieval strategy unless explicitly configured.

The optional installer accepts `--retrieval-tool natural_memory_context` and
`--read-answer-only`; use `--allow-follow-up-tools` to disable the latter. Existing
private choices are preserved when reinstalling without these flags. Changes
require restarting Unsloth, not merely refreshing its browser page. Verify the
HTTP chat route as well as standalone tool/model calls before declaring deployment
active. Planning adds latency; quality and speed remain model/provider dependent.

## Frontier-client fast path

A frontier client can perform structured refinements with its own model during
the answer turn. It does not need a second language-model planner before every
turn. With no `.nlq.json`, the mandatory gate retains local hybrid retrieval.
The persistent Codex instructions explicitly prefer direct `query_memory`
refinements over starting the optional separate NLQ planner.

An owner-only `.recall-http.json` in the installation may point the mandatory
gate at its already-running authenticated MCP server:

```json
{"url":"http://YOUR_SERVER:8093/mcp","token_file":"/path/to/private/token"}
```

Both files must be owner-only. Before using evidence, the client verifies the
installation fingerprint and namespace. A mismatch or unavailable backend stops
generation; it never silently switches databases or answers without memory.
Synchronous native clients use HTTP directly instead of launching Python and MCP
server processes for every turn. Existing asynchronous callers retain a bounded
subprocess route when necessary. Without this explicit configuration, stdio is
still supported. Private HTTP/NLQ configurations are excluded from Git/packages.
Restart already-open terminal clients to load the direct-call optimization;
persistent conversation references and database contents are preserved.

## Exact NLQ plan cache

The viewer reuses schema-validated plans for identical questions in a bounded
RAM cache (128 entries, 300-second expiry). The key includes the complete catalog
fingerprint and server calendar date. Clarifications and plans containing clock
times are not cached; failed execution evicts its plan. Similar questions,
changed entities, negations, catalog changes and day rollovers do not reuse an
entry merely because of semantic proximity.

Only plans are cached: every request executes fresh database queries, including
cache hits. Responses expose `plan_cache` and per-stage `retrieval_timing`.
A cold, unseen NLQ question still needs its language-model planner. This cache
improves repeated requests; it is not a claim of instant planning for new input.
Tests verify fresh reads on hits, exact matching, expiry, bounds, catalog/date
invalidation, immutable returned plans and failure eviction.

Performance measurements distinguish memory retrieval from complete answers.
Removing a preliminary planner and subprocess startups reduces host-side recall;
model generation, tool round trips and optional graph focus still add latency.
A complete cold answer must not be compared to a retrieval-only measurement.

## Identity discovery before choosing a frame

The NLQ backend now calls read-only `literal_memory_candidates` before planning.
It discovers literal word matches across actual stored values, supplying their
frames, roles and bounded source excerpts as per-question planning evidence.
Person information and confirmed alias corrections may live in case notes or
conversation concepts, rather than a contact profile or a populated alias field.
The planner must use the observed storage shape instead of assuming a profile.

This is generic lexical discovery, not a grammar parser, fuzzy name expansion,
automatic alias creation or a database migration. An unknown short name must not
be expanded into a longer similar-looking branch. Clarification is appropriate
when identity evidence is missing. A model may still overgeneralize an alternative
spelling in the question; an inferred alias is not automatically saved or verified.

Per-question evidence is separate from the cached shared catalog and participates
in the plan-cache fingerprint. Newly changed or removed identity evidence changes
the key; plans are not reused against different discovery evidence. SQL data is
still read fresh. Normal frontier-client hybrid retrieval retains its prior
strict-anchor behavior; this discovery step applies to the optional NLQ path.

In Unsloth's opt-in read-answer-only mode, a clarification decision also suppresses
further tool-catalog discovery: the model should ask the requested clarification,
not search for a tool whose name resembles the unknown entity. This is a read
workflow; automatic write/learning behavior is not validated by these tests.

### Remote Unsloth live graph
The natural-memory adapter explicitly starts a graph turn before retrieval and retains the actual injected context after retrieval. This uses existing bounded, viewer-leased telemetry: no extra model call, database write or stored event log. The stable graph is labelled **Prepared memory context**, not model-selected supporting evidence. It does not attest which records the generator used or validate their relevance. Each new host turn clears the preceding graph, including empty/clarification results. Cortex native model-selected focus remains distinct.

### Dedicated Unsloth memory mode and legacy UI cleanup
`install.py --always-retrieve` explicitly opts a dedicated client into mandatory retrieval even when the browser sends `mcp_enabled=false` / `tool_choice=none`. It does not enable model tools or writes. `--respect-mcp-toggle` restores the default per-request MCP gate. Retrieval failure still blocks generation. The installer backs up and removes only the known obsolete `cortex-read-context-v3.js` / `v4.js` script tags, which rewrote questions and restricted roles in the browser. Other scripts are preserved. Reload existing browser tabs after upgrading; already-executed JavaScript remains active until reload.

### Structural identity discovery
NLQ discovery restricts literal anchors to typed identity bindings (`entity`, `person`, `alias`, `contact`, `branch`, `company`, and `entità`), across frames. Descriptive attributes cannot displace the subject merely because they contain a rare or exact query word. Compact identity labels receive a length-normalized discovery score. This is a generic schema-level retrieval rule, not a list of natural-language commands, people or keywords. Bounded matches are still candidates, not proof of identity or absence. Native strict-anchor hybrid retrieval retains its existing behavior. Source roles, dates and identity scope remain authoritative; generated chat history is not ground truth.

In the dedicated `read_answer_only` phase, the generation payload contains only the latest original user request and the freshly retrieved evidence. Earlier generated answers, tool transcripts and older questions are not repassed as facts or new requests. This prevents an earlier generated error from being recycled as memory evidence; the saved conversation is unchanged. This phase is read-only verbalization, not general tool-agent execution. It does not guarantee that an arbitrary generator will faithfully use the remaining evidence.
