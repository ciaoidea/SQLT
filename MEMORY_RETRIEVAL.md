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
