# SQL Tensor

**A SQL and vector memory framework for AI agents.**

SQL Tensor gives agents persistent memory they can query. Information is stored
as structured events and semantic vectors, so an agent can retrieve relevant
facts without repeatedly reading large Markdown memory files. The aim is to
reduce unnecessary context and token usage; a comparative token benchmark has
not yet been published.

**CORTEX is the current implementation of SQL Tensor.** It provides the memory
engine, numerical learning operators and an MCP server. A terminal client is
included, but other compatible agents can use the same MCP interface.

## What it does

- **Remember:** save new information with its original source, participant roles,
  explicit negation and context. It remains available across conversations.
- **Find exact information:** use SQL filters for names, codes, fields, counts
  and complete paginated lists.
- **Find related information:** use vector search when the question and the
  stored information use different wording.
- **Learn from outcomes:** update small numerical prediction operators when an
  observed outcome is explicitly supplied. Predictions remain separate from facts.

For example, an agent can store “Luca is the contact for project Aurora” and,
from a new conversation, retrieve it when asked “Who should I contact about
Aurora?” The source remains available for checking the answer.

## How it works

```text
AI agent / language model
          |
         MCP
          |
        CORTEX
          +-- SQL queries and semantic retrieval
          +-- Tensor operations and outcome learning
          |
        MariaDB
          +-- Events, sources, roles and context
          +-- Semantic vectors and learned operators
```

The language model interprets requests and writes answers. CORTEX manages the
memory. MariaDB stores it, a local ONNX encoder converts text into semantic
vectors, and NumPy performs tensor operations.

CORTEX is biologically inspired: it combines immediate experience storage,
associative retrieval and learning from observed outcomes. This describes the
organization of its functions, rather than a simulation of biological neurons.

New information becomes retrievable as soon as storage completes. No separate
training cycle is needed. Storing a fact updates the memory; explicitly teaching
an outcome can also update an operator's coefficients. Neither operation
retrains the language model or the semantic encoder.

## Quick start

Requires Linux, Python 3.11+, Docker and internet access for the initial setup.

```sh
git clone https://github.com/ciaoidea/SQLT.git CORTEX
cd CORTEX
./install.sh --client none --no-link
```

The installer creates a Python environment and a dedicated MariaDB 11.8 container
with an empty `CORTEX` database, fresh credentials and a local-only port (`13307`
by default). It prepares the encoder and checks the MCP connection. This command
does not change global client registrations or create a global launcher.

To connect an MCP client, use the command and arguments in the generated
`mcp.config.json`. For the included Codex terminal client:

```sh
./cortex
```

The terminal client requires Codex to be installed and authenticated separately.
It can execute commands with your user's permissions. The memory server itself
can be used independently by other MCP clients.

A [source installer archive](downloads/cortex-installer.tar.gz) and its
[SHA-256 checksum](downloads/cortex-installer.tar.gz.sha256) are also available.
See [INSTALL.md](INSTALL.md) for setup options and client registration.

## Code and evidence

This distribution contains source code, synthetic tests and documentation. It
ships without user memories, database dumps, account credentials or model weights.

[EXPERIMENTS.md](EXPERIMENTS.md) documents the previously recorded controlled
operator experiment, including its baselines and limits. It demonstrates a
specific numerical learning task, not general reasoning or a self-organizing
memory topology.

## Persistent terminal sessions

The terminal agent saves native Codex conversations and automatically resumes its
last thread for the same OS user, namespace, model and working directory. Cortex
keeps only a private session-ID reference in `~/.local/state/cortex/sessions/`.
`/new` starts a fresh conversation without deleting MariaDB memory or old native
logs. Explicit `/verify` memory checks remain ephemeral and isolated. Resume
failures are reported, never silently replaced with a new thread. Existing open
clients need a normal restart; earlier ephemeral conversations are not recoverable.
The local MCP configuration of this installation is preserved.

## Visual query browser

An optional standalone Python web server is included in `sqlt.net/`:

```sh
.venv/bin/python sqlt.net/server.py
```

Open http://127.0.0.1:8090/. Ask a natural-language question, inspect the 3D
network and review the executed SQL below it. No external web server or hosting
account is needed. Natural-language planning uses the installed Codex CLI/login;
local MCP retrieves the stored facts. See [viewer documentation](sqlt.net/README.txt)
for setup, limits and the distinction between requested grouping and visual hubs.

### Concurrent terminal sessions

If a saved Codex thread is still owned by another app-server, Cortex uses native
thread/fork to continue from its stored history instead of failing with an active
writer error. The original terminal and its conversation remain untouched. Future
turns in the two windows are separate branches; an unfinished source turn is not
a completed answer. Recovery occurs before turn/start and never replays a model
turn or memory write. Normal single-window resumes keep the original thread.

## Continuous memory and conversation context

MariaDB memory persists independently of the language conversation. Codex uses
native automatic compaction using the model default, preserving the current
thread instead of opening a new one. Cortex does not force a token threshold;
large single turns and tool outputs can exceed it. Compaction requires a working
model connection and can lose details; exact facts must remain in MCP memory.
Automatic waiting notices are disabled in the terminal. The total turn budget is
1800 seconds by default; a separate 300-second transport inactivity watchdog
closes a silent connection without replaying operations. Protocol events reset
the inactivity watchdog, not the total budget. Long pastes receive at least 600
seconds when an explicitly smaller total timeout is selected.
No failed or cancelled model operation is automatically replayed. Restart the
terminal client and affected MCP services after updating their source files.
This does not promise infinite disk capacity or constant retrieval latency.

## Mandatory terminal memory retrieval

The streaming Cortex shell performs a read-only semantic_recall via its own
installation MCP server before every natural-language model turn. Failure blocks
submission to the model; it never silently falls back to conversation alone.
Up to six candidates with source excerpts are included as untrusted evidence.
This guarantees a retrieval attempt, not relevant matches or exhaustive coverage.
Exact lookups, identity checks and complete lists still require further MCP SQL
queries and pagination. Shell commands do not trigger retrieval. This gate applies
to the included streaming shell, not arbitrary external MCP clients. Restart
existing terminals to activate changes. No memory write is performed by the gate.

### Linked case evidence

Before generation the gate expands semantic seeds through exact stored case,
project, subject or entity bindings. Up to four anchors and two 25-row pages per
anchor are queried read-only, including explicit negative facts. At most 18
events and 24000 characters of event data are injected; excerpts and omitted
records are flagged. These are candidate evidence, not a verified full case state.
The model must distinguish requests, completed checks, sent messages, replies
and pending matters, preserving the scope and source of every number. Current
external status still requires inspecting the source and follow-up messages.
Unlinked or unlabeled records may be missed; absence is never inferred from the
preflight. No new case status is invented or written by this mechanism.


APFC adds evidence-bound identities, revisioned operational state and small local outcome predictors. See [APFC.md](APFC.md) for tools, migration and measured-test scope.


## Live graph dynamics

The Memory Workbench has one Graph view with optional Live dynamics. It shows
real query construction and bounded memory-tool activity only while visible.
New chat requests clear the previous graph. Exploratory candidates remain
provisional; the model can select supporting evidence for the active request
with the read-only `focus_memory` tool. The selected graph remains after flares
fade. This visual selection is not independent verification or access to hidden
model activations. See [Live dynamics](LIVE_DYNAMICS.md) for limits and tests.
