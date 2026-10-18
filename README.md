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
