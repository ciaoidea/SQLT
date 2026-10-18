# CORTEX — persistent memory and tensor operations

A standalone distribution of a model-independent memory engine. Language models
interpret requests and generate answers through MCP; local code persists evidence,
retrieves events and learns small numerical outcome operators.

- MariaDB 11.8+ stores structured events, source evidence and native vectors.
- A frozen multilingual FastEmbed/ONNX encoder supports semantic retrieval.
- Parameterized SQL supports exact filters and complete paginated results.
- NumPy implements role-bound tensors and supervised outcome operators.
- Predictions are hypotheses; they are not automatically stored as observations.

## Installation

```sh
tar -xzf cortex-installer.tar.gz
cd CORTEX
./install.sh --client none --no-link
```

The installer creates a private Python environment, generates fresh database
credentials and provisions an isolated MariaDB container. This distribution
requires the database name **CORTEX**, including capitalization. It refuses the
operational lowercase `cortex` database. No user memories or credentials are
included in the source package.

See [INSTALL.md](INSTALL.md) for setup and [EXPERIMENTS.md](EXPERIMENTS.md) for the
previous controlled experiment and its limits. The source tree contains no
operational transcripts, database dumps, model weights or legacy SQLT adapter.

## Use

`./cortex mcp` starts the MCP server. `./cortex` starts the terminal client and
requires a separately authenticated Codex client. Other compatible MCP clients
can use the server without that terminal interface. The built-in Codex launchers
override the Cortex MCP command for their process to use this installation,
without modifying the global client configuration. Client registration is
optional; do not replace an existing operational server registration when testing
this independent copy.

## Boundaries

The encoder is frozen and does not learn from stored experiences. The learned
outcome readout has eight coefficients and chooses among supplied participants;
it is not a general reasoning engine or a self-organizing memory topology.
Structured representations preserve functional roles, explicit negation, source
and context. Similarity alone is not proof of a fact.
