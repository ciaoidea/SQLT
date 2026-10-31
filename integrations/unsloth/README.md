# Cortex retrieval for Unsloth Studio

The recommended mode makes one authenticated MCP context request before Qwen
answers. It reuses Cortex's existing hybrid lookup: literal stored-value anchors,
multilingual vector search, and bounded expansion of stored identities.
No extra language-model planner, phrase parser, domain-specific name rules,
or model-directed retrieval loop is used in this mode.

Install the backend hook with the existing Cortex MCP registration:

    python install.py --backend-root /path/to/studio/backend --server-id SERVER_ID --retrieval-tool memory_context --read-answer-only

Restart Unsloth Studio after installing. The hook uses the server registration's
existing authentication; credentials are never included in this package.
Local opt-in deployments may additionally use --always-retrieve; the default
respects the chat's MCP switch. The installer retires the obsolete Cortex
browser fetch interceptor without changing unrelated frontend scripts.

The current question triggers retrieval before generation. The model receives
compact facts with identity, dates, negation, source excerpts and coverage flags.
Previous generated answers are excluded as evidence. Tool discovery and further
model tool calls are disabled for this answer phase; the database and retrieval
server still perform the actual MCP lookup. Retrieval failure blocks generation
rather than silently answering without memory.

The fixed policy precedes variable evidence so the backend can reuse its prompt
prefix. Injected evidence is replaced each turn and is not persisted in the chat
history. Graph notifications occur only when a live viewer is watching.

This is bounded retrieval, not exhaustive analytics or model training. Missing
candidates do not prove absence, stored incidents do not establish current live
status, and Qwen can still misinterpret supplied facts. This mode does not offer
autonomous tool exploration or guaranteed conversational reference resolution.
The separate --allow-follow-up-tools mode remains experimental and is not the
recommended configuration for small local models.

Offline regression checks:

    python test_gate.py
    python -m unittest test_memory_context test_memory_gate
