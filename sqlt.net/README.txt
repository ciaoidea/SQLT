SQL Tensor visual query browser
===============================

From the CORTEX directory, after installing the main package:

  .venv/bin/python sqlt.net/server.py

Open http://127.0.0.1:8090/ in your browser.
Optional: --port 9000 --namespace demo. Ctrl+C stops the server.
Start under the same OS user that owns your local configuration and Codex login.
No additional web server, public domain, TLS certificate or hosting account is
required. The server binds only to 127.0.0.1 and validates the Host header.
It is a local development server, without public-user authentication.

The package contains source code and synthetic tests, not database credentials,
memories, browser transcripts or deployment configuration. It uses the existing
CORTEX .env.json privately through the local MCP server. Never publish that file.
No database reset, migration of user observations or automatic writes occur
when opening the viewer. It uses the read-only memory and APFC/detail MCP tools described below. The browser never receives database connection credentials.

Interface
---------
Natural Language Query at the top, rotating 3D graph in the center, connectivity
legend and an expandable Vector Query panel below. Type a question and press
Enter. Clarifications appear above the same input field; reply there to retain
the original request. Reset starts a new query. The interface is in English;
queries and stored source text may use other languages.

Drag to rotate, right-drag to pan, scroll to zoom. Hover for a description; click
a node to view original evidence. Hierarchical node area reflects all distinct
descendants in the current graph. More connected nodes are brighter, from red
through yellow and blue to white. Soft projected envelopes follow each cluster.
The layout is illustrative topology, not semantic embedding coordinates.
There are no decorative stars or invented relationships.

Planning and retrieval
----------------------
The optional natural-language planner uses the installed Codex CLI and login,
model gpt-6.1-sol. These planning requests require network access. It receives
frame/role metadata and limited categorical examples, not original source bodies.
Shell and app tools are disabled. Output is validated against plan.schema.json
and the real frame/role catalog. No language keyword parser or arbitrary SQL
execution is used. Without a configured Codex login, the initial read-only graph
remains available, but natural-language planning will report an error.

Exact filters use parameterized SQL and pagination. Semantic queries use native
MariaDB cosine ranking with optional SQL filters applied before ranking. Semantic
top-k is explicitly partial. Similarity is not logical proof or confidence.
The bottom panel shows actually executed SELECT templates and bound parameters;
query-vector coefficients are replaced by their type and SHA-256 in the UI.
The full vector remains in the local API response. Grouping is performed by the
application over retrieved records, not misrepresented as SQL GROUP BY.

Grouping accepts up to four role names. For branch_connectivity only, the
existing structured entity contract company | city | region | address supplies
derived grouping fields. This decodes stored values, without guessing geography.
When company is requested, the visualization places it first to provide one hub
per company. The requested order and actual visual order are reported separately.
Duplicate labels at different addresses retain their distinct branch identity.

Limits
------
The catalog scans at most 2000 active events and refreshes after two minutes.
Exact queries retrieve at most 1000 rows per branch, semantic queries at most 16.
Graph truncation is labeled partial. The renderer caps the visible graph at 380
nodes. Cycles are bounded and original graph links are retained. Initial views
sample stored profiles and contexts; an empty database produces an empty graph.
No searches or returned records are written to access logs or static assets.
Vendor: Cytoscape.js 3.33.1 (MIT); visualization uses Canvas without requiring GPU.

Tests
-----
  .venv/bin/python -m unittest discover -s sqlt.net -p 'test_*.py'


Memory Workbench (read-only first release)
-----------------------------------------
The interface now includes a dataset/collection navigator, paginated data grid,
exact field filters with removable chips, Table/Graph views, and structured APFC
practice details with clickable source-event references. The natural-language
query field remains the main entry point. SQL templates/parameters and the
validated plan remain visible in the bottom panel. The graph keeps its 3D
rotation and recorded-link versus similarity distinction.

New read-only MCP tools: apfc_browse and memory_event. APFC browsing exposes
cases, experiences, policies and registered identities; archived evidence is
excluded. A policy row describes its declared schema and does not, by itself,
prove training occurred. Practice detail shows the actual observed outcomes.

Natural query plans select memory or an APFC dataset. SQL filters are validated
against that dataset's fields. Memory retains cosine retrieval and SQL/hybrid
queries; APFC browsing currently supports one exact relational selection.
Unsupported joins, field sorting, SQL aggregates and formulas require clarification.
This version has no write forms, arbitrary SQL editor or database admin access.

Dataset browsing uses 50-row pages. Natural memory tables show the first 50
retrieved records, with paging for a single exact relational selection. Semantic
and union selections remain explicitly partial. Graph size remains bounded;
3D positions illustrate topology, not embedding coordinates. Planning runs
outside the MCP data lock so table browsing is not held up by LLM generation.
Opening a table, graph or source does not train a model or write observations.

Live dynamics: optional observed MCP memory activity overlay. See ../LIVE_DYNAMICS.md.
Static mode remains the default; collection stops when no Live graph is visible.

Graph is the single graph view. Its in-canvas Live dynamics checkbox streams actual query construction; uncheck it for direct static results. Query completion always preserves the canonical static graph.
