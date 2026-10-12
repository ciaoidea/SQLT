# SQLT

**SQL-stored neural weights, selectively loaded MoE experts, CPU/CUDA PyTorch inference.**

SQLT maps IBM Granite 3.1 3B A800M Instruct into MariaDB native `VECTOR`
columns and relational metadata. A terminal chat reconstructs the model from
the database. The learned router selects experts; SQL retrieves the selected
matrices by parameter ID, and PyTorch performs the neural arithmetic.

This is a database-backed inference and text fine-tuning experiment.
Conversation does not automatically update weights; neural arithmetic runs in PyTorch. It is not
a RAG application and does not store a serialized model checkpoint as a BLOB.

## Install on Linux

Requirements: Python **3.10–3.13**, Docker Engine and Docker Compose, an x86-64
CPU supporting PyTorch INT8 execution, internet access, **40+ GiB free disk** and
preferably **24 GiB RAM**. Full source verification previously peaked at about
13 GiB process RAM; MariaDB and the OS need additional memory. Chat used about
5.2 GiB process RAM on the original machine, excluding MariaDB.

From the repository directory:

```bash
./install.sh --check
./install.sh
./sqlt
```

For Debian/Ubuntu machines missing system packages:

```bash
./install.sh --system-deps
```

The `--system-deps` option uses `apt-get` and root or `sudo`. It installs Python
venv and missing Docker/Compose packages. If Docker is installed but inaccessible
to your user, configure Docker access before running the installer; the script
does not silently change group membership. Python itself must already exist.
Other Linux distributions need Python venv and Docker Compose installed first.

Installation performs these stages:

1. Generates a private `.env` with separate random database and root passwords.
2. Creates `.venv`, selects CPU/CUDA PyTorch 2.8.0 automatically, validates it,
   and installs the application dependencies.
3. Starts an isolated MariaDB 11.8.9 Compose service on `127.0.0.1:3308`.
4. Tests native vector support and creates all ten model tables in `SQLT`.
5. Downloads the pinned official model files and validates checkpoint SHA-256.
6. Imports each actual matrix row as a native vector and creates expert bindings.
7. Compares every stored coefficient to its source, then checks two lossless
   inference steps and tokenizer behavior against the source architecture.
8. Tests offline chat startup, then removes conversion checkpoints by default.

The installer detects NVIDIA GPU capability and driver version, selects an
official CUDA 12.8 or 12.6 PyTorch build when supported, and runs a small real
GPU computation. An existing working CUDA build is preserved. If CUDA cannot
be installed or validated, it installs and validates CPU PyTorch automatically.
A temporary GPU out-of-memory condition retains CUDA support; chat can use CPU
until GPU resources are available. Other GPU vendors use CPU by default.
NVIDIA drivers must already be installed; the installer does not replace them.
`./install.sh --check` reports detected GPUs, the proposed build and validation
of an existing PyTorch environment without installing packages.

Downloading, importing and full verification can take hours on a small CPU or
slow connection. Each major stage writes a local progress log. For example:

```bash
tail -f download-progress.log
tail -f import-progress.log
```

Rerunning the installer preserves verified data. An interrupted import resumes
at complete, committed parameter blocks; an existing block with different
source coefficients is rejected. A checkpoint checksum failure discards the
invalid partial shard so the next run can download it again. No automatic
destructive database reset is performed.

If you copied `.env.example` to `.env`, replace its placeholder passwords before
installation. Otherwise the installer generates them. Do not change Docker
initialization credentials after the persistent volume has been initialized;
doing so does not change the users already stored in MariaDB.

## Installation options

```bash
# Install the database and empty model tables, without downloading the model:
./install.sh --database-only

# Also start optional phpMyAdmin on http://127.0.0.1:8089:
./install.sh --phpmyadmin

# Keep original conversion files after successful verification:
./install.sh --keep-source

# Also run the optional, slow chat benchmark:
./install.sh --benchmark

# Override automatic PyTorch selection:
./install.sh --device cpu
./install.sh --device cuda
```

phpMyAdmin login uses server `db`, the configured database user and the password
in your private `.env`. The regular chat needs no web application.

To use an existing MariaDB 11.8+ instance, first create a **dedicated empty
database** and an account with privileges on that database. Copy `.env.example`
to `.env`, fill in your connection settings, then run:

```bash
./install.sh --existing-db
```

This option does not start or manage Docker. The installer refuses a database
whose model metadata identifies a different source model or revision.

## Terminal chat

```bash
./sqlt
./sqlt --max-tokens 1024
./sqlt --cache-mib 4096
./sqlt --device cpu
./sqlt --device cuda
```

Automatic mode checks available RAM and GPU memory, then briefly measures a
representative layer on CPU and CUDA, including estimated transfers from RAM.
It can select CPU even when CUDA is available if the local measurement favors
CPU. This estimates token decoding; SQL loading and prompt processing can have
different performance. It prints the selected device at startup; `/status`
also reports resource budgets, calibration and the selection reason.
`--device cuda` skips the speed comparison but still requires sufficient GPU
memory. Unavailable CUDA or GPU out-of-memory cause a CPU fallback; an
interrupted response is regenerated on CPU. Later responses stay on CPU.

CUDA uses lossless BF16 expert storage with FP32 arithmetic; `--int8` applies
only on CPU. The GPU expert cache is limited by `--cache-mib` and free GPU
memory, reserving shared weights and 768 MiB for context and temporary
tensors. This may change output compared with approximate CPU INT8 execution.
GPU execution also retains recently read experts in a separate RAM cache, so
GPU eviction usually requires only a RAM-to-GPU transfer. This cache defaults
to at most 6144 MiB, is reduced to leave room for shared weights and 4096 MiB
of available system memory,
and keeps exact BF16 coefficients or original FP32 values as needed. Configure
it with `./sqlt --host-cache-mib 6144`; `0` disables it. `/status` reports RAM
usage, cache hits and database reads. The first request still needs to read
uncached experts; chat shows elapsed time while waiting for its first token.
The RAM cache lasts until the chat exits and survives GPU-to-CPU fallback.
GPU host buffers use pinned memory when available and asynchronous transfers.
Single-token GPU execution batches the selected expert matrices while retaining
FP32 arithmetic. Two independent SQL readers prefetch selected, uncached
experts; `--db-workers 0` disables prefetch, and `1`, `2` or `4` select the
reader count. Aggregate SQL timing can exceed wall time when reads overlap.
CPU INT8 prepares each real matrix once rather than packing temporary weights
first. Before each response, memory pressure can reduce cache budgets or switch
GPU execution to CPU. CPU caches also respect available RAM at startup.

For a repeatable local measurement:

```bash
.venv/bin/python benchmark_runtime.py --device auto --output performance.json
.venv/bin/python -m unittest test_torch_setup test_installer test_runtime_cache test_runtime_device test_resource_policy test_runtime_chat test_int8_projection
```

CUDA dependencies are installed automatically by `./install.sh`; no separate
PyTorch installation command is needed. Installing a CUDA build enables the
GPU path, while chat still chooses CPU or GPU according to memory and measured
latency.

Commands: `/quit`, `/reset`, `/status`. The default input context limit is 1024
tokens, with up to 512 generated output tokens. Chat history and KV cache live
in RAM and are cleared on exit. No conversation-induced weight learning is
implemented.

Default CPU execution uses approximate INT8 experts in RAM. Database weights remain
unchanged. To use lossless expert coefficients with more RAM and CPU work:

```bash
.venv/bin/python runtime.py --cache-mib 6144 chat
```

## Storage and routing

The pinned model is `ibm-granite/granite-3.1-3b-a800m-instruct`, revision
`a02780686e08a03fe0d2679a293b5c74a90efa89`:

- 3,298,793,472 unique coefficients; approximately 800M active parameters.
- 32 Transformer layers, 40 experts per layer, 8 selected per token per layer.
- 1,280 experts in total; 256 expert activations across all layers for one token.
- `VECTOR(1536)` and `VECTOR(512)` store matrix rows; normalization gains use scalars.
- SQL contains tokenizer vocabulary/merges, model configuration and layer bindings.
- Original BF16 coefficients are represented exactly as FP32 SQL vector values.

The schema is provided in [schema.sql](schema.sql) and created by
[schema.py](schema.py). Tables are:

| Tables | Contents |
| --- | --- |
| `moe_model` | Architecture, source revision, tokenizer settings and readiness |
| `moe_parameters` | Matrix identity, shape, dimensions, checksum and role |
| `moe_vectors_1536`, `moe_vectors_512` | Native vector rows of neural weights |
| `moe_scalars` | Scalar coefficients |
| `moe_layers`, `moe_experts` | Router and expert parameter bindings |
| `moe_tokens`, `moe_merges` | Tokenizer vocabulary and merges |
| `moe_events` | Import and verification events |

The pretrained router performs selection, and SQL addresses weights through
primary keys rather than an ANN index.

Shared coefficients and routers load at startup, about 1,064 MiB as FP32.
Expert matrices load on demand into a per-layer LRU cache. The default 3,584 MiB
expert cache can eventually hold almost all INT8 experts, so warm performance
does not prove that a much larger model can run quickly with a tiny RAM footprint.

## Observed results and limitations

The original MoEv2 deployment ran on an Intel i3-1115G4 CPU with about 23 GiB
RAM. A later measurement found:

| Measurement | Observed value |
| --- | --- |
| Startup including Python imports | 7.50 seconds |
| First short answer with empty expert cache | 72.36 seconds |
| Repeated short answer | 1.95–1.96 seconds for 11 tokens |
| Longer warm responses | About 5–6 tokens/second |
| Peak chat process RAM | About 5.13 GiB, excluding MariaDB |
| SQL table/index allocation | About 16.07 GB / 14.97 GiB |

These are historical observations using a different installed PyTorch build,
not performance guarantees for a new installation. OS and database caches were
not flushed, and another user chat was open. Full coefficient equality and two
lossless logit steps were verified; this is not a general language-quality
benchmark. Default INT8 execution can change outputs. The model produced an
incorrect explanation of Pythagoras in testing. Neither superior accuracy nor a
performance advantage over equivalent file-backed inference has been established.

## Stopping the database

```bash
docker compose --project-name sqlt --env-file .env -f compose.yaml stop
```

The named `sqlt_database` volume persists model data. A later installer run
starts the service again. Deleting that volume deletes the imported model; no
volume deletion command is part of installation.

## Source and model license

The IBM model is published under Apache-2.0. Model weights are downloaded from
the [official pinned Hugging Face repository](https://huggingface.co/ibm-granite/granite-3.1-3b-a800m-instruct/tree/a02780686e08a03fe0d2679a293b5c74a90efa89)
and are not included in Git. The original model card is stored in `moe_model`.
MariaDB native vectors are documented in the
[MariaDB Vector overview](https://mariadb.com/docs/server/reference/sql-structure/vectors/vector-overview).

Private `.env` files, virtual environments, downloaded checkpoints and local
logs/reports are excluded by `.gitignore`.

## Learn from a UTF-8 text file

```bash
./learn /path/to/text.txt --dry-run
./learn /path/to/text.txt --steps 20
./sqlt
```

Next-token cross-entropy, backpropagation, gradient clipping and SGD update
actual base weights. Any UTF-8 language supported by the tokenizer is accepted;
quality depends on the pretrained model and corpus. This trainer supports the
SQLT Granite MoE architecture, not arbitrary model architectures.

By default, attention projections and normalization weights in the last layer,
plus the final normalization, are trained. Experts, embeddings and routers stay
frozen. `--last-layers N` expands the trainable subset and increases memory use.
This is partial fine-tuning, not full-model training. Training uses CPU without
INT8 operators; it can be substantially slower and use more RAM than chat.

Defaults: one training pass, 64-token sequences, 2,048 MiB expert cache, 8 MiB input limit.
By default every training block is used once in file order. Explicit --steps
cycles through the blocks if its count exceeds the number of blocks. The final 10% of tokens are reserved for validation.
Saving is refused for non-finite values, conflicting model revisions, or
validation loss worsening by more than 0.05 (`--max-validation-regression`).
Lower loss does not prove understanding: test independent questions and
unrelated tasks. Small or repetitive corpora can overfit and damage abilities.

Updated coefficients replace native VECTOR/scalar rows atomically, together
with checksums and model revision. Original changed coefficients remain in
native vector/scalar backup tables (`moe_training_*`) with run metadata. No
checkpoint or adapter is saved on disk. Backups consume DB space and are
retained. The command prints `saved_run`; undo that currently applied run with:

```bash
./learn --restore RUN_ID
```

Restore is atomic and creates another revision. Only the currently applied run
can be undone; this is not a general history manager. Restart chats after
training or restore. Existing chats refuse their next response if their loaded
revision differs from the DB. A running response holds a shared revision lock,
so publishing updated weights waits for it. The original source-weight equality
verifier will correctly find differences after fine-tuning.

`--dry-run` loads the model/tokenizer and prints a plan without writing or
training. Installing this feature alone does not modify model weights.


### Teach from a file in terminal chat

```text
/learn /path/to/lesson.txt
```

`/learn` accepts a UTF-8 file path (spaces are supported). Each training block
is used exactly once; the final 10% of tokens is held out for validation, not
trained. A pass therefore requires multiple optimizer steps for a longer file.
The chat releases its model, runs the trainer, then reloads the saved DB weights
and clears the conversation. Bare `/learn` displays usage. Inline explanations
and `/learn N text` have been replaced by the file command.

Standalone `./learn /path/to/file.txt` also defaults to one complete training
pass. Explicit `--steps N` remains available for advanced use.

### Terminal appearance

Interactive terminals display a compact SQLT header, a colored input prompt,
live Markdown responses (including code blocks), a waiting indicator and brief
response timing. `/help` lists commands. The renderer uses Rich; redirected
output, `TERM=dumb`, `NO_COLOR`, or a missing Rich installation retains plain
output. This changes presentation only, not model inference or learning.

### Model-selected mathematical tool

Default chat uses only a short conversational system prompt and streams a
single neural response. It does not supply mathematical instructions or dispatch
tools. The experimental automatic math integration is now opt-in with
`sqlt --math-tools`: production responses showed that its instructions can
misroute ordinary entity questions as calculus requests. It is not considered
reliable. No keyword-based user-language router is used.

In experimental math mode Granite chooses a structured operation and arguments.
Valid calls invoke SymPy and display the actual result without another neural
explanation pass. This remains distinct from the model's own understanding.
The `/math` shortcut and word-based routing have been removed.

The local executor evaluates arithmetic and supported functions (`sqrt`, `sin`,
`cos`, `tan`, `log`, `exp`, `abs`) and solves polynomial equations up to degree 4
with one unknown in the real domain. Roots are checked by substitution.
Expression syntax uses a restricted AST (no eval), input limits and a 5-second
worker timeout. This syntax handling executes expressions chosen by the model;
it does not interpret the user's language. At most one valid calculation is run per turn; one argument-repair attempt is allowed.

No model weights are changed by tool use. Correct tool selection, translation
of a word problem and faithful explanation remain model capabilities, not
accuracy guarantees. `ask` JSON output includes the selected expression and
verified tool result for inspection. Training commands remain separate.

### Input editing and pause

Interactive input supports Up/Down to recall complete messages from this session,
bracketed multiline paste (terminal Paste / Ctrl+Shift+V), Enter to send and
Alt+Enter to insert a newline. Input history stays in RAM; no history file is
written. Prompt Toolkit provides editing, with a plain-input fallback.

While generating, Esc pauses and another Esc resumes. Pause is cooperative:
current tensor or database operations finish first, then the next token/expert
checkpoint waits. It does not cancel the response or release the model from RAM.
The terminal settings are restored on completion, error or Ctrl+C. This pause
control applies to chat generation, not the separate learning subprocess.

### Calculus and differential equations

The same model-selected calculation now accepts an `operation` argument:
`derivative`, `integral`, `limit`, `series`, or `ode` (default `evaluate`).
Ask naturally, for example:

- Find an antiderivative of `x*exp(x^2)`.
- Differentiate `x^3*sin(x)` twice with respect to x.
- Integrate `exp(-x^2)` from minus infinity to infinity.
- Find the limit of `sin(x)/x` as x approaches zero.
- Expand `sin(x)` about zero through the term in x^5.
- Solve `y''(x) + y(x) = 0`.

The model must select the operation and arguments; no user-language keyword
router is involved. Tool arguments include `variable` (x/y/z/t), `order`, a
mixed-partial `variables` list, string bounds `lower`/`upper`, a string `point`
and limit `direction` (+, -, or +-). The executor accepts `oo` for infinity.
ODE expressions use `diff(y(x),x,2)` or the corresponding y(t) notation.
Only general ODE solutions are currently exposed, without initial conditions,
PDEs or coupled systems. Orders are bounded (calculus up to 10, ODE syntax up to 4).

Indefinite integral results mention the arbitrary constant C and are checked
by differentiation when SymPy can confirm equality. General ODE solutions are
checked by substitution. Two-sided finite-point limits compare both sides.
Unevaluated integrals and non-finite results are explicitly identified. SymPy
cannot provide explicit solutions for every problem; worker timeouts (10 seconds
for calculus, 5 for arithmetic) return an error rather than hanging chat.
If a definite integral stays unevaluated, numerical quadrature is attempted at
30 and 45 digits. A stable estimate is labelled approximate; agreement is not a
universal convergence proof. Infinite bounds alone do not imply an infinite result.
Model tool selection and final explanation still need independent evaluation.

Invalid tool argument choices are returned to the model for one correction
attempt; they are never executed as another operation by a language keyword rule.


### Installer feature coverage

A full `./install.sh` installation includes the file trainer, model-selected
mathematical/calculus tools, Rich terminal rendering, Prompt Toolkit input
editing and Esc pause control. Both schema creation paths (`schema.py` and
`schema.sql`) include native-vector training rollback tables. Full installs run
`feature_check.py` offline plus the existing offline chat startup check before
writing a ready installation report with capability metadata.

To update an existing configured installation without reimporting ready weights:

```bash
./install.sh --existing-db
```

`--database-only` creates the core and training-history schema, without installing
or checking the full inference/training environment.


### Launch from any directory

Run `sqlt` to open the terminal chat. The installer registers a command wrapper
in `/usr/local/bin` for root or `~/.local/bin` for a regular user (that directory
must be on PATH). `./sqlt` also works inside the project folder. Runtime options
are forwarded, for example `sqlt --device cpu`. The old `chat` launcher is renamed.

A short output prefix is held before streaming prose; recognized internal
protocol stays hidden. One
model correction attempt is allowed; if it still fails, chat explicitly reports
that no valid calculation was performed. This does not guarantee that the model
always chooses or explains an operation correctly. Restart an already running
chat after runtime updates to load the new code and begin with clean context.


### Faster file learning

`/learn /path/to/file.txt` still trains each training block once by default and
writes the same selected base weights transactionally to MariaDB. Frozen layers
are now processed layer by layer across the corpus. Their exact detached
activations are kept temporarily in RAM, so updates and both validation passes
run only the trainable suffix. This also reduces repeated SQL expert reads;
experts share the existing cache budget during preparation. No INT8 training or
extra model files are introduced. Preparation is reported per frozen layer.

Activation storage has a 512 MiB budget including a conservative allowance for
intermediate copies; split large files if the budget is exceeded. The original
full-forward path remains available with `./learn FILE --no-prefix-cache`.
SQL persistence batches rows into a connection-local staging table and applies
joined updates, retaining revision checks, atomic rollback and restore history.
The initial database load and the first frozen-prefix computation still take
time; speedup depends on corpus size and repeated updates. Training output
includes preparation and training timings.

A read-only comparison on the installed 3B model used two 24-token blocks:
full forward evaluation took 148.15 seconds and prefix preparation plus suffix
evaluation took 70.79 seconds (2.09x), with matching losses. Expert database
loads fell from 2,187 to 1,177. These timings exclude initial model loading,
backpropagation and persistence; they are not an end-to-end `/learn` benchmark.
Independent gradient tests compare losses, gradients and updated parameters
against the original training path. MariaDB tests cover batched vector writes,
rollback, revision conflicts and restore using temporary tables.


### Read-only learning comparisons

To compare the currently applied training run with its original weights:

```bash
.venv/bin/python evaluate_learning.py --run 7 \
  --question "Che cosa è Directa SIM?" \
  --question "In quale città e in quale anno è stata fondata Directa SIM?"
```

The evaluator asks the same questions in fresh contexts, with a neutral system
prompt and no source text. It emits the current answers first, then loads the
checksum-verified original selected parameters from training history into RAM
and emits the previous answers. Database weights and revision are unchanged.
Expert execution uses the same approximate INT8 path as CPU chat. Truncated
answers are explicitly marked `finish_reason: length`. This comparison is a
qualitative diagnostic, not proof of comprehension or a general knowledge score.


### Learning optimizer

File learning now defaults to AdamW at learning rate `1e-5`, with no weight
decay, instead of SGD at `1e-4`. A production one-pass run on Directa changed the
selected parameters by only about `1.1e-6` of their combined L2 norm under SGD;
validation loss improved only slightly. This motivated an adaptive optimizer,
but does not establish that AdamW learns those facts after one pass. The
selected parameters, one-pass schedule, SQL persistence and validation regression
check are retained. AdamW requires two extra moment tensors in RAM (about
48 MiB for 6.3 million FP32 parameters). Optimizer state is temporary and is not
resumed across separate training runs. To reproduce the original optimizer:

```bash
./learn /path/to/file.txt --optimizer sgd --learning-rate 0.0001
```

A successful run means finite updates passed the held-out loss check and were
saved, not that an answer-level knowledge test passed. Compare fresh-context
answers before calling a factual lesson successful.

The read-only comparison of Directa run 7 produced identical before/after
answers on both probes: it confused Directa SIM with a phone SIM and gave the
incorrect founding place/year "Milano, 2006". This demonstrates no answer-level
improvement on these probes under the CPU INT8 inference path. The AdamW
replacement is covered by gradient/update tests, but its factual learning on
this production corpus has not yet been established.
