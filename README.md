# SQLTensor

**SQL-stored neural weights, selectively loaded MoE experts, CPU PyTorch inference.**

SQLTensor maps IBM Granite 3.1 3B A800M Instruct into MariaDB native `VECTOR`
columns and relational metadata. A terminal chat reconstructs the model from
the database. The learned router selects experts; SQL retrieves the selected
matrices by parameter ID, and PyTorch performs the neural arithmetic.

This is a database-backed inference experiment. It does not implement online
learning, automatic weight updates, or neural inference inside SQL. It is not
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
./chat
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
2. Creates `.venv` and installs CPU PyTorch 2.8.0 and the application dependencies.
3. Starts an isolated MariaDB 11.8.9 Compose service on `127.0.0.1:3308`.
4. Tests native vector support and creates all ten model tables in `SQLTensor`.
5. Downloads the pinned official model files and validates checkpoint SHA-256.
6. Imports each actual matrix row as a native vector and creates expert bindings.
7. Compares every stored coefficient to its source, then checks two lossless
   inference steps and tokenizer behavior against the source architecture.
8. Tests offline chat startup, then removes conversion checkpoints by default.

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
./chat
./chat --max-tokens 1024
./chat --cache-mib 4096
```

Commands: `/quit`, `/reset`, `/status`. The default input context limit is 1024
tokens, with up to 512 generated output tokens. Chat history and KV cache live
in RAM and are cleared on exit. No conversation-induced weight learning is
implemented.

Default execution uses approximate INT8 experts in RAM. Database weights remain
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
docker compose --project-name sqltensor --env-file .env -f compose.yaml stop
```

The named `sqltensor_database` volume persists model data. A later installer run
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
