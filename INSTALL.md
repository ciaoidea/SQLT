# Install CORTEX on Linux

Requires Python 3.11+, Docker and internet access for initial dependencies and the
public multilingual encoder. Debian/Ubuntu system dependencies can be installed
by the bootstrap script. Run as the intended application user.

```sh
./install.sh --client none --no-link
```

Defaults: container `cortex-public-mariadb`, database `CORTEX`, localhost port
`13307`, dedicated Docker volume, fresh random credentials. The database is
case-sensitive on Linux. Lowercase `cortex` is deliberately rejected.

The installer prepares `.venv`, schema, native VECTOR support, encoder and a local
MCP definition. It performs a real MCP handshake and status check. It does not
seed examples or copy operational memories. Credentials remain in private local
files and are excluded from the source archive.

Client integrations can be enabled explicitly:

```sh
./install.sh --client codex --no-link
./install.sh --client claude --no-link
./install.sh --client gemini --no-link
```

Existing conflicting `cortex` client registrations are preserved and rejected;
this independent installation does not replace the operational server. The
`--client none --no-link` setup avoids global client and launcher changes.
Built-in Codex launchers route Cortex MCP to this installation with per-process
configuration overrides. Other native clients require their own explicit setup.

To use another port on a fresh installation:

```sh
./install.sh --port 13308 --container cortex-public-secondary --client none --no-link
```

For an existing dedicated MariaDB 11.8+ database named `CORTEX`, provide a private
PyMySQL JSON configuration using `--database existing --config-file PATH`.
Use an account restricted to that database. Never supply operational credentials.
An existing local `.env.json` is reused, with no password replacement or reset.

```sh
./install.sh --check --client none --no-link
.venv/bin/python -m unittest test_installer -q
python3 make_package.py
```

The resulting `dist/cortex-installer.tar.gz` contains only allowlisted source,
tests and public documentation. It excludes local settings, DB state, verification
outputs, caches, dependencies and account authentication. Extracting and installing
it creates an independent installation with new credentials and empty memory.

## Optional visual browser

The installer includes `sqlt.net/` and its Python dependencies. After setup, run
`.venv/bin/python sqlt.net/server.py` from the installed CORTEX directory and open
http://127.0.0.1:8090/. Select a different local port using `--port 9000`.
Natural-language planning requires the same user's installed Codex CLI and login.
The browser reuses the private database configuration; it does not provision a
second database or require an external web server. See `sqlt.net/README.txt`.
