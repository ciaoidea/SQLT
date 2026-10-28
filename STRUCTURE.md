# Application structure tracking

CORTEX is the source; cortex is the operational installation. This tool records
SHA-256 manifests of explicitly allowed application files. It excludes memory,
credentials, local MCP configuration, environments, model downloads and runtime
state. Existing target-only integrations are left in place.

```sh
python3 structure_sync.py record --message "Describe the change"
python3 structure_sync.py init --target ~/cortex
python3 structure_sync.py plan --target ~/cortex
python3 structure_sync.py apply --target ~/cortex
```

Initialize each target once. Initialization preserves its existing differences;
only subsequent recorded source changes are deployed. Source file deletions are
tracked. Independently modified target files block deployment when they conflict.
Deployments are explicit, not automatic. Stop affected services before deployment
and restart them afterwards. Individual files are replaced atomically; a whole
release is not atomic. Backups and deployment logs live under target .structure/;
source manifests and revision history live under source .structure/.

Changes are detected when record runs, not continuously. Do not edit private data
into allowed source files: an allowlist is not a secret scanner.

## Database structure

No database connection, SQL execution or data copying is performed by this tool.
Schema-related application code is tracked, but SQL migrations require separate
review and execution against the target database, preserving its records. Never
copy .env.json from the source: the target must keep its own database identity.
A code update that needs a migration must not be activated until that migration
has been tested. This tool does not update dependencies or running processes.
