"""Per-process MCP routing to this installation; no global config writes."""
import json
from pathlib import Path
import os
import sys


def local_mcp_args(root=None):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    return ['-c', 'mcp_servers.cortex.command=' + json.dumps(str(root / '.venv/bin/python')),
            '-c', 'mcp_servers.cortex.args=' + json.dumps([str(root / 'mcp_server.py')]),
            '-c', 'mcp_servers.cortex.enabled=true']


if __name__ == '__main__':
    os.execvp('codex', ['codex', *local_mcp_args(), *sys.argv[1:]])
