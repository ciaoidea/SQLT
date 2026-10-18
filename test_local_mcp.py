"""The clean distribution must route its client to its own MCP server."""
import json
from pathlib import Path
import unittest
from clients.local_mcp import local_mcp_args


class LocalMCPTests(unittest.TestCase):
    def test_routes_to_own_paths_without_operational_config(self):
        root = Path('/tmp/clean installation')
        args = local_mcp_args(root)
        settings = dict(item.split('=', 1) for item in args[1::2])
        self.assertEqual(json.loads(settings['mcp_servers.cortex.command']), str(root / '.venv/bin/python'))
        self.assertEqual(json.loads(settings['mcp_servers.cortex.args']), [str(root / 'mcp_server.py')])
        self.assertEqual(settings['mcp_servers.cortex.enabled'], 'true')


if __name__ == '__main__':
    unittest.main()
