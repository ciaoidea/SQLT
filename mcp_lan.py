"""Authenticated Streamable HTTP access to the existing Cortex MCP tools."""
import ipaddress
import json
import os
from pathlib import Path
import secrets

from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import PlainTextResponse
import uvicorn


class LanAccess:
    def __init__(self, app, token, networks):
        if len(token) < 32:
            raise ValueError('MCP token must contain at least 32 characters')
        self.app, self.token = app, token
        self.networks = [ipaddress.ip_network(n) for n in networks]

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        try:
            peer = ipaddress.ip_address(scope['client'][0])
            allowed = any(peer in n for n in self.networks)
        except (ValueError, TypeError, KeyError):
            allowed = False
        if not allowed:
            return await PlainTextResponse('Forbidden', status_code=403)(scope, receive, send)
        headers = scope.get('headers', [])
        auth = [v for k, v in headers if k.lower() == b'authorization']
        expected = ('Bearer ' + self.token).encode()
        if len(auth) != 1 or not secrets.compare_digest(auth[0], expected):
            return await PlainTextResponse('Unauthorized', status_code=401,
                headers={'WWW-Authenticate': 'Bearer'})(scope, receive, send)
        return await self.app(scope, receive, send)


def main():
    config_path = Path(os.environ.get('CORTEX_MCP_LAN_CONFIG',
        str(Path.home()/'.config/cortex/mcp-lan.json')))
    config = json.loads(config_path.read_text())
    token = Path(config['token_file']).read_text().strip()
    from mcp_server import mcp
    host, port = config['host'], int(config.get('port', 8093))
    # Explicit host/origin allowlists retain SDK DNS-rebinding protection.
    mcp.settings.transport_security = TransportSecuritySettings(
        allowed_hosts=[f'{host}:{port}'],
        allowed_origins=config.get('allowed_origins', []))
    mcp.settings.stateless_http = True
    app = LanAccess(mcp.streamable_http_app(), token, config['allowed_networks'])
    uvicorn.run(app, host=host, port=port, proxy_headers=False,
        access_log=False, log_level='warning', limit_concurrency=32)


if __name__ == '__main__':
    main()
