"""Installer scope, credential handling and non-destructive client configuration."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import installer


def options(**changes):
    values=dict(database='auto',config_file=None,client='none',container='cortex-mariadb',
                port=13307,image='mariadb:11.8',skip_model_download=True,no_link=True,check=False)
    values.update(changes)
    return argparse.Namespace(**values)


class InstallerTests(unittest.TestCase):
    def test_atomic_private_json_has_restricted_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json'
            installer.write_json(path,{'password':'dummy-test-secret'})
            self.assertEqual(path.stat().st_mode&0o777,0o600)
            self.assertEqual(json.loads(path.read_text())['password'],'dummy-test-secret')
    def test_merge_preserves_other_clients_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'settings.json'
            installer.write_json(path,{'theme':'dark','mcpServers':{'other':{'command':'other'}}})
            server={'command':'/test/python','args':['/test/server.py']}
            installer.merge_server(path,server);installer.merge_server(path,server)
            result=json.loads(path.read_text())
            self.assertEqual(result['theme'],'dark')
            self.assertEqual(result['mcpServers']['other'],{'command':'other'})
            self.assertEqual(result['mcpServers']['cortex'],server)
    def test_conflicting_server_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'settings.json'
            installer.write_json(path,{'mcpServers':{'cortex':{'command':'different'}}})
            before=path.read_bytes()
            with self.assertRaises(installer.InstallError):installer.merge_server(path,{'command':'new'})
            self.assertEqual(path.read_bytes(),before)
    def test_existing_db_config_never_invokes_docker_or_changes_credentials(self):
        with tempfile.TemporaryDirectory() as directory,patch('installer.ROOT',Path(directory)):
            path=Path(directory)/'.env.json'
            installer.write_json(path,{'database':'CORTEX','password':'dummy-private-value'})
            before=path.read_bytes();setup=installer.Installer(options())
            with patch.object(setup,'docker') as docker:
                setup.database();docker.assert_not_called()
            self.assertEqual(path.read_bytes(),before)
    def test_database_other_than_cortex_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory,patch('installer.ROOT',Path(directory)):
            path=Path(directory)/'.env.json';installer.write_json(path,{'database':'SQLT'})
            with self.assertRaises(installer.InstallError):installer.Installer(options()).database()
    def test_operational_lowercase_database_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory,patch('installer.ROOT',Path(directory)):
            installer.write_json(Path(directory)/'.env.json',{'database':'cortex'})
            with self.assertRaises(installer.InstallError):installer.Installer(options()).database()

    def test_external_config_copy_preserves_password_without_printing_it(self):
        with tempfile.TemporaryDirectory() as directory,patch('installer.ROOT',Path(directory)):
            source=Path(directory)/'external.json'
            config={'database':'CORTEX','password':'dummy-secret','host':'127.0.0.1'}
            installer.write_json(source,config)
            installer.Installer(options(config_file=str(source))).database()
            self.assertEqual(json.loads((Path(directory)/'.env.json').read_text()),config)
    def test_check_only_does_not_install_or_register(self):
        setup=installer.Installer(options(check=True))
        with patch.object(setup,'dependencies') as deps,patch.object(setup,'database'),\
             patch.object(setup,'initialize'),patch.object(setup,'smoke'),\
             patch.object(setup,'register') as register,patch.object(setup,'model') as model:
            setup.execute()
            deps.assert_not_called();register.assert_not_called();model.assert_not_called()
    def test_check_missing_config_does_not_provision_database(self):
        with tempfile.TemporaryDirectory() as directory,patch('installer.ROOT',Path(directory)):
            setup=installer.Installer(options(check=True))
            with patch.object(setup,'docker') as docker,self.assertRaises(installer.InstallError):setup.database()
            docker.assert_not_called()
    def test_client_none_generates_only_portable_definition(self):
        with tempfile.TemporaryDirectory() as directory,patch('installer.ROOT',Path(directory)):
            setup=installer.Installer(options(client='none'))
            with patch.object(setup,'client_binary') as install_client:
                setup.register();install_client.assert_not_called()
            config=json.loads((Path(directory)/'mcp.config.json').read_text())
            self.assertEqual(config['mcpServers']['cortex']['args'],[str(Path(directory)/'mcp_server.py')])
            self.assertNotIn('password',json.dumps(config))
    def test_link_conflict_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory,patch('installer.Path.home',return_value=Path(directory)):
            path=Path(directory)/'.local/bin/cortex';path.parent.mkdir(parents=True);path.write_text('existing')
            with self.assertRaises(installer.InstallError):installer.Installer(options(no_link=False)).link()
            self.assertEqual(path.read_text(),'existing')


if __name__=='__main__':unittest.main()
