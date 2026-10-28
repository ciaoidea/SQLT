"""Idempotent Linux installer. No operational memory deletion or secret output."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
PACKAGES={'codex':'@openai/codex','claude':'@anthropic-ai/claude-code','gemini':'@google/gemini-cli'}
LABEL='org.cortex.installation'


class InstallError(RuntimeError):pass


def run(command,*,capture=False,env=None,timeout=900):
    result=subprocess.run([str(x) for x in command],cwd=ROOT,env=env,
                          text=True,capture_output=capture,timeout=timeout)
    if result.returncode:
        # Never include captured DB/CLI stderr: it may contain private settings.
        raise InstallError(f'Command failed: {Path(str(command[0])).name} (exit {result.returncode})')
    return result.stdout if capture else ''


def write_json(path,data,mode=0o600):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name+'.')
    try:
        os.fchmod(descriptor,mode)
        with os.fdopen(descriptor,'w') as stream:
            json.dump(data,stream,indent=2,ensure_ascii=False);stream.write('\n')
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


class Installer:
    def __init__(self,args):
        self.args=args
        self.python=ROOT/'.venv/bin/python'
        self.env=dict(os.environ)
        self.env['PATH']=os.pathsep.join([str(ROOT/'.tools/clients/node_modules/.bin'),
                                         str(ROOT/'.tools/node/bin'),self.env.get('PATH','')])
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
            self.env[name]='1'
        self.apt_updated=False
        self.report={'database':None,'clients':[]}

    def apt(self,packages):
        if not shutil.which('apt-get'):raise InstallError('Install required system packages: '+' '.join(packages))
        prefix=[] if os.geteuid()==0 else ['sudo']
        if prefix and not shutil.which('sudo'):raise InstallError('Root/sudo is required for system dependencies')
        if not self.apt_updated:
            run(prefix+['apt-get','update']);self.apt_updated=True
        run(prefix+['apt-get','install','-y']+packages)

    def dependencies(self):
        print('Preparing the Python environment…',flush=True)
        if not self.python.exists():
            result=subprocess.run([sys.executable,'-m','venv',str(ROOT/'.venv')],capture_output=True,text=True)
            if result.returncode:
                self.apt(['python3-venv','ca-certificates'])
                run([sys.executable,'-m','venv',str(ROOT/'.venv')])
        run([self.python,'-m','pip','install','--disable-pip-version-check','-r',ROOT/'requirements.txt'],env=self.env)
        run([self.python,'-m','pip','check'],env=self.env)

    def docker(self):
        if not shutil.which('docker'):
            self.apt(['docker.io'])
        for prefix in (['docker'],['sudo','docker']):
            if prefix[0]=='sudo' and not shutil.which('sudo'):continue
            result=subprocess.run(prefix+['info'],capture_output=True,text=True)
            if result.returncode==0:return prefix
        prefix=[] if os.geteuid()==0 else ['sudo']
        if shutil.which('systemctl'):
            run(prefix+['systemctl','enable','--now','docker'])
            for docker in (['docker'],['sudo','docker']):
                if docker[0]=='sudo' and not shutil.which('sudo'):continue
                if subprocess.run(docker+['info'],capture_output=True).returncode==0:return docker
        raise InstallError('Docker is unavailable. Start the Docker engine and rerun; no existing containers were changed.')

    def database(self):
        target=ROOT/'.env.json'
        if target.exists():
            config=json.loads(target.read_text())
            if config.get('database')!='CORTEX':raise InstallError('Existing configuration points outside CORTEX; refusing changes')
            if self.args.config_file and Path(self.args.config_file).resolve()!=target.resolve():
                raise InstallError('Existing .env.json is preserved; remove this conflicting option')
            target.chmod(0o600)
            self.report['database']='existing'
            print('Reusing the existing Cortex database configuration.',flush=True)
            # A stopped installer-owned container can be resumed safely.
            state=ROOT/'.install/database.json'
            if state.exists() and not self.args.check:
                saved=json.loads(state.read_text());docker=self.docker()
                self.check_container(docker,saved['container'])
                run(docker+['start',saved['container']],capture=True)
                self.wait_database(docker,saved['container'])
            return
        if self.args.check:raise InstallError('Missing .env.json; run installation first')
        if self.args.config_file:
            config=json.loads(Path(self.args.config_file).read_text())
            if config.get('database')!='CORTEX':raise InstallError('External configuration must select database CORTEX')
            write_json(target,config);self.report['database']='existing';return
        if self.args.database=='existing':
            raise InstallError('Use --config-file /path/to/private.json for an existing CORTEX database')
        docker=self.docker();name=self.args.container
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,80}',name):raise InstallError('Invalid container name')
        private=ROOT/'.install';private.mkdir(mode=0o700,exist_ok=True);private.chmod(0o700)
        state_path=private/'database.json';env_path=private/'mariadb.env'
        owner=hashlib.sha256(str(ROOT).encode()).hexdigest()[:20]
        existing=subprocess.run(docker+['inspect',name],capture_output=True,text=True)
        if existing.returncode==0:
            self.check_container(docker,name)
            if not state_path.exists() or not env_path.exists():raise InstallError('Managed DB secrets are missing; refusing to replace credentials')
            state=json.loads(state_path.read_text());config=state['connection']
            run(docker+['start',name],capture=True)
        else:
            if state_path.exists():
                state=json.loads(state_path.read_text())
                if state['container']!=name:raise InstallError('Database setup already belongs to another container')
                config=state['connection']
                if not env_path.exists():raise InstallError('Saved database credentials are missing')
            else:
                with socket.socket() as probe:
                    try:probe.bind(('127.0.0.1',self.args.port))
                    except OSError:raise InstallError('Database port is busy; choose --port before installation') from None
                config={'host':'127.0.0.1','port':self.args.port,'user':'cortex',
                        'password':secrets.token_urlsafe(32),'database':'CORTEX','charset':'utf8mb4'}
                descriptor=os.open(env_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                with os.fdopen(descriptor,'w') as stream:
                    stream.write('MARIADB_DATABASE=CORTEX\nMARIADB_USER=cortex\n')
                    stream.write('MARIADB_PASSWORD='+config['password']+'\n')
                    stream.write('MARIADB_ROOT_PASSWORD='+secrets.token_urlsafe(40)+'\n')
                state={'container':name,'volume':name+'-data','connection':config,'owner':owner}
                write_json(state_path,state)
            with socket.socket() as probe:
                try:probe.bind(('127.0.0.1',config['port']))
                except OSError:raise InstallError('Database port is busy; choose --port before the first installation') from None
            volume=subprocess.run(docker+['volume','inspect',state['volume']],capture_output=True,text=True)
            if volume.returncode==0:
                info=json.loads(volume.stdout)[0]
                if (info.get('Labels') or {}).get(LABEL)!=owner:
                    raise InstallError('Database volume exists but belongs to another installation')
            else:run(docker+['volume','create','--label',LABEL+'='+owner,state['volume']],capture=True)
            run(docker+['run','--detach','--name',name,'--restart','unless-stopped',
                '--label',LABEL+'='+owner,'--env-file',str(env_path),
                '--publish',f"127.0.0.1:{config['port']}:3306",'--volume',state['volume']+':/var/lib/mysql',
                '--memory','768m','--cpus','1','--health-cmd','healthcheck.sh --connect --innodb_initialized',
                '--health-interval','5s','--health-timeout','3s','--health-retries','24',
                self.args.image,'--innodb-buffer-pool-size=128M','--max-connections=32'],capture=True)
        self.wait_database(docker,name)
        write_json(target,config)
        self.report['database']='managed-docker'

    def wait_database(self,docker,name):
        print('Waiting for the dedicated MariaDB Vector database…',flush=True)
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            state_text=run(docker+['inspect','--format','{{.State.Health.Status}}',name],capture=True)
            if state_text.strip()=='healthy':break
            time.sleep(1)
        else:raise InstallError('MariaDB did not become healthy. Credentials and volume were retained for retry.')

    def check_container(self,docker,name):
        info=json.loads(run(docker+['inspect',name],capture=True))[0]
        expected=hashlib.sha256(str(ROOT).encode()).hexdigest()[:20]
        if (info['Config'].get('Labels') or {}).get(LABEL)!=expected:
            raise InstallError('Container name belongs to another installation; refusing to modify it')

    def initialize(self):
        print('Checking MariaDB Vector and Cortex tables…',flush=True)
        code='''
import json,re
from storage import connect
with connect() as c:
 with c.cursor() as q:
  q.execute('SELECT VERSION()');version=q.fetchone()[0]
  match=re.match(r'(\\d+)\\.(\\d+)',version)
  if 'mariadb' not in version.lower() or not match or tuple(map(int,match.groups()))<(11,8):
   raise RuntimeError('Cortex requires MariaDB 11.8 or newer with VECTOR support')
  q.execute('CREATE TEMPORARY TABLE cortex_install_vector_probe(v VECTOR(384) NOT NULL)')
  q.execute('SELECT VEC_DISTANCE_COSINE(VEC_FromText(%s),VEC_FromText(%s))',('[1,0]','[1,0]'))
  assert abs(q.fetchone()[0])<1e-6
  q.execute('DROP TEMPORARY TABLE cortex_install_vector_probe')
print(json.dumps({'version':version,'vector_probe':True}))
'''
        details=json.loads(run([self.python,'-c',code],capture=True,env=self.env))
        if not self.args.check:
            run([self.python,'-c','from episodic_memory import EpisodicMemory;from apfc import ensure_schema;m=EpisodicMemory();ensure_schema(m.c);m.close()'],env=self.env,capture=True)
        else:
            check="from storage import connect\nwith connect() as c:\n with c.cursor() as q:\n  q.execute(\"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name IN ('episodic_events','episodic_bindings','episodic_entities','episodic_operators','episodic_outcomes','semantic_encoders','episodic_semantic_vectors')\");assert q.fetchone()[0]==7,'Incomplete Cortex schema'\n  q.execute(\"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name LIKE 'apfc_%'\");assert q.fetchone()[0]==10,'Incomplete APFC schema'"
            run([self.python,'-c',check],env=self.env,capture=True)
        self.report.update(details)

    def model(self):
        if self.args.skip_model_download:return
        print('Preparing the local multilingual encoder (public model download on first install)…',flush=True)
        run([self.python,'-c','import semantic_memory;s=semantic_memory._local_identity();print("Encoder ready:",s[1]["model"])'],env=self.env)
        self.report['encoder_ready']=True

    def node(self):
        node=shutil.which('node',path=self.env['PATH'])
        if node:
            try:
                if int(run([node,'-p','process.versions.node.split(".")[0]'],capture=True))>=22:return
            except (InstallError,ValueError):pass
        arch={'x86_64':'x64','aarch64':'arm64'}.get(platform.machine())
        if platform.system()!='Linux' or not arch:raise InstallError('Install Node.js 22+ manually for this platform')
        base='https://nodejs.org/dist/latest-v22.x/'
        print('Installing a private Node.js runtime…',flush=True)
        listing=urllib.request.urlopen(base+'SHASUMS256.txt',timeout=30).read().decode()
        match=re.search(r'^([a-f0-9]{64})\s+(node-v22\.\d+\.\d+-linux-'+arch+r'\.tar\.xz)$',listing,re.M)
        if not match:raise InstallError('Official Node.js archive/checksum not found')
        tools=ROOT/'.tools';tools.mkdir(exist_ok=True)
        if (tools/'node').exists():raise InstallError('Private Node runtime exists but is incompatible; refusing to replace it')
        with tempfile.TemporaryDirectory(dir=tools) as temp:
            archive=Path(temp)/'node.tar.xz'
            with urllib.request.urlopen(base+match[2],timeout=60) as source,archive.open('wb') as target:
                shutil.copyfileobj(source,target)
            with archive.open('rb') as source:
                if hashlib.file_digest(source,'sha256').hexdigest()!=match[1]:raise InstallError('Node.js checksum mismatch')
            with tarfile.open(archive) as package:package.extractall(temp,filter='data')
            shutil.move(str(Path(temp)/match[2].removesuffix('.tar.xz')),tools/'node')

    def client_binary(self,name):
        executable=shutil.which(name,path=self.env['PATH'])
        if executable:return executable
        self.node()
        npm=shutil.which('npm',path=self.env['PATH'])
        if not npm:raise InstallError('npm is unavailable')
        print('Installing the '+name+' client locally…',flush=True)
        run([npm,'install','--prefix',ROOT/'.tools/clients','--no-audit','--no-fund',PACKAGES[name]],env=self.env)
        executable=shutil.which(name,path=self.env['PATH'])
        if not executable:raise InstallError('Client installation did not create '+name)
        return executable

    def register(self):
        selected=[x.strip() for x in self.args.client.split(',')]
        if selected==['auto']:
            selected=[name for name in PACKAGES if shutil.which(name,path=self.env['PATH'])] or ['codex']
        if selected==['none']:selected=[]
        if any(name not in PACKAGES for name in selected):raise InstallError('Choose clients from auto, none, codex, claude, gemini')
        definition={'command':str(self.python),'args':[str(ROOT/'mcp_server.py')]}
        write_json(ROOT/'mcp.config.json',{'mcpServers':{'cortex':definition}},0o644)
        for name in dict.fromkeys(selected):
            executable=self.client_binary(name)
            if name=='codex':
                result=subprocess.run([executable,'mcp','get','cortex','--json'],env=self.env,capture_output=True,text=True)
                if result.returncode==0:
                    server=json.loads(result.stdout);transport=server['transport']
                    if transport.get('command')!=definition['command'] or transport.get('args')!=definition['args']:
                        raise InstallError('Codex already has another cortex server; registration was preserved')
                    if server.get('enabled') is False:raise InstallError('Cortex is disabled in Codex; existing choice preserved')
                else:run([executable,'mcp','add','cortex','--',self.python,ROOT/'mcp_server.py'],env=self.env,capture=True)
            elif name=='claude':
                merge_server(Path.home()/'.claude.json',dict(type='stdio',**definition))
            else:merge_server(Path.home()/'.gemini/settings.json',definition)
            self.report['clients'].append(name)
            print('MCP configured for '+name+'. Account sign-in remains in that client.',flush=True)
        for filename in ('CLAUDE.md','GEMINI.md'):
            destination=ROOT/filename
            if not destination.exists():destination.write_text('# Cortex\n\nRead AGENTS.md for memory operations and evidence rules. Use the Cortex MCP\nserver for persistent memory; consult it before answering memory questions.\n')

    def link(self):
        if self.args.no_link:return
        directory=Path.home()/'.local/bin';directory.mkdir(parents=True,exist_ok=True)
        target=directory/'cortex'
        if target.is_symlink() and target.resolve()==ROOT/'cortex':return
        if target.exists() or target.is_symlink():raise InstallError('Existing ~/.local/bin/cortex was preserved; use --no-link')
        target.symlink_to(ROOT/'cortex')

    def smoke(self):
        code='''
import asyncio,json,sys
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
async def main():
 async with stdio_client(StdioServerParameters(command=sys.executable,args=['mcp_server.py'])) as io:
  async with ClientSession(*io) as client:
   await client.initialize()
   tools={t.name for t in (await client.list_tools()).tools}
   assert {'remember_events','semantic_recall','memory_status'}<=tools
   result=await client.call_tool('memory_status',{'namespace':'main'})
   assert not result.isError
   print(json.dumps({'mcp_smoke':True,'tools':len(tools)}))
asyncio.run(main())
'''
        self.report.update(json.loads(run([self.python,'-c',code],capture=True,env=self.env)))

    def execute(self):
        if sys.version_info<(3,11):raise InstallError('Python 3.11 or newer is required')
        if platform.system()!='Linux':raise InstallError('This installer currently supports Linux')
        if self.args.check:
            if not self.python.exists():raise InstallError('Python environment missing; run installation first')
        else:self.dependencies()
        self.database();self.initialize()
        if not self.args.check:self.model();self.register();self.link()
        self.smoke()
        if not self.args.check:write_json(ROOT/'.install/last-success.json',self.report)
        print('Cortex '+('check passed.' if self.args.check else 'installation completed.'),flush=True)
        if not self.args.check:print('Start: ./cortex  |  MCP server: ./cortex mcp',flush=True)


def merge_server(path,definition):
    """Preserve unrelated settings; refuse a conflicting server of the same name."""
    path=Path(path)
    data=json.loads(path.read_text()) if path.exists() else {}
    servers=data.setdefault('mcpServers',{})
    if not isinstance(servers,dict):raise InstallError('Invalid MCP configuration; existing file preserved')
    if 'cortex' in servers and servers['cortex']!=definition:
        raise InstallError('Client already has another cortex server; existing file preserved')
    servers['cortex']=definition
    write_json(path,data)


def arguments():
    parser=argparse.ArgumentParser(description='Install Cortex, MariaDB Vector and MCP client integration')
    parser.add_argument('--database',choices=('auto','docker','existing'),default='auto')
    parser.add_argument('--config-file',help='Private JSON connection settings for an existing database named CORTEX')
    parser.add_argument('--client',default='auto',help='auto, none, or comma-separated codex,claude,gemini')
    parser.add_argument('--container',default='cortex-public-mariadb')
    parser.add_argument('--port',type=int,default=13307)
    parser.add_argument('--image',default='mariadb:11.8')
    parser.add_argument('--skip-model-download',action='store_true')
    parser.add_argument('--no-link',action='store_true')
    parser.add_argument('--check',action='store_true',help='Check installed DB/schema/MCP without provisioning or registration')
    return parser.parse_args()


if __name__=='__main__':
    try:Installer(arguments()).execute()
    except KeyboardInterrupt:print('\nInstallation interrupted; existing memory is preserved.',file=sys.stderr);raise SystemExit(130)
    except (InstallError,OSError,ValueError,subprocess.TimeoutExpired) as error:
        # Avoid traceback or accidental serialization of credential-bearing objects.
        print('Installation failed: '+(str(error) if isinstance(error,InstallError) else type(error).__name__),file=sys.stderr)
        raise SystemExit(1)
