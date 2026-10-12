"""Install SQLT, its MariaDB VECTOR schema, and verified Granite weights."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
from torch_setup import ensure_torch, nvidia_devices, preferred_build, probe_torch

ROOT = Path(__file__).resolve().parent
REPO = 'ibm-granite/granite-3.1-3b-a800m-instruct'
REVISION = 'a02780686e08a03fe0d2679a293b5c74a90efa89'

def command(args, **kwargs):
    return subprocess.run(args, check=True, cwd=ROOT, **kwargs)

def compose_command():
    if shutil.which('docker'):
        probe = subprocess.run(['docker', 'compose', 'version'], capture_output=True)
        if probe.returncode == 0:
            return ['docker', 'compose']
    if shutil.which('docker-compose'):
        return ['docker-compose']
    raise RuntimeError('Docker Compose is missing; use --system-deps on Debian/Ubuntu')

def system_dependencies():
    if not shutil.which('apt-get'):
        raise RuntimeError('--system-deps supports Debian/Ubuntu; install Python venv and Docker Compose manually on other distributions')
    prefix = [] if os.geteuid() == 0 else ['sudo']
    command(prefix + ['apt-get', 'update'])
    command(prefix + ['apt-get', 'install', '-y', 'python3-venv', 'ca-certificates'])
    if not shutil.which('docker'):
        command(prefix + ['apt-get', 'install', '-y', 'docker.io'])
    try:
        compose_command()
    except RuntimeError:
        for package in ('docker-compose-v2', 'docker-compose-plugin', 'docker-compose'):
            check = subprocess.run(['apt-cache', 'show', package], capture_output=True)
            if check.returncode == 0:
                command(prefix + ['apt-get', 'install', '-y', package])
                break
    if shutil.which('systemctl') and subprocess.run(['docker','info'],capture_output=True).returncode:
        command(prefix + ['systemctl', 'enable', '--now', 'docker'])

def read_env():
    from_env = {}
    if (ROOT / '.env').exists():
        for line in (ROOT / '.env').read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                from_env[key.strip()] = value.strip()
    from_env.update({k: v for k, v in os.environ.items() if k.startswith('SQLTENSOR_')})
    return from_env

def configure(existing):
    env = read_env()
    if not (ROOT / '.env').exists() and not existing:
        defaults = dict(SQLTENSOR_DB_HOST='127.0.0.1', SQLTENSOR_DB_PORT='3308',
            SQLTENSOR_DB_NAME='SQLT', SQLTENSOR_DB_USER='sqlt',
            SQLTENSOR_DB_PASSWORD=secrets.token_hex(24),
            SQLTENSOR_ROOT_PASSWORD=secrets.token_hex(24), SQLTENSOR_ADMIN_PORT='8089')
        defaults.update(env)
        fd = os.open(ROOT / '.env', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as handle:
            handle.write('\n'.join(f'{key}={value}' for key,value in defaults.items())+'\n')
        env = read_env()
    if not env.get('SQLTENSOR_DB_PASSWORD'):
        raise RuntimeError('Set SQLTENSOR_DB_PASSWORD in .env before using --existing-db')
    if env['SQLTENSOR_DB_PASSWORD'].startswith('replace_with_'):
        raise RuntimeError('Replace the placeholder passwords in .env, or remove that example copy to generate fresh credentials')
    if not existing and (env.get('SQLTENSOR_DB_HOST', '127.0.0.1') != '127.0.0.1' or not env.get('SQLTENSOR_ROOT_PASSWORD')):
        raise RuntimeError('Managed Docker requires host 127.0.0.1 and SQLTENSOR_ROOT_PASSWORD; use --existing-db for a remote database')
    return env

def environment(full, device='auto'):
    python = ROOT / '.venv/bin/python'
    if not python.exists():
        command([sys.executable, '-m', 'venv', str(ROOT / '.venv')])
    command([str(python), '-m', 'pip', 'install', '--upgrade', 'pip'])
    torch_report = None
    if full:
        torch_report = ensure_torch(python, device)
        command([str(python), '-m', 'pip', 'install', '-r', str(ROOT / 'requirements.txt')])
    else:
        command([str(python), '-m', 'pip', 'install', 'PyMySQL>=1.1,<2', 'numpy>=1.26,<3'])
    return python, torch_report

def stage(python, label, script, args=(), input_text=None, offline=False):
    log = ROOT / (label + '.log')
    print(f'{label}: running (log: {log.name})', flush=True)
    env = os.environ.copy()
    if offline:
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    with log.open('w') as handle:
        command([str(python), str(ROOT / script), *args], stdout=handle,
            stderr=subprocess.STDOUT, input=input_text, text=True, env=env)

def database_stage():
    from database import connect
    # No torch/transformers imports are needed for the schema-only installation.
    from schema import initialize
    for attempt in range(120):
        try:
            connection = connect()
            break
        except Exception:
            if attempt == 119:
                raise RuntimeError('Database is not reachable after 240 seconds; check the container and .env credentials') from None
            time.sleep(2)
    with connection as c, c.cursor() as cur:
        cur.execute("SELECT VERSION() version, VEC_ToText(VEC_FromText('[1,2]')) vector_test")
        result = cur.fetchone()
        if json.loads(result['vector_test']) != [1,2]:
            raise RuntimeError('Native MariaDB VECTOR check failed')
    initialize()
    with connect() as c,c.cursor() as cur:
        cur.execute('SELECT repo,source_commit,status FROM moe_model WHERE id=1')
        model = cur.fetchone()
        if model and (model['repo'] != REPO or model['source_commit'] != REVISION):
            raise RuntimeError('This database contains a different model; refusing to overwrite it')
        cur.execute('SELECT COUNT(*) n FROM moe_parameters')
        if model is None and cur.fetchone()['n']:
            raise RuntimeError('Orphaned parameter rows found; refusing to overwrite this database')
    return result, model

def inspect(device='auto'):
    report = dict(project='SQLT', python=sys.version.split()[0],
        supported_python=(3,10) <= sys.version_info[:2] <= (3,13),
        free_project_disk_gib=round(shutil.disk_usage(ROOT).free/1024**3,1),
        env_exists=(ROOT/'.env').exists(),venv_exists=(ROOT/'.venv/bin/python').exists(),
        expected_database='SQLT', default_port=3308)
    devices = nvidia_devices() if device != 'cpu' else []
    build, reason = preferred_build(device, devices)
    report['pytorch_plan'] = dict(requested=device, candidate_build=build,
        reason=reason, detected_gpus=devices)
    if (ROOT/'.venv/bin/python').exists():
        report['installed_pytorch'] = probe_torch(ROOT/'.venv/bin/python', cuda=device != 'cpu')
    try:
        report['compose_command']=compose_command()
        probe=subprocess.run(['docker','info'],capture_output=True)
        report['docker_accessible']=probe.returncode == 0
    except RuntimeError as exc:
        report['docker_accessible']=False;report['docker_message']=str(exc)
    print(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true',help='Inspect prerequisites without changing files, Docker, or the DB')
    p.add_argument('--system-deps',action='store_true',help='Install Debian/Ubuntu Python venv and Docker packages using apt')
    p.add_argument('--database-only',action='store_true',help='Install MariaDB and all empty tables; do not download weights')
    p.add_argument('--existing-db',action='store_true',help='Use an empty, dedicated database configured in .env; do not manage Docker')
    p.add_argument('--keep-source',action='store_true',help='Keep the downloaded conversion checkpoints after verification')
    p.add_argument('--benchmark',action='store_true',help='Also run the slow optional chat benchmark')
    p.add_argument('--phpmyadmin',action='store_true',help='Start the optional localhost phpMyAdmin service')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto',
        help='Automatically install and validate CPU/CUDA PyTorch; CPU fallback if CUDA is unavailable')
    p.add_argument('--db-worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.db_worker:
        result, model=database_stage()
        print(json.dumps(dict(server=result,model=model)));return
    if a.check:inspect(a.device);return
    if not (3,10)<=sys.version_info[:2]<=(3,13):
        p.error('Python 3.10 through 3.13 is required by this tested dependency configuration')
    if a.existing_db and a.phpmyadmin:p.error('--phpmyadmin requires managed Docker')
    with (ROOT/'.install.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another SQLT installation is running') from None
        if a.system_deps:system_dependencies()
        if not a.database_only and shutil.disk_usage(ROOT).free < 35*1024**3:
            # A verified existing install does not need fresh download space.
            if not (ROOT/'installation-report.json').exists():
                raise RuntimeError('At least 35 GiB free space is required for a fresh model installation; 40+ GiB recommended')
        configure(a.existing_db)
        python, torch_report=environment(not a.database_only, a.device)
        if not a.existing_db:
            compose=compose_command()+['--project-name','sqlt','--env-file',str(ROOT/'.env'),'-f',str(ROOT/'compose.yaml')]
            command(['docker','info'],stdout=subprocess.DEVNULL)
            command(compose+['up','-d','db'])
            if a.phpmyadmin:command(compose+['--profile','tools','up','-d','phpmyadmin'])
        db_result=subprocess.run([str(python),str(ROOT/'install.py'),'--db-worker'],cwd=ROOT,text=True,capture_output=True,check=True)
        status=json.loads(db_result.stdout)
        print('MariaDB native VECTOR and all SQLT tables are ready.',flush=True)
        if a.database_only:
            print('Schema installed. Run ./install.sh to download, import and verify the model.');return
        model=status['model']
        if not model or model['status']!='ready':
            stage(python,'download-progress','download.py')
            if not model or model['status']=='importing':
                stage(python,'import-progress','store.py',('.conversion',))
            elif model['status']!='imported':
                raise RuntimeError('Unsupported model status: '+str(model['status']))
            for phase in ('weights','reference','compare'):
                stage(python,phase+'-verification-progress','verify.py',(phase,'.conversion'),offline=True)
        stage(python,'offline-chat-smoke','runtime.py',('--device',a.device,'--cache-mib','3584','--int8','chat'),input_text='/quit\n',offline=True)
        if a.benchmark:stage(python,'benchmark-progress','benchmark.py',offline=True)
        if not a.keep_source:
            if (ROOT/'.conversion').exists():shutil.rmtree(ROOT/'.conversion')
            (ROOT/'reference-logits.npz').unlink(missing_ok=True)
        report=dict(project='SQLT',repo=REPO,source_commit=REVISION,ready=True,
            database=read_env().get('SQLTENSOR_DB_NAME','SQLT'),
            source_checkpoints_kept=a.keep_source,offline_chat_smoke=True,
            pytorch=torch_report)
        (ROOT/'installation-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('SQLT installation complete. Start the terminal chat with ./chat',flush=True)

if __name__=='__main__':
    try:main()
    except (RuntimeError,subprocess.CalledProcessError) as exc:
        print('Installation stopped: '+str(exc),file=sys.stderr)
        print('Fix the reported issue and rerun ./install.sh. Existing model rows are preserved.',file=sys.stderr)
        raise SystemExit(1)
