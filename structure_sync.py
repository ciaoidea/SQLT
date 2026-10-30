#!/usr/bin/env python3
"""Track public application files and explicitly deploy changes without copying data."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import fcntl

ROOT = Path(__file__).resolve().parent
STATE = ROOT / '.structure'

def digest(path):
    if path.is_symlink():
        raise ValueError('Symlink refused: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

def inventory():
    from make_package import FILES
    names = set(FILES) | {'structure_sync.py', 'STRUCTURE.md'}
    names.update(p.name for p in ROOT.glob('test_*.py'))
    names.update(str(p.relative_to(ROOT)) for p in (ROOT / 'clients').glob('*.py'))
    names.update('sqlt.net/' + n for n in ('live.js','map.html','map.css','map.js','server.py','planner.py','plan.schema.json','test_viewer.py','test_plan_cache.py','README.txt','vendor/cytoscape.min.js','vendor/LICENSE-cytoscape'))
    return {n: digest(ROOT / n) for n in sorted(names) if (ROOT / n).is_file()}

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)

def run(args):
    head = STATE / 'head.json'
    current = inventory()
    if args.command == 'record':
        old = json.loads(head.read_text())['files'] if head.exists() else {}
        changes = {n: {'before':old.get(n), 'after':current.get(n)} for n in sorted(old.keys() | current.keys()) if old.get(n) != current.get(n)}
        if not changes:
            print('No application changes.'); return
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        entry = {'revision':stamp,'message':args.message,'files':current,'changes':changes}
        save(STATE / 'revisions' / (stamp + '.json'), entry)
        save(head, entry)
        print(json.dumps({'revision':stamp,'changed_files':list(changes)}, indent=2)); return
    if not head.exists():
        raise ValueError('Run record first')
    revision = json.loads(head.read_text())
    if revision['files'] != current:
        raise ValueError('Unrecorded changes: run record first')
    target = Path(args.target).resolve()
    if target == ROOT or not (target / 'cortex').is_file():
        raise ValueError('Target must be a different Cortex installation')
    local = target / '.structure'
    baseline = local / 'deployment.json'
    if args.command == 'init':
        if baseline.exists(): raise ValueError('Target already initialized')
        save(baseline, {'source_files':current,'target_files':{n:digest(target/n) for n in current},'revision':revision['revision']})
        print('Baseline recorded. No application files or database changed.'); return
    if not baseline.exists(): raise ValueError('Run init for this target first')
    previous = json.loads(baseline.read_text())
    changed = [n for n in sorted(current.keys() | previous['source_files'].keys()) if current.get(n) != previous['source_files'].get(n)]
    conflicts = [n for n in changed if digest(target/n) not in (previous['target_files'].get(n), current.get(n))]
    report = {'revision':revision['revision'],'changed_files':changed,'local_conflicts':conflicts,'database':'Not touched. Schema changes require reviewed, explicit migrations.'}
    print(json.dumps(report, indent=2))
    if args.command == 'plan': return
    if conflicts: raise ValueError('Local changes conflict; nothing applied')
    if not changed: return
    backup = local / 'backups' / revision['revision']
    for n in changed:
        destination = target / n
        digest(destination)
        # Refuse symlinked parent directories too.
        if any(p.is_symlink() for p in destination.parents if p != target.parent):
            raise ValueError('Symlinked target directory')
    for n in changed:
        destination = target / n
        if destination.is_file():
            copy = backup / n; copy.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(destination,copy)
        if n not in current:
            destination.unlink(missing_ok=True); continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent,delete=False) as temporary:
            temporary.write((ROOT/n).read_bytes()); temporary_path = Path(temporary.name)
        shutil.copymode(ROOT/n,temporary_path)
        os.replace(temporary_path,destination)
    save(baseline, {'source_files':current,'target_files':{n:digest(target/n) for n in current},'revision':revision['revision']})
    save(local/'deployments'/(revision['revision']+'.json'),report)
    print('Application updated. Restart affected services. Backups: '+str(backup))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['record','init','plan','apply'])
    parser.add_argument('--target',default=str(Path.home()/'cortex'))
    parser.add_argument('--message',default='Application structure update')
    args=parser.parse_args()
    STATE.mkdir(exist_ok=True)
    with (STATE/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run(args)

if __name__=='__main__':
    main()
