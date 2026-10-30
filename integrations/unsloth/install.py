"""Install/remove the opt-in Cortex pre-generation hook in an existing Unsloth backend."""
import argparse
import datetime
import json
from pathlib import Path
import shutil

BEGIN = '    # BEGIN CORTEX AUTOMATIC MEMORY\n'
END = '    # END CORTEX AUTOMATIC MEMORY\n'
ANCHOR = '    request = _DisconnectPolicyRequest(\n'
HOOK = BEGIN + '''    from core.inference.cortex_memory_gate import enrich as cortex_enrich
    try:
        await cortex_enrich(payload)
    except Exception:
        # Never continue without memory after an opted-in retrieval failure.
        raise HTTPException(status_code=503, detail="Cortex automatic memory retrieval failed; generation was not started.") from None
''' + END


def install(backend, remove=False):
    route = backend/'routes/inference.py'
    original = route.read_text()
    text = original
    if BEGIN in text:
        start = text.index(BEGIN);end = text.index(END,start)+len(END)
        text = text[:start]+text[end:]
    if not remove:
        if text.count(ANCHOR)!=1:
            raise RuntimeError('Unsupported Unsloth version: expected a single generation entry point; no changes made')
        text = text.replace(ANCHOR,HOOK+ANCHOR,1)
    if text!=original:
        backup=Path.home()/'.local/state/cortex-unsloth-backups'/datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        backup.mkdir(parents=True,mode=0o700)
        shutil.copy2(route,backup/'inference.py')
        compile(text,str(route),'exec')
        temp=route.with_suffix('.py.cortex-tmp');temp.write_text(text)
        shutil.copymode(route,temp);temp.replace(route)
    module=backend/'core/inference/cortex_memory_gate.py'
    if remove:module.unlink(missing_ok=True)
    else:shutil.copy2(Path(__file__).with_name('cortex_memory_gate.py'),module)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend-root',type=Path,required=True)
    p.add_argument('--server-id')
    p.add_argument('--namespace',default='main')
    p.add_argument('--uninstall',action='store_true')
    args=p.parse_args()
    install(args.backend_root,args.uninstall)
    config=Path.home()/'.config/unsloth-cortex-memory.json'
    if args.server_id:
        config.parent.mkdir(parents=True,exist_ok=True)
        config.write_text(json.dumps({'enabled':not args.uninstall,'server_id':args.server_id,
                                      'namespace':args.namespace},indent=2)+'\n');config.chmod(0o600)
    elif args.uninstall and config.exists():
        value=json.loads(config.read_text());value['enabled']=False
        config.write_text(json.dumps(value,indent=2)+'\n')
    print('Hook removed.' if args.uninstall else 'Hook installed. Restart Unsloth Studio to activate.')


if __name__=='__main__':main()
