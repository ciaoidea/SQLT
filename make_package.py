"""Build a source installer archive using an explicit public-file allowlist."""
import argparse
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent
FILES=('nlq_context.py','memory_context.py','mcp_lan.py','MEMORY_RETRIEVAL.md','integrations/unsloth/install.py','integrations/unsloth/cortex_memory_gate.py','integrations/unsloth/test_gate.py','integrations/unsloth/README.md','query_flow.py','live_dynamics.py','LIVE_DYNAMICS.md','structure_sync.py','STRUCTURE.md','AGENTS.md','README.md','INSTALL.md','CLAUDE.md','GEMINI.md','requirements.txt','cortex',
       'install.sh','installer.py','make_package.py','storage.py','episodic_memory.py',
       'apfc.py','apfc_browser.py','apfc_experience.py','APFC.md','verify_apfc.py','verify_apfc_chat.py','semantic_memory.py','relational_memory.py','mcp_server.py','memory_admin.py','encoder_runtime.py',
       'encoder_worker.py','verify_mcp.py','verify_semantic.py','verify_gpt_mcp.py',
       'verify_natural_chat.py','verify_forgetting.py','EXPERIMENTS.md','.gitignore')


def build(destination):
    paths=[ROOT/name for name in FILES if (ROOT/name).is_file()]
    paths+=sorted(ROOT.glob('test_*.py'))
    paths+=sorted((ROOT/'clients').glob('*.py'))
    viewer_files=('live.js','map.html','map.css','map.js','server.py','planner.py','plan.schema.json',
                  'test_viewer.py','test_plan_cache.py','README.txt','vendor/cytoscape.min.js','vendor/LICENSE-cytoscape')
    paths += [ROOT/'sqlt.net'/name for name in viewer_files if (ROOT/'sqlt.net'/name).is_file()]
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(destination,'w:gz') as archive:
        for path in paths:
            if path.is_symlink():raise ValueError('Refusing symlink in source package')
            info=archive.gettarinfo(str(path),arcname='CORTEX/'+str(path.relative_to(ROOT)))
            info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
            info.mode=0o755 if path.name in ('cortex','install.sh') else 0o644
            with path.open('rb') as source:archive.addfile(info,source)
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'dist/cortex-installer.tar.gz'))
    print(build(parser.parse_args().output))
