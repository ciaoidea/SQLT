#!/usr/bin/env python3
"""Real language-client lifecycle check; isolated synthetic scenario, no router calls."""
import json
from pathlib import Path
import uuid
from apfc import APFC
from clients.codex_chat import CodexMemoryClient
from episodic_memory import EpisodicMemory


def run():
    ns='check_'+uuid.uuid4().hex[:12];key='router-test-'+uuid.uuid4().hex[:6]
    report=dict(scope='Synthetic narrative through real language client and real MCP; no network equipment controlled.',namespace=ns)
    try:
        first=CodexMemoryClient(ns).turn('Apriamo la pratica '+key+'. Il router è raggiungibile via Ethernet, ma Internet non funziona. Ho appena riconnesso i dati mobili. Aspetto di verificare se Internet torna al prossimo controllo. Tieni traccia del lavoro e dell’intervento effettuato; non conosco ancora il risultato.',fresh=True)
        m=EpisodicMemory(ns)
        try:before=APFC(m).status()
        finally:m.close()
        report['first']=dict(answer=first['answer'],seconds=first['seconds'],tools=[c['tool'] for c in first['calls']],status=before)
        print('Intervention: '+json.dumps(report['first'],ensure_ascii=False),flush=True)
        if before['experiences'].get('pending',0)!=1 or before['learned_observations']!=0:
            raise AssertionError('Natural-language intervention was not tracked as pending without training')
        second=CodexMemoryClient(ns).turn('Aggiornamento della pratica '+key+': al primo controllo successivo alla riconnessione Internet funziona di nuovo. Ethernet è rimasta collegata. Questo è il risultato osservato di quello stesso intervento.',fresh=True)
        m=EpisodicMemory(ns)
        try:after=APFC(m).status()
        finally:m.close()
        report['second']=dict(answer=second['answer'],seconds=second['seconds'],tools=[c['tool'] for c in second['calls']],status=after,learning_updated=second.get('learning_updated'))
        print('Outcome: '+json.dumps(report['second'],ensure_ascii=False),flush=True)
        if after['experiences'].get('pending',0)!=0 or after['experiences'].get('completed',0)!=1 or after['learned_observations']!=1:
            raise AssertionError('Natural-language observed outcome did not complete exactly one learning update')
        report['passed']=True
    finally:
        m=EpisodicMemory(ns)
        try:m.delete_namespace()
        finally:m.close()
        out=Path(__file__).resolve().parent/'outputs'/'apfc-chat-verification.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':
    r=run();print(json.dumps({'passed':r.get('passed',False),'report':'outputs/apfc-chat-verification.json'}))
