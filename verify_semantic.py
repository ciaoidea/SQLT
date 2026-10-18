"""Live local semantic retrieval tests: no GPT and no synonym/intent parser."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import json
from pathlib import Path
import resource
import statistics
import sys
import time
import uuid
from episodic_memory import EpisodicMemory
import semantic_memory as semantic


def run():
    namespace='check_sem_'+uuid.uuid4().hex[:10]
    memory=EpisodicMemory(namespace)
    other=EpisodicMemory(namespace+'_b')
    report={'scope':'Local multilingual semantic retrieval, without GPT, lexical aliases or exact-label query filters. Similarity is not logical entailment.',
            'model':semantic.MODEL,'queries':[]}
    try:
        facts=[
            ('phone',[('entity','ufficio it'),('number','0000123456')],
             'Il numero telefonico dell’ufficio IT è 0000123456.'),
            ('phone',[('entity','risorse umane'),('number','0000654321')],
             'Per contattare telefonicamente le risorse umane si chiama 0000654321.'),
            ('network',[('entity','accesso vpn'),('address','vpn.example.test')],
             'Il portale per accedere alla rete privata aziendale VPN è vpn.example.test.'),
            ('foundation',[('entity','directa'),('place','torino'),('year','1995')],
             'Directa, intermediario di trading online, fu fondata a Torino nel 1995.'),
            ('office',[('entity','ufficio acquisti'),('floor','terzo')],
             'L’ufficio acquisti si trova al terzo piano.'),
            ('recipe',[('entity','pane'),('ingredient','farina')],
             'Per preparare il pane si possono usare farina, acqua, lievito e sale.'),
        ]
        ids=[]
        for frame,roles,source in facts:
            ids.append(memory.remember(frame,[dict(role=r,value=v) for r,v in roles],source)['event_id'])
        checks=[
            ('come posso contattare l’IT',0),
            ('Come posso mettermi in contatto con il reparto informatico?',0),
            ('How can I phone the IT department?',0),
            ('Qual è il recapito telefonico del personale?',1),
            ('Dove accedo alla rete privata dell’azienda?',2),
            ('In quale città e anno è nato il broker Directa?',3),
            ('A quale piano devo andare per parlare con gli acquisti?',4),
            ('Quali ingredienti servono per fare il pane?',5),
        ]
        times=[]
        for question,expected in checks:
            started=time.perf_counter();found=semantic.search(memory,question,limit=3)
            times.append(time.perf_counter()-started)
            ranks=[e['event_id'] for e in found['events']]
            report['queries'].append(dict(query=question,top1_correct=ranks[0]==ids[expected],
                expected_rank=ranks.index(ids[expected])+1 if ids[expected] in ranks else None,
                encoding_ms=found['encoding_seconds']*1000,sql_ms=found['sql_seconds']*1000))
        report['top1_correct']=sum(q['top1_correct'] for q in report['queries'])
        report['top3_correct']=sum(q['expected_rank'] is not None for q in report['queries'])
        # Preserve direction and polarity as evidence; do not pretend the encoder
        # alone proves which logical proposition follows from a sentence.
        transfer=[dict(role='agent',value='marco'),dict(role='recipient',value='luca'),dict(role='patient',value='libro')]
        positive=memory.remember('give',transfer,'Marco dà un libro a Luca.')['event_id']
        negative=memory.remember('give',transfer,'Marco non dà un libro a Luca.',negated=True)['event_id']
        reverse=memory.remember('give',[dict(role='agent',value='luca'),dict(role='recipient',value='marco'),dict(role='patient',value='libro')],'Luca dà un libro a Marco.')['event_id']
        found=semantic.search(memory,'Passaggio di un libro fra Marco e Luca',limit=16)['events']
        by_id={e['event_id']:e for e in found}
        report['role_and_negation_evidence_preserved']=(not by_id[positive]['negated'] and by_id[negative]['negated'] and
            by_id[positive]['roles']!=by_id[reverse]['roles'])
        memory.deactivate([ids[0]])
        report['archive_excluded']=ids[0] not in [e['event_id'] for e in semantic.search(memory,checks[0][0],limit=16)['events']]
        memory.restore([ids[0]])
        reloaded=EpisodicMemory(namespace)
        try:
            report['restart_retrieval']=semantic.search(reloaded,checks[0][0])['events'][0]['event_id']==ids[0]
        finally:reloaded.close()
        report['other_namespace_isolated']=semantic.search(other,checks[0][0])['events']==[]
        long_source=('Questa nota descrive materiali e procedure di archivio. '*40)+\
            'La chiave della cassaforte è custodita nel cassetto viola del magazzino.'
        long_id=memory.remember('storage',[dict(role='item',value='chiave cassaforte'),dict(role='place',value='cassetto viola')],long_source)['event_id']
        found=semantic.search(memory,'Dove viene conservata la chiave della cassaforte?',limit=3)
        report['long_source_tail_retrieved']=found['events'][0]['event_id']==long_id
        with memory.c.cursor() as q:
            q.execute('SHOW CREATE TABLE episodic_semantic_vectors')
            report['native_vector_index']='VECTOR KEY' in q.fetchone()[1]
        report['warm_search_p50_ms']=statistics.median(times)*1000
        report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        report['resident_client_mib']=int(next(line for line in open('/proc/self/status') if line.startswith('VmRSS:')).split()[1])/1024
        from encoder_runtime import request
        report['shared_encoder_rss_mib']=request('status')['rss_mib']
        report['torch_loaded']='torch' in sys.modules
        report['passed']=report['top1_correct']==len(checks) and all(report[k] for k in (
            'role_and_negation_evidence_preserved','archive_excluded','restart_retrieval',
            'other_namespace_isolated','long_source_tail_retrieved','native_vector_index'))
    finally:
        memory.delete_namespace();memory.close()
        other.delete_namespace();other.close()
        Path(__file__).with_name('verification-semantic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report.get('passed',False)

if __name__=='__main__':raise SystemExit(0 if run() else 1)
