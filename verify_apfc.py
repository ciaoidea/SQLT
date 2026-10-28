#!/usr/bin/env python3
"""Bounded synthetic APFC learning benchmark, explicitly not general intelligence."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import json
import resource
import time
from pathlib import Path
import uuid
import numpy as np
from apfc import APFC, _CACHE
from episodic_memory import EpisodicMemory,tensor


def fixture(m,source):
    roles=[{'role':'entity','value':'synthetic numeric control'}]
    with m.c.cursor() as q:
        q.execute('INSERT INTO episodic_events(namespace,frame,negated,event_time,place,roles,source,provenance,representation) VALUES(%s,%s,0,%s,%s,%s,%s,%s,%s)',(m.namespace,'synthetic_apfc','','',json.dumps(roles),source,'supervised_example',tensor(roles).tobytes()));ident=q.lastrowid
        q.execute('INSERT INTO episodic_bindings VALUES(%s,%s,%s)',(ident,'entity',roles[0]['value']))
    m.c.commit();return ident


def benchmark():
    start=time.perf_counter();trials=[]
    for seed in range(3):
        rng=np.random.default_rng(seed);m=EpisodicMemory('check_'+uuid.uuid4().hex[:12]);a=APFC(m)
        try:
            a.register_policy('synthetic',['x_unitless','z_unitless'],['a','b'])
            test=rng.uniform(-1,1,(40,2));observations=[]
            for _ in range(40):
                x,z=rng.uniform(-1,1,2)
                for action,sign in [('a',1),('b',-1)]:
                    features={'x_unitless':float(x),'z_unitless':float(z)}
                    target=.5+sign*(.3*x+.15*z)
                    before=fixture(m,'Synthetic observed input '+json.dumps(features))
                    outcome=fixture(m,'Synthetic measured reward '+str(target)+' for '+action)
                    t=time.perf_counter();a.observe('synthetic',action,features,float(target),before,outcome,uuid.uuid4().hex);observations.append((time.perf_counter()-t)*1000)
            errors=[];baseline=[];correct=[];latencies=[];predictions=[]
            for x,z in test:
                t=time.perf_counter();r=a.predict('synthetic',{'x_unitless':float(x),'z_unitless':float(z)});latencies.append((time.perf_counter()-t)*1000)
                values=[v['estimated_reward'] for v in r['actions']];targets=[.5+.3*x+.15*z,.5-.3*x-.15*z]
                predictions.append(values);errors.extend((np.array(values)-targets)**2);baseline.extend((.5-np.array(targets))**2);correct.append(int(np.argmax(values)==np.argmax(targets)))
            _CACHE.clear();other=EpisodicMemory(m.namespace)
            try:
                x,z=test[0];loaded=APFC(other).predict('synthetic',{'x_unitless':float(x),'z_unitless':float(z)})
                reload_error=float(np.max(abs(np.array([v['estimated_reward'] for v in loaded['actions']])-predictions[0])))
            finally:other.close()
            trials.append(dict(seed=seed,training_observations=80,heldout_contexts=40,test_mse=float(np.mean(errors)),constant_baseline_mse=float(np.mean(baseline)),correct_action_fraction=float(np.mean(correct)),reload_max_abs_difference=reload_error,observe_p50_ms=float(np.median(observations)),observe_p95_ms=float(np.percentile(observations,95)),predict_p50_ms=float(np.median(latencies)),predict_p95_ms=float(np.percentile(latencies,95))))
        finally:m.delete_namespace();m.close()
    return dict(scope='Synthetic two-action linear reward task; supplied numeric features and known action set. Not natural-language comprehension or a validated real-world policy.',trials=trials,elapsed_seconds=time.perf_counter()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,per_action_tensor_bytes=(17+17*17)*8,cache_limit=32)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);args=p.parse_args();report=benchmark()
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
