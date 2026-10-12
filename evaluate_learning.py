"""Read-only before/after answers using a training run's original SQL weights.

Historical weights replace current parameters only in this process's RAM.
No model weights or database revisions are written.
"""
import argparse
import hashlib
import json
import numpy as np
import torch
from runtime import DatabaseMoE
from model_tools import CHAT_INSTRUCTIONS


def original_weights(connection, run_id):
    with connection.cursor() as cur:
        cur.execute('SELECT p.*,b.sha256 AS original_sha256 FROM moe_training_parameters b JOIN moe_parameters p ON p.id=b.parameter_id WHERE b.run_id=%s',(run_id,))
        metadata=cur.fetchall()
    if not metadata: raise ValueError('No saved original parameters for this run')
    result={}
    for meta in metadata:
        dim=meta['dimensions']
        if dim not in (0,512,1536): raise ValueError('Unsupported vector dimension')
        table=f'moe_training_vectors_{dim}' if dim else 'moe_training_scalars'
        field='coefficients' if dim else 'coefficient'
        with connection.cursor() as cur:
            cur.execute(f'SELECT unit_no,{field} FROM {table} WHERE run_id=%s AND parameter_id=%s ORDER BY unit_no',(run_id,meta['id']))
            rows=cur.fetchall()
        if [row['unit_no'] for row in rows]!=list(range(meta['row_count'])):
            raise ValueError('Missing historical parameter rows')
        raw=(b''.join(row[field] for row in rows) if dim else
             np.asarray([row[field] for row in rows],dtype='<f4').tobytes())
        if hashlib.sha256(raw).hexdigest()!=meta['original_sha256']:
            raise ValueError('Historical parameter checksum mismatch')
        result[meta['name']]=torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy().reshape(json.loads(meta['shape'])))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=int,required=True)
    parser.add_argument('--question',action='append',required=True)
    parser.add_argument('--tokens',type=int,default=80)
    args=parser.parse_args()
    if args.tokens<1:parser.error('tokens must be positive')
    model=DatabaseMoE(3584,quantized_experts=True,device='cpu',db_workers=2)
    try:
        with model.connection.cursor() as cur:
            cur.execute('SELECT saved_revision FROM moe_training_runs WHERE id=%s',(args.run,))
            run=cur.fetchone()
        if not run or run['saved_revision']!=model.metadata['revision']:
            raise ValueError('Compare only the currently applied training run')
        old=original_weights(model.connection,args.run)
        for phase in ('after','before'):
            if phase=='before':
                with torch.no_grad():
                    for name,weight in old.items():model.model.get_parameter(name).copy_(weight)
            for question in args.question:
                model.reset_context()
                messages=[{'role':'system','content':CHAT_INSTRUCTIONS},{'role':'user','content':question}]
                response=model.generate(messages,max_tokens=args.tokens)
                print(json.dumps(dict(phase=phase,question=question,answer=response['answer'],
                    finish_reason=response['finish_reason'],seconds=response['seconds'],
                    database_weights_changed=False),ensure_ascii=False),flush=True)
    finally:model.close()

if __name__=='__main__':main()
