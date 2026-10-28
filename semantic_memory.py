"""Learned multilingual embeddings and native MariaDB semantic retrieval.

The encoder is frozen and local. New experiences are encoded immediately; this
does not claim to update encoder/GPT weights or derive logic from similarity.
"""
from functools import lru_cache
import base64
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import numpy as np
from storage import retry_transaction

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('RAYON_NUM_THREADS','1')

MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
DIM = 384
CHUNK_CHARS = 480
CHUNK_STRIDE = 384
_lock = threading.RLock()
_encoder = None
_identity = None


def encoder():
    global _encoder, _identity
    with _lock:
        if _encoder is None:
            from fastembed import TextEmbedding
            model = TextEmbedding(MODEL, cache_dir=str(Path(__file__).parent/'.semantic-model'),
                                  threads=1, providers=['CPUExecutionProvider'])
            model_dir = Path(model.model._model_dir)
            artifact = model_dir/'model_optimized.onnx'
            with artifact.open('rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            config = dict(model=MODEL, weights_sha256=checksum, dimension=DIM,
                          chunk_chars=CHUNK_CHARS, stride=CHUNK_STRIDE, pooling='fastembed')
            _identity = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),config
            _encoder = model
        return _encoder


def _local_identity():
    encoder()
    return _identity


def chunks(text):
    if not isinstance(text,str) or not text.strip() or len(text.encode('utf-8'))>60000:
        raise ValueError('Semantic text must contain text and fit in 60000 UTF-8 bytes')
    # Character windows are language-independent and preserve all source text;
    # no intent parser, keyword rules or hand-written synonym dictionary.
    result=[]
    for offset in range(0,len(text),CHUNK_STRIDE):
        result.append(text[offset:offset+CHUNK_CHARS])
        if offset+CHUNK_CHARS>=len(text):break
    return result


def _local_embed_passage(text):
    return _local_embed_passages([text])[0]


def _local_embed_passages(texts):
    groups=[chunks(text) for text in texts]
    if not 1<=len(groups)<=64 or sum(map(len,groups))>40:
        raise ValueError('Encoder batch budget exceeded')
    passages=[part for group in groups for part in group]
    with _lock:
        vectors=list(encoder().passage_embed(passages,batch_size=8,parallel=None))
    result=[]
    for passage,vector in zip(passages,vectors):
        vector=np.asarray(vector,dtype='<f4')
        norm=np.linalg.norm(vector)
        if vector.shape!=(DIM,) or not np.isfinite(vector).all() or norm<=0:
            raise ArithmeticError('Invalid semantic embedding')
        result.append((passage,(vector/norm).astype('<f4').tobytes()))
    grouped=[];offset=0
    for group in groups:
        grouped.append(result[offset:offset+len(group)]);offset+=len(group)
    return grouped


def _local_embed_query(text):
    parts=chunks(text)
    with _lock:
        vectors=list(encoder().query_embed(parts,batch_size=8,parallel=None))
    vector=np.mean(vectors,axis=0).astype('<f4')
    norm=np.linalg.norm(vector)
    if vector.shape!=(DIM,) or not np.isfinite(vector).all() or norm<=0:
        raise ArithmeticError('Invalid query embedding')
    return (vector/norm).astype('<f4').tobytes()


def identity():
    from encoder_runtime import request
    result=request('identity')
    return result['encoder_id'],result['config']


def embed_passage(text):
    return embed_passages([text])[0]


def embed_passages(texts):
    from encoder_runtime import request
    unique=list(dict.fromkeys(texts))
    groups={text:chunks(text) for text in unique}
    encoded={}
    # Keep each IPC request bounded, even for a long original source.
    pending=[]; locations=[]
    def flush():
        result=request('passages',pending)
        if len(result['vectors'])!=len(pending):raise ArithmeticError('Invalid embedding batch')
        for (text,index),raw in zip(locations,result['vectors']):
            if len(raw)!=1:raise ArithmeticError('Invalid fragment response')
            vector=base64.b64decode(raw[0],validate=True)
            if len(vector)!=DIM*4:raise ArithmeticError('Invalid shared embedding shape')
            encoded[text][index]=(groups[text][index],vector)
    for text in unique:
        encoded[text]=[None]*len(groups[text])
        for index,fragment in enumerate(groups[text]):
            pending.append(fragment);locations.append((text,index))
            if len(pending)==32:
                flush();pending=[];locations=[]
    if pending:flush()
    return [encoded[text] for text in texts]


@lru_cache(maxsize=64)
def embed_query(text):
    from encoder_runtime import request
    chunks(text)
    vector=base64.b64decode(request('query',text)['vector'],validate=True)
    if len(vector)!=DIM*4:raise ArithmeticError('Invalid shared query shape')
    return vector


def ensure_schema(connection):
    with connection.cursor() as q:
        q.execute('''CREATE TABLE IF NOT EXISTS semantic_encoders(
            encoder_id CHAR(64) PRIMARY KEY, config JSON NOT NULL,
            created TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6)) ENGINE=InnoDB''')
        q.execute('''CREATE TABLE IF NOT EXISTS episodic_semantic_vectors(
            id BIGINT AUTO_INCREMENT PRIMARY KEY, event_id BIGINT NOT NULL,
            chunk_no INT NOT NULL, encoder_id CHAR(64) NOT NULL,
            source_sha256 CHAR(64) NOT NULL, fragment TEXT NOT NULL,
            embedding VECTOR(384) NOT NULL,
            UNIQUE KEY event_chunk(event_id,chunk_no,encoder_id),
            FOREIGN KEY(event_id) REFERENCES episodic_events(id) ON DELETE CASCADE,
            FOREIGN KEY(encoder_id) REFERENCES semantic_encoders(encoder_id),
            VECTOR INDEX semantic_cosine(embedding) M=8 DISTANCE=cosine) ENGINE=InnoDB''')


def insert_vectors(cursor,event_id,source,encoded):
    insert_vector_batch(cursor,[(event_id,source,encoded)])


def insert_vector_batch(cursor,entries):
    encoder_id,config=identity()
    # The registry is immutable: avoid touching the shared parent row on every
    # event. Duplicate INSERT IGNORE locks can invalidate concurrent FK checks
    # under MariaDB snapshot isolation, and serve no purpose once registered.
    cursor.execute('SELECT 1 FROM semantic_encoders WHERE encoder_id=%s',(encoder_id,))
    if not cursor.fetchone():
        cursor.execute('INSERT IGNORE INTO semantic_encoders VALUES(%s,%s,DEFAULT)',
                       (encoder_id,json.dumps(config)))
    rows=[]
    for event_id,source,encoded in entries:
        digest=hashlib.sha256(source.encode()).hexdigest()
        rows.extend((event_id,i,encoder_id,digest,part,vector) for i,(part,vector) in enumerate(encoded))
    cursor.executemany('''INSERT IGNORE INTO episodic_semantic_vectors
        (event_id,chunk_no,encoder_id,source_sha256,fragment,embedding)
        VALUES(%s,%s,%s,%s,%s,%s)''',
        rows)


@retry_transaction
def backfill(memory,batch_size=32):
    """Incremental, restartable migration; preserves original events/operators."""
    started=time.perf_counter();encoder_id,_=identity()
    memory.c.commit()
    with memory.c.cursor() as q:
        q.execute('''SELECT e.id,e.source FROM episodic_events e
            WHERE e.namespace=%s AND NOT EXISTS(SELECT 1 FROM episodic_semantic_vectors v
            WHERE v.event_id=e.id AND v.encoder_id=%s) ORDER BY e.id LIMIT %s''',
            (memory.namespace,encoder_id,batch_size))
        pending=q.fetchall()
    memory.c.commit()
    for event_id,source in pending:
        encoded=embed_passage(source)
        memory.c.begin()
        try:
            with memory.c.cursor() as q:
                q.execute('SELECT source FROM episodic_events WHERE id=%s AND namespace=%s FOR UPDATE',
                          (event_id,memory.namespace))
                row=q.fetchone()
                if row and row[0]==source:insert_vectors(q,event_id,source,encoded)
            memory.c.commit()
        except BaseException:memory.c.rollback();raise
    with memory.c.cursor() as q:
        q.execute('''SELECT COUNT(*) FROM episodic_events e WHERE e.namespace=%s
            AND NOT EXISTS(SELECT 1 FROM episodic_semantic_vectors v
            WHERE v.event_id=e.id AND v.encoder_id=%s)''',(memory.namespace,encoder_id))
        remaining=q.fetchone()[0]
    return dict(indexed=len(pending),remaining=remaining,seconds=time.perf_counter()-started)


def search(memory,text,limit=8,negated=None,include_sql=False,filters=None,frame=None):
    if not isinstance(limit,int) or isinstance(limit,bool) or not 1<=limit<=16:
        raise ValueError('Semantic retrieval limit must be 1..16')
    if negated is not None and not isinstance(negated,bool):raise ValueError('Invalid polarity')
    started=time.perf_counter();vector=embed_query(text);encoder_id,_=identity()
    encoded_seconds=time.perf_counter()-started
    clauses=['e.namespace=%s','e.active=1','v.encoder_id=%s']
    params=[vector,memory.namespace,encoder_id]
    if negated is not None:clauses.append('e.negated=%s');params.append(negated)
    if filters or frame is not None:
        from relational_memory import where
        clause,extra=where(memory.namespace,frame,filters or [],negated)
        clauses.append(clause);params.extend(extra)
    params.append(limit)
    memory.c.commit();sql_started=time.perf_counter()
    with memory.c.cursor() as q:
        sql='''SELECT e.id,e.frame,e.negated,e.roles,e.source,e.event_time,e.place,
            MIN(VEC_DISTANCE_COSINE(v.embedding,%s)) AS distance
            FROM episodic_semantic_vectors v JOIN episodic_events e ON e.id=v.event_id
            WHERE '''+' AND '.join(clauses)+'''
            GROUP BY e.id,e.frame,e.negated,e.roles,e.source,e.event_time,e.place
            ORDER BY distance,e.id LIMIT %s'''
        q.execute(sql,params)
        events=[dict(event_id=r[0],frame=r[1],negated=bool(r[2]),roles=json.loads(r[3]),
                     source=r[4],time=r[5],place=r[6],distance=float(r[7]),
                     similarity=1-float(r[7])) for r in q.fetchall()]
    result=dict(events=events,count=len(events),encoder=MODEL,dimension=DIM,
                encoder_id=encoder_id,encoding_seconds=encoded_seconds,
                sql_seconds=time.perf_counter()-sql_started,
                scope='Semantic nearest-neighbor candidates, not logical proof or calibrated confidence. '
                      'Check source evidence, roles, negation, identity and context before answering. '
                      'An unrelated nearest neighbor is not evidence that the requested fact exists.')
    if include_sql:
        result['sql_trace']=[dict(sql=sql,parameters=[dict(type='VECTOR(384)',encoding='base64-float32-le',
            value=base64.b64encode(vector).decode(),sha256=hashlib.sha256(vector).hexdigest())]+params[1:])]
    return result
