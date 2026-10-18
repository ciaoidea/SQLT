"""Shared encoder, bounded IPC and safe transaction-retry controls."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import stat
import subprocess
import sys
import unittest
from unittest.mock import MagicMock,patch
import uuid
import pymysql
from encoder_runtime import request,SOCKET,RUNTIME
from episodic_memory import EpisodicMemory
from storage import retry_transaction


class RuntimeTests(unittest.TestCase):
    def test_new_clients_share_model_and_cached_vectors(self):
        text='Shared encoder control '+uuid.uuid4().hex
        first=request('query',text)
        code="import json;from encoder_runtime import request;print(json.dumps(request('query',%r)))"%text
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,
                              capture_output=True,text=True,check=True)
        second=json.loads(result.stdout)
        self.assertEqual(first['pid'],second['pid'])
        self.assertEqual(first['vector'],second['vector'])
        self.assertTrue(second['cache_hit'])
        self.assertLessEqual(second['cached_queries'],64)
        self.assertEqual(stat.S_IMODE(SOCKET.stat().st_mode),0o600)
        self.assertEqual(stat.S_IMODE(RUNTIME.stat().st_mode),0o700)
        self.assertNotIn('fastembed',sys.modules)
    def test_concurrent_clients_share_worker(self):
        queries=['Parallel semantic runtime check '+uuid.uuid4().hex for _ in range(8)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(lambda text:request('query',text),queries))
        self.assertEqual(len({r['pid'] for r in results}),1)
    def test_invalid_operation_does_not_mutate_service(self):
        before=request('status')['pid']
        with self.assertRaises(RuntimeError):request('invalid')
        self.assertEqual(request('status')['pid'],before)
    def test_idle_exit_race_retries_only_pure_encoding(self):
        failed=MagicMock()
        failed.__enter__.return_value=failed
        failed.sendall.side_effect=BrokenPipeError()
        ready=MagicMock()
        ready.__enter__.return_value=ready
        ready.makefile.return_value.__enter__.return_value.readline.return_value=b'{"pid":123}\n'
        with patch('encoder_runtime._connection',side_effect=[failed,ready]) as connect,patch('encoder_runtime.time.sleep'):
            self.assertEqual(request('status')['pid'],123)
            self.assertEqual(connect.call_count,2)
    def test_retry_only_known_transaction_conflicts(self):
        class Operation:
            namespace='check_retry'
            def __init__(self):self.c=MagicMock();self.calls=0
            @retry_transaction
            def execute(self,code):
                self.calls+=1
                if self.calls==1:raise pymysql.OperationalError(code,'Injected conflict')
                return 'committed'
        operation=Operation()
        with patch('storage.time.sleep'),patch('storage.namespace_write_lock'):
            self.assertEqual(operation.execute(1020),'committed')
        self.assertEqual(operation.calls,2)
        operation.c.rollback.assert_called_once()
        ambiguous=Operation()
        with patch('storage.namespace_write_lock'),self.assertRaises(pymysql.OperationalError):ambiguous.execute(2013)
        self.assertEqual(ambiguous.calls,1)
    def test_concurrent_writes_remain_atomic_and_isolated(self):
        namespaces=['check_parallel_'+uuid.uuid4().hex[:10] for _ in range(4)]
        def write(namespace):
            memory=EpisodicMemory(namespace)
            try:
                for i in range(8):
                    memory.remember('runtime_test',[dict(role='entity',value=str(i))],
                                    f'Parallel test observation {i}.')
                return memory.status()['events']
            finally:memory.close()
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                self.assertEqual(list(pool.map(write,namespaces)),[8]*4)
        finally:
            for namespace in namespaces:
                memory=EpisodicMemory(namespace)
                try:memory.delete_namespace()
                finally:memory.close()
    def test_same_namespace_parallel_learning_keeps_observations(self):
        namespace='check_parallel_'+uuid.uuid4().hex[:10]
        def learn(_):
            memory=EpisodicMemory(namespace)
            try:
                for _ in range(3):
                    event=memory.remember('give',[dict(role='agent',value='marco'),
                        dict(role='recipient',value='luca'),dict(role='patient',value='libro')],
                        'Parallel fixture: Marco gives Luca a book.')
                    memory.observe_outcome(event['event_id'],'holder','luca','Observed holder Luca')
            finally:memory.close()
        memory=EpisodicMemory(namespace)
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(learn,range(4)))
            self.assertEqual(memory.status()['events'],12)
            with memory.c.cursor() as cursor:
                cursor.execute('SELECT examples FROM episodic_operators WHERE namespace=%s',(namespace,))
                self.assertEqual(cursor.fetchone()[0],12)
        finally:memory.delete_namespace();memory.close()


if __name__=='__main__':unittest.main()
