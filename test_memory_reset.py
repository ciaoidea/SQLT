"""Confirmed reset controls; real writes are restricted to owned test fixtures."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch,MagicMock
import uuid
from episodic_memory import EpisodicMemory
from memory_admin import reset_memory
from clients.codex_chat import CodexMemoryClient,reset_interactively


class ResetTests(unittest.TestCase):
    def setUp(self):
        self.memory=EpisodicMemory('check_reset_'+uuid.uuid4().hex[:8])
    def tearDown(self):
        self.memory.delete_namespace();self.memory.close()
    def remember(self,memory):
        return memory.remember('give',[dict(role='agent',value='marco'),
            dict(role='recipient',value='luca'),dict(role='patient',value='libro')],
            'Marco gives Luca a book.')['event_id']
    def test_reset_cascades_and_preserves_other_namespace(self):
        other=EpisodicMemory(self.memory.namespace+'_b')
        try:
            ident=self.remember(self.memory)
            self.memory.observe_outcome(ident,'holder','luca','Observed holder Luca')
            archived=self.remember(self.memory);self.memory.deactivate([archived])
            other_id=self.remember(other)
            result=reset_memory(self.memory.namespace,'RESET '+self.memory.namespace)
            self.assertEqual(result['deleted_episodes'],2)
            self.assertEqual(result['deleted_operators'],1)
            self.assertEqual(self.memory.status()['events'],0)
            self.assertEqual(self.memory.status()['archived_events'],0)
            with self.memory.c.cursor() as cursor:
                for table in ('episodic_semantic_vectors','episodic_bindings','episodic_outcomes'):
                    cursor.execute(f'SELECT COUNT(*) FROM {table} WHERE event_id IN (%s,%s)',(ident,archived))
                    self.assertEqual(cursor.fetchone()[0],0)
                for table in ('episodic_entities','episodic_operators'):
                    cursor.execute(f'SELECT COUNT(*) FROM {table} WHERE namespace=%s',(self.memory.namespace,))
                    self.assertEqual(cursor.fetchone()[0],0)
            self.assertEqual(other.find('give',[])[0]['event_id'],other_id)
        finally:other.delete_namespace();other.close()
    def test_wrong_confirmation_cannot_open_connection(self):
        with patch('memory_admin.connect') as connect:
            for confirmation in ('','RESET main','yes'):
                with self.assertRaises(ValueError):reset_memory(self.memory.namespace,confirmation)
            connect.assert_not_called()
    def test_failed_transaction_restores_cascaded_rows(self):
        ident=self.remember(self.memory)
        from storage import connect
        connection=connect();original=connection.cursor
        def failing_cursor():
            cursor=original();execute=cursor.execute
            def run(sql,args=None):
                if sql.startswith('DELETE FROM episodic_operators'):
                    raise RuntimeError('Simulated transaction failure')
                return execute(sql,args)
            cursor.execute=run
            return cursor
        with patch('memory_admin.connect',return_value=connection),patch.object(connection,'cursor',side_effect=failing_cursor):
            with self.assertRaisesRegex(RuntimeError,'Simulated'):reset_memory(self.memory.namespace,'RESET '+self.memory.namespace)
        self.memory.c.rollback()
        self.assertEqual(self.memory.find('give',[])[0]['event_id'],ident)
        with self.memory.c.cursor() as cursor:
            cursor.execute('SELECT COUNT(*) FROM episodic_semantic_vectors WHERE event_id=%s',(ident,))
            self.assertGreater(cursor.fetchone()[0],0)
    def test_client_discards_history_only_after_commit(self):
        client=CodexMemoryClient(self.memory.namespace)
        client.transport=MagicMock();client.last={'answer':'Old session fact'}
        with patch('memory_admin.reset_memory',side_effect=RuntimeError('failure')):
            with self.assertRaises(RuntimeError):client.reset_memory('RESET '+client.namespace)
        client.transport.close.assert_not_called()
        self.assertIsNotNone(client.last)
        with patch('memory_admin.reset_memory',return_value={}):client.reset_memory('RESET '+client.namespace)
        client.transport.close.assert_called_once()
        self.assertIsNone(client.last)
    def test_interactive_cancel_and_confirmation(self):
        client=MagicMock();client.namespace=self.memory.namespace
        for answer in ('','yes','RESET main'):
            with patch('builtins.input',return_value=answer),redirect_stdout(io.StringIO()):reset_interactively(client)
        client.reset_memory.assert_not_called();client.prepare.assert_not_called()
        with patch('builtins.input',return_value='RESET '+client.namespace),redirect_stdout(io.StringIO()):reset_interactively(client)
        client.reset_memory.assert_called_once_with('RESET '+client.namespace)
        client.prepare.assert_called_once()

if __name__=='__main__':unittest.main()
