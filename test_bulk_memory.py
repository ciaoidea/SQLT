"""Atomic bulk memory and real terminal bracketed-paste controls."""
import asyncio
import unittest
from unittest.mock import patch
import uuid
from episodic_memory import EpisodicMemory
import semantic_memory
from clients.terminal_input import create_session
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput


class BulkTests(unittest.TestCase):
    def setUp(self):self.memory=EpisodicMemory('check_bulk_'+uuid.uuid4().hex[:10])
    def tearDown(self):self.memory.delete_namespace();self.memory.close()
    def events(self):
        return [dict(frame='branch',roles=[dict(role='company',value='società di prova'),
                    dict(role='branch',value='sede '+city),dict(role='city',value=city),
                    dict(role='phone',value=phone)],
                    source=f'La sede di prova di {city} si chiama al {phone}.')
                for city,phone in [('torino','00001111'),('bologna','00002222'),('trento','00003333')]]
    def test_batch_preserves_complete_entities_and_semantic_recall(self):
        result=self.memory.remember_many(self.events())
        self.assertEqual(result['stored'],3)
        self.assertEqual(len(set(result['event_ids'])),3)
        found=semantic_memory.search(self.memory,'Come telefono alla sede di Trento?')['events'][0]
        self.assertIn(dict(role='phone',value='00003333'),found['roles'])
        self.assertEqual(found['event_id'],result['event_ids'][2])
        with self.memory.c.cursor() as cursor:
            cursor.execute('SELECT COUNT(*) FROM episodic_semantic_vectors WHERE event_id IN (%s,%s,%s)',result['event_ids'])
            self.assertEqual(cursor.fetchone()[0],3)
    def test_invalid_final_record_writes_nothing(self):
        events=self.events();events[-1]['roles'].append(events[-1]['roles'][0])
        with patch('semantic_memory.embed_passages') as encode:
            with self.assertRaises(ValueError):self.memory.remember_many(events)
            encode.assert_not_called()
        self.assertEqual(self.memory.status()['events'],0)
    def test_late_insert_failure_rolls_back_whole_batch(self):
        original=self.memory.c.cursor
        def cursor_factory():
            cursor=original();execute=cursor.execute
            def run(sql,args=None):
                statement=sql if isinstance(sql,str) else bytes(sql).decode()
                if statement.startswith('INSERT INTO episodic_bindings'):
                    raise RuntimeError('Injected batch failure')
                return execute(sql,args)
            cursor.execute=run;return cursor
        with patch.object(self.memory.c,'cursor',side_effect=cursor_factory):
            with self.assertRaisesRegex(RuntimeError,'Injected'):self.memory.remember_many(self.events())
        self.assertEqual(self.memory.status()['events'],0)
        self.assertEqual(semantic_memory.search(self.memory,'sede di Trento')['events'],[])
    def test_source_budgets_validate_before_encoding(self):
        events=self.events()*22
        with self.assertRaises(ValueError):self.memory.remember_many(events)
        with self.assertRaises(ValueError):self.memory.remember_many([dict(self.events()[0],source='x'*8193)])
    def test_one_encoder_ipc_batch_for_short_sources(self):
        from encoder_runtime import request
        with patch('encoder_runtime.request',wraps=request) as ipc:
            groups=semantic_memory.embed_passages([e['source'] for e in self.events()])
        self.assertEqual(len(groups),3)
        self.assertEqual([call.args[0] for call in ipc.call_args_list],['passages'])
    def test_135_entities_use_three_atomic_batches_without_loss(self):
        events=[dict(frame='branch',roles=[dict(role='company',value='società prova'),
                    dict(role='branch',value=f'sede {i}'),dict(role='phone',value=f'0000{i:04d}')],
                    source=f'La sede {i} della società di prova risponde al numero 0000{i:04d}.')
                for i in range(135)]
        ids=[]
        for offset in range(0,len(events),64):
            ids.extend(self.memory.remember_many(events[offset:offset+64])['event_ids'])
        self.assertEqual(len(set(ids)),135)
        self.assertEqual(self.memory.status()['events'],135)
        for i in (0,63,64,127,128,134):
            found=self.memory.find('branch',[dict(role='branch',value=f'sede {i}')])
            self.assertEqual(len(found),1)
            self.assertIn(dict(role='phone',value=f'0000{i:04d}'),found[0]['roles'])


class PasteTests(unittest.IsolatedAsyncioTestCase):
    async def test_135_line_paste_is_one_message_not_135_commands(self):
        pasted='\n'.join(f'Filiale di prova {i}: via Esempio {i+1}' for i in range(135))
        with create_pipe_input() as pipe:
            session=create_session(input=pipe,output=DummyOutput())
            task=asyncio.create_task(session.prompt_async('❯ '))
            await asyncio.sleep(.02)
            pipe.send_text('\x1b[200~'+pasted+'\x1b[201~\r')
            received=await asyncio.wait_for(task,2)
            self.assertEqual(received,pasted)
            self.assertEqual(len(received.splitlines()),135)
            task=asyncio.create_task(session.prompt_async('❯ '))
            await asyncio.sleep(.02);pipe.send_text('/quit\r')
            self.assertEqual(await asyncio.wait_for(task,2),'/quit')
    async def test_alt_enter_inserts_newline_enter_sends(self):
        with create_pipe_input() as pipe:
            session=create_session(input=pipe,output=DummyOutput())
            task=asyncio.create_task(session.prompt_async('❯ '))
            await asyncio.sleep(.02);pipe.send_text('prima\x1b\rseconda\r')
            self.assertEqual(await asyncio.wait_for(task,2),'prima\nseconda')


if __name__=='__main__':unittest.main()
