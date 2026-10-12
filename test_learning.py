"""Training gradients and real MariaDB vector transaction checks."""
import hashlib
import json
import os
import unittest
import torch
from transformers import GraniteMoeConfig, GraniteMoeForCausalLM
from database import connect
from learn import blocks, select_parameters, train_step, save_weights, restore
from runtime import SQLExperts, load_parameter


class LearningTests(unittest.TestCase):
    def test_blocks_have_targets(self):
        batches = blocks(list(range(11)), 4)
        self.assertEqual([b.tolist()[0] for b in batches], [[0,1,2,3],[3,4,5,6],[6,7,8,9],[9,10]])
        self.assertEqual(blocks([1], 4), [])

    def test_backprop_through_sql_experts_changes_only_selected_weights(self):
        torch.manual_seed(13)
        cfg = GraniteMoeConfig(vocab_size=32, hidden_size=16, intermediate_size=8,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            num_local_experts=2, num_experts_per_tok=1)
        model = GraniteMoeForCausalLM(cfg).eval()
        class Cache:
            layer_count = 2
            router_seconds = 0
            routes = []
            def prepare(self, layer, selected): pass
            def get(self, layer, expert): return self.weights[layer, expert]
        cache = Cache()
        cache.weights = {}
        for i, layer in enumerate(model.model.layers):
            block = layer.block_sparse_moe
            for expert in range(2):
                cache.weights[i, expert] = (
                    block.input_linear.weight[expert].detach().clone(),
                    block.output_linear.weight[expert].detach().clone())
            layer.block_sparse_moe = SQLExperts(i, block.router, cache, block.activation)
        selected = select_parameters(model, 1)
        before = {n:p.detach().clone() for n,p in model.named_parameters()}
        batch = torch.tensor([[1,2,3,4,5,6]])
        initial = float(model(input_ids=batch, labels=batch, use_cache=False).loss.detach())
        optimizer = torch.optim.SGD(selected.values(), lr=0.03)
        for _ in range(5): train_step(model, batch, optimizer)
        final = float(model(input_ids=batch, labels=batch, use_cache=False).loss.detach())
        self.assertLess(final, initial)
        self.assertTrue(any(not torch.equal(before[n],p) for n,p in selected.items()))
        for name, parameter in model.named_parameters():
            if name not in selected: self.assertTrue(torch.equal(before[name], parameter), name)

    @unittest.skipUnless(os.environ.get('SQLT_TEST_DB') == '1', 'Set SQLT_TEST_DB=1 for temporary-table MariaDB tests')
    def test_native_vector_commit_conflict_and_restore(self):
        # Temporary tables shadow production names only in this connection.
        # Closing it deletes all fixtures, including history; production is untouched.
        c = connect()
        try:
            with c.cursor() as cur:
                cur.execute('CREATE TEMPORARY TABLE moe_model(id INT PRIMARY KEY,revision BIGINT) ENGINE=InnoDB')
                cur.execute('INSERT INTO moe_model VALUES(1,7)')
                cur.execute('CREATE TEMPORARY TABLE moe_parameters(id INT PRIMARY KEY,name VARCHAR(200),shape JSON,dimensions INT,row_count INT,sha256 CHAR(64),role VARCHAR(30)) ENGINE=InnoDB')
                cur.execute('CREATE TEMPORARY TABLE moe_scalars(parameter_id INT,unit_no INT,coefficient DOUBLE,PRIMARY KEY(parameter_id,unit_no)) ENGINE=InnoDB')
                for dim in (512,1536):
                    cur.execute(f'CREATE TEMPORARY TABLE moe_vectors_{dim}(parameter_id INT,unit_no INT,coefficients VECTOR({dim}),PRIMARY KEY(parameter_id,unit_no)) ENGINE=InnoDB')
                cur.execute('CREATE TEMPORARY TABLE moe_events(id BIGINT AUTO_INCREMENT PRIMARY KEY,kind VARCHAR(40),metrics JSON) ENGINE=InnoDB')
                for dim in (0,512,1536):
                    table = f'moe_training_vectors_{dim}' if dim else 'moe_training_scalars'
                    field = f'coefficients VECTOR({dim})' if dim else 'coefficient DOUBLE'
                    cur.execute(f'CREATE TEMPORARY TABLE {table}(run_id BIGINT,parameter_id INT,unit_no INT,{field},PRIMARY KEY(run_id,parameter_id,unit_no)) ENGINE=InnoDB')
                cur.execute('CREATE TEMPORARY TABLE moe_training_runs(id BIGINT AUTO_INCREMENT PRIMARY KEY,base_revision BIGINT,saved_revision BIGINT,metrics JSON,created TIMESTAMP DEFAULT CURRENT_TIMESTAMP) ENGINE=InnoDB')
                cur.execute('CREATE TEMPORARY TABLE moe_training_parameters(run_id BIGINT,parameter_id INT,sha256 CHAR(64),PRIMARY KEY(run_id,parameter_id)) ENGINE=InnoDB')
            metadata = {}
            selected = {}
            for pid,dim in enumerate((0,512,1536), 1):
                name = f'test{dim}'
                weight = torch.arange(dim if dim else 3, dtype=torch.float32)
                if dim: weight = weight.reshape(1,dim)
                sha = hashlib.sha256(weight.numpy().astype('<f4').tobytes()).hexdigest()
                meta = dict(id=pid,name=name,shape=json.dumps(list(weight.shape)),dimensions=dim,
                            row_count=len(weight),sha256=sha,role='core')
                metadata[name] = meta
                selected[name] = torch.nn.Parameter(weight + 0.25)
                with c.cursor() as cur:
                    cur.execute('INSERT INTO moe_parameters(id,name,shape,dimensions,row_count,sha256,role) VALUES(%s,%s,%s,%s,%s,%s,%s)',
                                (pid,name,meta['shape'],dim,len(weight),sha,'core'))
                    if dim: cur.execute(f'INSERT INTO moe_vectors_{dim} VALUES(%s,0,%s)', (pid,weight.numpy().astype('<f4').tobytes()))
                    else: cur.executemany('INSERT INTO moe_scalars VALUES(%s,%s,%s)', [(pid,i,float(x)) for i,x in enumerate(weight)])
            c.commit()
            run = save_weights(c, metadata, selected, 7, {'test':True})
            with c.cursor() as cur:
                cur.execute('SELECT * FROM moe_parameters')
                for meta in cur.fetchall(): self.assertTrue(torch.equal(load_parameter(c, meta), selected[meta['name']]))
            with self.assertRaises(ValueError): save_weights(c, metadata, selected, 7, {})
            damaged = dict(selected)
            damaged['test512'] = torch.nn.Parameter(torch.full_like(selected['test512'], float('nan')))
            with self.assertRaises(ValueError): save_weights(c, metadata, damaged, 8, {})
            with c.cursor() as cur:
                cur.execute('SELECT revision FROM moe_model WHERE id=1')
                self.assertEqual(cur.fetchone()['revision'], 8)
                cur.execute('SELECT * FROM moe_parameters')
                for meta in cur.fetchall():
                    self.assertTrue(torch.equal(load_parameter(c, meta), selected[meta['name']]))
            restore(c, run)
            with c.cursor() as cur:
                cur.execute('SELECT * FROM moe_parameters')
                for meta in cur.fetchall():
                    self.assertEqual(meta['sha256'], metadata[meta['name']]['sha256'])
                    self.assertTrue(torch.equal(load_parameter(c, meta), selected[meta['name']] - 0.25))
                cur.execute('SELECT revision FROM moe_model WHERE id=1')
                self.assertEqual(cur.fetchone()['revision'],9)
            with self.assertRaises(ValueError): restore(c, run)
        finally: c.close()

if __name__ == '__main__': unittest.main()
