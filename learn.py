"""Bounded text fine-tuning of SQLT core weights; persistence stays in MariaDB."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from database import connect
from schema import initialize_history
from runtime import DatabaseMoE


def blocks(ids, length):
    return [torch.tensor([ids[i:i + length]], dtype=torch.long)
            for i in range(0, len(ids) - 1, length - 1)
            if len(ids[i:i + length]) >= 2]


def select_parameters(model, last_layers):
    count = len(model.model.layers)
    if not 1 <= last_layers <= count:
        raise ValueError('last-layers is outside the model architecture')
    selected = {}
    for name, parameter in model.named_parameters():
        parts = name.split('.')
        last = name == 'model.norm.weight'
        if len(parts) > 3 and parts[:2] == ['model', 'layers']:
            last = int(parts[2]) >= count - last_layers and (
                '.self_attn.' in name or 'layernorm.' in name)
        parameter.requires_grad_(last)
        if last:
            selected[name] = parameter
    if not selected:
        raise ValueError('No supported attention or normalization weights found')
    return selected



def save_weights(connection, metadata, selected, expected_revision, metrics):
    initialize_history(connection)
    try:
        connection.begin()
        with connection.cursor() as cur:
            cur.execute('SELECT revision FROM moe_model WHERE id=1 FOR UPDATE')
            revision = cur.fetchone()['revision']
            if revision != expected_revision:
                raise ValueError('Model changed during training; restart training on the new revision')
            cur.execute('INSERT INTO moe_training_runs(base_revision,saved_revision,metrics) VALUES(%s,%s,%s)',
                        (revision, revision + 1, json.dumps(metrics)))
            run_id = cur.lastrowid
            for name, parameter in selected.items():
                meta = metadata[name]
                array = parameter.detach().cpu().float().numpy().astype('<f4', copy=False)
                if not bool(torch.isfinite(parameter).all()):
                    raise ValueError('Non-finite trained weights')
                dim = meta['dimensions']
                if dim not in (0, 512, 1536):
                    raise ValueError('Unsupported SQL vector dimension')
                table = f'moe_vectors_{dim}' if dim else 'moe_scalars'
                backup = f'moe_training_vectors_{dim}' if dim else 'moe_training_scalars'
                field = 'coefficients' if dim else 'coefficient'
                cur.execute('INSERT INTO moe_training_parameters VALUES(%s,%s,%s)',
                            (run_id, meta['id'], meta['sha256']))
                cur.execute(f'INSERT INTO {backup} SELECT %s,parameter_id,unit_no,{field} FROM {table} WHERE parameter_id=%s',
                            (run_id, meta['id']))
                for start in range(0, len(array), 64):
                    values = [(array[i].tobytes() if dim else float(array[i]), meta['id'], i)
                              for i in range(start, min(start + 64, len(array)))]
                    cur.executemany(f'UPDATE {table} SET {field}=%s WHERE parameter_id=%s AND unit_no=%s', values)
                cur.execute('UPDATE moe_parameters SET sha256=%s WHERE id=%s',
                            (hashlib.sha256(array.tobytes()).hexdigest(), meta['id']))
            cur.execute('UPDATE moe_model SET revision=%s WHERE id=1', (revision + 1,))
            cur.execute('INSERT INTO moe_events(kind,metrics) VALUES(%s,%s)',
                        ('text_training', json.dumps(dict(metrics, run_id=run_id))))
        connection.commit()
        return run_id
    except BaseException:
        connection.rollback()
        raise


def restore(connection, run_id):
    initialize_history(connection)
    try:
        connection.begin()
        with connection.cursor() as cur:
            cur.execute('SELECT revision FROM moe_model WHERE id=1 FOR UPDATE')
            revision = cur.fetchone()['revision']
            cur.execute('SELECT * FROM moe_training_runs WHERE id=%s', (run_id,))
            run = cur.fetchone()
            if not run or revision != run['saved_revision']:
                raise ValueError('Only the currently applied training run can be restored')
            cur.execute('''SELECT p.*,b.sha256 AS original_sha256 FROM moe_training_parameters b
                JOIN moe_parameters p ON p.id=b.parameter_id WHERE b.run_id=%s''', (run_id,))
            for meta in cur.fetchall():
                dim = meta['dimensions']
                if dim not in (0, 512, 1536):
                    raise ValueError('Unsupported SQL vector dimension')
                table = f'moe_vectors_{dim}' if dim else 'moe_scalars'
                backup = f'moe_training_vectors_{dim}' if dim else 'moe_training_scalars'
                field = 'coefficients' if dim else 'coefficient'
                cur.execute(f'''UPDATE {table} t JOIN {backup} b
                    ON t.parameter_id=b.parameter_id AND t.unit_no=b.unit_no
                    SET t.{field}=b.{field} WHERE b.run_id=%s AND b.parameter_id=%s''', (run_id, meta['id']))
                cur.execute('UPDATE moe_parameters SET sha256=%s WHERE id=%s',
                            (meta['original_sha256'], meta['id']))
            cur.execute('UPDATE moe_model SET revision=%s WHERE id=1', (revision + 1,))
            cur.execute('INSERT INTO moe_events(kind,metrics) VALUES(%s,%s)',
                        ('training_restore', json.dumps({'run_id': run_id, 'revision': revision + 1})))
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def train_step(model, batch, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=batch, labels=batch, use_cache=False, output_router_logits=False).loss
    if not bool(torch.isfinite(loss)):
        raise ValueError('Non-finite loss; nothing saved')
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                  1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', nargs='?', type=Path)
    parser.add_argument('--restore', type=int, metavar='RUN_ID')
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument('--steps', type=int, help='Explicit number of block updates')
    schedule.add_argument('--one-pass', action='store_true', help='Train each training block once (default)')
    parser.add_argument('--sequence-length', type=int, default=64)
    parser.add_argument('--last-layers', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--cache-mib', type=int, default=2048)
    parser.add_argument('--max-bytes', type=int, default=8 * 1024 * 1024)
    parser.add_argument('--max-validation-regression', type=float, default=0.05)
    parser.add_argument('--dry-run', action='store_true', help='Show plan; do not train or write DB')
    args = parser.parse_args()
    if args.restore is not None:
        if args.file or args.dry_run:
            parser.error('--restore cannot be combined with a file or --dry-run')
        c = connect()
        try:
            restore(c, args.restore)
            print('Weights restored. Restart open chats.')
        finally:
            c.close()
        return
    if not args.file:
        parser.error('An input UTF-8 text file is required')
    if ((args.steps is not None and args.steps < 1) or not 4 <= args.sequence_length <= 512 or args.max_bytes < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.max_validation_regression) or args.max_validation_regression < 0):
        parser.error('Invalid training limits')
    with args.file.open('rb') as stream:
        raw = stream.read(args.max_bytes + 1)
    if len(raw) > args.max_bytes:
        parser.error('Input exceeds --max-bytes; split the corpus into smaller files')
    text = raw.decode('utf-8-sig', errors='strict')
    if not text.strip():
        parser.error('Input is empty')
    print('Loading database weights without INT8 for differentiable training...', flush=True)
    runtime = DatabaseMoE(args.cache_mib, quantized_experts=False, device='cpu', db_workers=0)
    try:
        if runtime.model.config.model_type != 'granitemoe':
            raise ValueError('This trainer currently supports the SQLT Granite MoE architecture')
        ids = runtime.tokenizer.encode(text, add_special_tokens=False)
        # Disjoint token ranges: validation never reuses a training target.
        cut = int(len(ids) * 0.9)
        training = blocks(ids[:cut], args.sequence_length)
        validation = blocks(ids[cut:], args.sequence_length)
        if not training or not validation:
            raise ValueError('Text is too short for separate training and validation token ranges')
        steps = len(training) if args.steps is None else args.steps
        selected = select_parameters(runtime.model, args.last_layers)
        metrics = dict(input_sha256=hashlib.sha256(raw).hexdigest(), tokens=len(ids),
                       steps=steps, sequence_length=args.sequence_length,
                       last_layers=args.last_layers, learning_rate=args.learning_rate,
                       trained_parameters=sum(p.numel() for p in selected.values()),
                       train_blocks=len(training), validation_blocks=len(validation))
        print(json.dumps(metrics), flush=True)
        if args.dry_run:
            return
        # eval mode removes dropout, but does not disable autograd. Frozen prefix
        # builds no graph; gradients start at the selected late-layer weights.
        runtime.model.eval()
        def evaluate():
            total = 0.0
            count = 0
            with torch.no_grad():
                for batch in validation:
                    loss = runtime.model(input_ids=batch, labels=batch, use_cache=False,
                                         output_router_logits=False).loss
                    n = batch.shape[1] - 1
                    total += float(loss) * n
                    count += n
            value = total / count
            if not math.isfinite(value):
                raise ValueError('Non-finite validation loss; nothing saved')
            return value
        metrics['validation_before'] = evaluate()
        optimizer = torch.optim.SGD(selected.values(), lr=args.learning_rate)
        for step in range(steps):
            loss = train_step(runtime.model, training[step % len(training)], optimizer)
            print(json.dumps({'step': step + 1, 'training_loss': loss}), flush=True)
        metrics['validation_after'] = evaluate()
        if metrics['validation_after'] > metrics['validation_before'] + args.max_validation_regression:
            raise ValueError('Validation regressed beyond the limit; nothing saved')
        metadata = {r['name']: r for r in runtime.parameters.values()}
        run_id = save_weights(runtime.connection, metadata, selected,
                              runtime.metadata['revision'], metrics)
        print(json.dumps(dict(metrics, saved_run=run_id)), flush=True)
        print('Updated weights are in MariaDB. Restart open chats. Loss is not a comprehension test.')
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
