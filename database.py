"""Portable SQLTensor database connection; no embedded credentials."""
import os
from pathlib import Path
import pymysql

ROOT = Path(__file__).resolve().parent

def settings():
    values = {}
    path = Path(os.environ.get('SQLTENSOR_ENV', ROOT / '.env'))
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                values[key.strip()] = value.strip()
    values.update({k: v for k, v in os.environ.items() if k.startswith('SQLTENSOR_')})
    return values

def connect():
    cfg = settings()
    password = cfg.get('SQLTENSOR_DB_PASSWORD')
    if not password:
        raise RuntimeError('SQLTENSOR_DB_PASSWORD is missing; run ./install.sh first')
    return pymysql.connect(host=cfg.get('SQLTENSOR_DB_HOST', '127.0.0.1'),
        port=int(cfg.get('SQLTENSOR_DB_PORT', '3308')),
        user=cfg.get('SQLTENSOR_DB_USER', 'sqltensor'), password=password,
        database=cfg.get('SQLTENSOR_DB_NAME', 'SQLTensor'), charset='utf8mb4',
        init_command='SET SESSION group_concat_max_len=33554432',
        cursorclass=pymysql.cursors.DictCursor, read_timeout=600, write_timeout=600)
