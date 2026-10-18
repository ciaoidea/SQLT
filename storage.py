"""Connection restricted to the Cortex persistence database."""
import json
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import time
import pymysql


@contextmanager
def namespace_write_lock(connection,namespace):
    """Serialize namespace mutations, including first-time operator creation."""
    name='cortex.write.'+namespace
    with connection.cursor() as cursor:
        cursor.execute('SELECT GET_LOCK(%s,5)',(name,))
        if cursor.fetchone()[0]!=1:raise TimeoutError('Memory namespace is busy; no change was made')
    try:yield
    finally:
        with connection.cursor() as cursor:cursor.execute('SELECT RELEASE_LOCK(%s)',(name,))


def retry_transaction(function):
    """Restart known transaction conflicts, never retry an ambiguous commit."""
    @wraps(function)
    def call(*args,**kwargs):
        for attempt in range(5):
            try:
                connection=getattr(args[0],'c',None) if args else None
                if connection is None:return function(*args,**kwargs)
                with namespace_write_lock(connection,args[0].namespace):
                    return function(*args,**kwargs)
            except pymysql.OperationalError as error:
                connection=getattr(args[0],'c',None) if args else None
                if connection is not None:connection.rollback()
                if error.args[0] not in (1020,1205,1213) or attempt==4:raise
                time.sleep(.01*(attempt+1))
    return call


def connect():
    config = json.loads(Path(__file__).with_name('.env.json').read_text())
    if config.get('database') != 'CORTEX':
        raise ValueError('Refusing another database')
    # Vector-index and FK activity must not read a stale REPEATABLE READ view
    # when another namespace commits. This is local to Cortex connections.
    return pymysql.connect(**config,init_command='SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED')
