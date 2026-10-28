"""Explicit local administration, never exposed as a model-callable MCP tool."""
from storage import connect,retry_transaction,namespace_write_lock


@retry_transaction
def reset_memory(namespace, confirmation):
    """Permanently clear one confirmed namespace in a single transaction."""
    if not isinstance(namespace,str) or not namespace.isidentifier() or len(namespace)>32:
        raise ValueError('Invalid namespace')
    if confirmation != f'RESET {namespace}':
        raise ValueError('Reset confirmation does not match the namespace')
    with connect() as connection,namespace_write_lock(connection,namespace):
        connection.begin()
        try:
            with connection.cursor() as cursor:
                cursor.execute('SELECT COUNT(*) FROM episodic_events WHERE namespace=%s',(namespace,))
                episodes=cursor.fetchone()[0]
                from apfc import clear_namespace
                clear_namespace(connection,namespace)
                # Bindings, outcomes and semantic vectors follow their event FK.
                cursor.execute('DELETE FROM episodic_events WHERE namespace=%s',(namespace,))
                cursor.execute('DELETE FROM episodic_operators WHERE namespace=%s',(namespace,))
                operators=cursor.rowcount
                cursor.execute('DELETE FROM episodic_entities WHERE namespace=%s',(namespace,))
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    return dict(namespace=namespace,deleted_episodes=episodes,deleted_operators=operators)
