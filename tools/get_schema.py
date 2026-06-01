import sqlite3
from config import DB_PATH


def get_schema(table_name: str = None) -> str:
    """
    Get database schema information.
    Call with no arguments to list all available tables.
    Call with a table name to get column details and foreign key relationships for that table.
    Always call this before writing any SQL query.
    """
    conn = sqlite3.connect(str(DB_PATH))
    try:
        # mode 1: no table name provided — return list of all tables
        if table_name is None:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            return "\n".join(t[0] for t in tables)
        # mode 2: table name provided — return columns and foreign keys for that table
        cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        fks = conn.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()
        lines = [f"Table: {table_name}", "Columns:"]
        for col in cols:
            pk = " (PK)" if col[5] else ""
            notnull = " NOT NULL" if col[3] else ""
            lines.append(f"  {col[1]} {col[2]}{pk}{notnull}")
        if fks:
            lines.append("Foreign Keys:")
            for fk in fks:
                lines.append(f"  {fk[3]} -> {fk[2]}.{fk[4]}")
        return "\n".join(lines)
    finally:
        conn.close()