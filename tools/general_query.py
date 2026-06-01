import sqlite3
from config import DEMO_CUSTOMER_ID, DB_PATH


def general_query(query: str) -> str:
    """
    Run a read-only SQL query against the Chinook music store database.
    Always include 'CustomerId = {customer_id}' somewhere in the query to scope
    results to the current customer. For customer-specific lookups (invoices,
    purchase history), filter the main WHERE clause by CustomerId. For catalog
    searches (artist, genre, album), put CustomerId in a NOT IN subquery to
    exclude tracks the customer already owns.
    """
    if f"CustomerId = {DEMO_CUSTOMER_ID}" not in query:
        return (
            f"Error: query must filter by CustomerId = {DEMO_CUSTOMER_ID}. "
            "Always scope queries to the current customer."
        )

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    # execute query, format results as pipe-delimited table, handle errors gracefully
    try:
        rows = conn.execute(query).fetchall()
        if not rows:
            return "No results found."
        keys = list(rows[0].keys())
        lines = [" | ".join(keys)]
        lines += [" | ".join(str(row[k]) for k in keys) for row in rows]
        return "\n".join(lines)
    except Exception as e:
        return f"Query error: {e}"
    finally:
        conn.close()