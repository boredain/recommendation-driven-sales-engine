import sqlite3
from config import DB_PATH

_CUSTOMER_TABLES = {"invoice", "invoiceline", "customer", "employee"}


def catalog_search(query: str) -> str:
    """
    Run a read-only SQL query against catalog tables only (Track, Album, Artist,
    Genre, MediaType, Playlist, PlaylistTrack). No customer filtering required.
    Use this to fetch catalog data such as all available genres, or to search
    tracks by artist or album without needing a customer scope.
    Do NOT use for customer data — use general_query for invoices and purchase history.
    """
    query_lower = query.lower()
    for table in _CUSTOMER_TABLES:
        if table in query_lower:
            return (
                f"Error: catalog_search cannot query customer tables ({table}). "
                "Use general_query for customer data."
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