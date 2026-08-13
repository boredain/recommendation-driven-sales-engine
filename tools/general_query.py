import re
import sqlite3
from config import DEMO_CUSTOMER_ID, DB_PATH

_ARTIST_SCOPE = """ar.ArtistId IN (
        SELECT al2.ArtistId FROM Album al2
        LEFT JOIN Artist ar2 ON al2.ArtistId = ar2.ArtistId
        LEFT JOIN Track t2 ON t2.AlbumId = al2.AlbumId
        WHERE ar2.Name LIKE ? OR al2.Title LIKE ? OR t2.Name LIKE ?
    )"""

_GENRE_SCOPE = """t.GenreId IN (
        SELECT t2.GenreId FROM Track t2
        LEFT JOIN Album al2 ON t2.AlbumId = al2.AlbumId
        LEFT JOIN Artist ar2 ON al2.ArtistId = ar2.ArtistId
        LEFT JOIN Genre g2 ON t2.GenreId = g2.GenreId
        WHERE ar2.Name LIKE ? OR al2.Title LIKE ? OR t2.Name LIKE ? OR g2.Name LIKE ?
    )"""

_WIDENED_SCOPES = (("artist", _ARTIST_SCOPE, 3), ("genre", _GENRE_SCOPE, 4))

_WIDENED_UNOWNED_SQL = """
    SELECT t.Name AS Track, ar.Name AS Artist, g.Name AS Genre, t.UnitPrice AS Price
    FROM Track t
    LEFT JOIN Album al ON t.AlbumId = al.AlbumId
    LEFT JOIN Artist ar ON al.ArtistId = ar.ArtistId
    LEFT JOIN Genre g ON t.GenreId = g.GenreId
    WHERE {scope}
      AND t.TrackId NOT IN (
        SELECT il.TrackId FROM InvoiceLine il
        JOIN Invoice i ON il.InvoiceId = i.InvoiceId
        WHERE i.CustomerId = {customer_id}
      )
    ORDER BY t.Name
    LIMIT 5
"""


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
            # a catalog-availability search that finds nothing must never dead-end the
            # customer, so the widening retries and similar_music handoff happen here
            # rather than being left to the model
            if _is_unowned_catalog_query(query):
                return _unowned_fallback(conn, query)
            return "No results found."
        return _format_rows(rows)
    except Exception as e:
        return f"Query error: {e}"
    finally:
        conn.close()


def _format_rows(rows) -> str:
    keys = list(rows[0].keys())
    lines = [" | ".join(keys)]
    lines += [" | ".join(str(row[k]) for k in keys) for row in rows]
    return "\n".join(lines)


def _is_unowned_catalog_query(query: str) -> bool:
    """True when the query looks for catalog tracks the customer does not already own."""
    lowered = query.lower()
    return "not in" in lowered and "invoiceline" in lowered


def _search_terms(query: str) -> list:
    """Pull the searched-for names (artist, album, track, genre) out of the SQL literals."""
    terms = []
    for literal in re.findall(r"'([^']*)'", query):
        term = literal.strip().strip("%").strip()
        if term and not term.isdigit() and term not in terms:
            terms.append(term)
    return terms


def _widen_unowned(conn, terms):
    """Retry the unowned lookup scoped to the artist, then the genre, of each search term."""
    for label, scope, placeholders in _WIDENED_SCOPES:
        for term in terms:
            sql = _WIDENED_UNOWNED_SQL.format(scope=scope, customer_id=DEMO_CUSTOMER_ID)
            rows = conn.execute(sql, [f"%{term}%"] * placeholders).fetchall()
            if rows:
                return label, term, rows
    return None


def _unowned_fallback(conn, query: str) -> str:
    terms = _search_terms(query)
    widened = _widen_unowned(conn, terms) if terms else None
    if widened:
        label, term, rows = widened
        return (
            "No unowned tracks matched that exact request, so it was widened to the "
            f'{label} of "{term}". Present these as "Available to Purchase" with their '
            "prices in this same reply, and do not tell the customer there is nothing "
            "to purchase:\n" + _format_rows(rows)
        )

    from subagents.similar_music import run_similar_music

    output = run_similar_music(" ".join(terms) if terms else "this request")
    if not output:
        return "No results found."
    return (
        "No unowned tracks matched this request at the artist or genre level, so the "
        "similar_music subagent was run for you. Present its output below to the "
        "customer exactly as-is, without rewriting or adding commentary:\n\n" + output
    )
