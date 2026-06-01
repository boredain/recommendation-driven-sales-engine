import sqlite3
from datetime import date
from config import DEMO_CUSTOMER_ID, DB_PATH
from langgraph.types import interrupt


def purchase_track(track_name: str, artist_name: str, price: float) -> str:
    """
    Purchase a track for the current customer. Looks up the track by name and
    artist, then inserts a new Invoice and InvoiceLine into the database.
    Only call this after the customer has indicated which track they want to buy.
    """
    # pause the agent and surface a confirmation dialog to the user before any DB write;
    # agent is frozen until the user responds with "approve" or "reject"
    response = interrupt({
        "message": f'Confirm purchase of "{track_name}" by {artist_name} for ${price:.2f}?',
        "options": ["approve", "reject"],
    })

    # handle both resume formats: LangGraph Studio sends a raw string, API calls may send a dict
    if isinstance(response, dict):
        approved = str(response.get("decision", response.get("approved", ""))).lower() == "approve"
    else:
        approved = str(response).lower().strip() == "approve"

    # if user rejected, exit immediately without opening a DB connection
    if not approved:
        return "Purchase cancelled."

    conn = sqlite3.connect(str(DB_PATH))
    try:
        # resolve TrackId internally via JOIN — the LLM only knows track name and artist name,
        # never raw IDs; parameterized query (?) prevents SQL injection
        track_row = conn.execute(
            """
            SELECT t.TrackId FROM Track t
            JOIN Album al ON t.AlbumId = al.AlbumId
            JOIN Artist ar ON al.ArtistId = ar.ArtistId
            WHERE t.Name = ? AND ar.Name = ?
            """,
            (track_name, artist_name),
        ).fetchone()

        # guard: track/artist combination not found in catalog
        if not track_row:
            return f'Could not find "{track_name}" by {artist_name} in the catalog.'

        track_id = track_row[0]

        # reuse billing address from customer's most recent invoice to avoid re-asking
        billing_row = conn.execute(
            "SELECT BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode "
            "FROM Invoice WHERE CustomerId = ? LIMIT 1",
            (DEMO_CUSTOMER_ID,),
        ).fetchone()

        # guard: no prior invoices found, cannot determine billing info
        if not billing_row:
            return "Could not retrieve billing information for this customer."

        billing_address, billing_city, billing_state, billing_country, billing_postal = billing_row

        # insert Invoice row; capture auto-generated InvoiceId via lastrowid for the line item
        cursor = conn.execute(
            "INSERT INTO Invoice (CustomerId, InvoiceDate, BillingAddress, BillingCity, "
            "BillingState, BillingCountry, BillingPostalCode, Total) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (DEMO_CUSTOMER_ID, date.today().isoformat(), billing_address, billing_city,
             billing_state, billing_country, billing_postal, price),
        )
        invoice_id = cursor.lastrowid

        # insert InvoiceLine row linking the new Invoice to the purchased Track
        conn.execute(
            "INSERT INTO InvoiceLine (InvoiceId, TrackId, UnitPrice, Quantity) VALUES (?, ?, ?, ?)",
            (invoice_id, track_id, price, 1),
        )
        # commit both inserts as one transaction — either both are saved or neither is
        conn.commit()
        return f'"{track_name}" by {artist_name} has been added to your purchases. Enjoy the music!'
    except Exception as e:
        # rollback undoes both inserts — prevents a half-written purchase in the DB
        conn.rollback()
        return f"Purchase failed: {e}"
    finally:
        conn.close()