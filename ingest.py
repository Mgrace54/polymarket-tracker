"""
ingest.py — Job 1 of 3
Polls Polymarket Gamma API every 45 minutes via GitHub Actions cron.
Upserts markets + outcomes, inserts per-outcome snapshots.

Known V1 limitation:
    Polymarket Gamma API returns volume at the market level, not per-outcome.
    Per-outcome cumulative_volume is approximated as:
        market_volume * outcome_probability
    This affects ΔP/ΔV precision in compute_signals.py.
    Replace with CLOB API per-token volume in V2 without schema changes.

Required environment variables:
    DATABASE_URL  — Supabase PostgreSQL connection string
                    format: postgresql://user:password@host:5432/postgres
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import psycopg2
import psycopg2.extras
import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
REQUEST_TIMEOUT = 30
INTER_PAGE_SLEEP = 0.5   # seconds between paginated requests — stay polite


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_connection():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise EnvironmentError("DATABASE_URL environment variable is not set")
    return psycopg2.connect(url)


def load_config(cur) -> dict:
    cur.execute("SELECT key, value FROM pipeline_config")
    return {row[0]: row[1] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Polymarket API
# ---------------------------------------------------------------------------

def fetch_markets_page(offset: int, limit: int, volume_floor: float) -> list[dict]:
    """Fetch one page of active markets from the Gamma API."""
    resp = requests.get(
        f"{GAMMA_API_BASE}/markets",
        params={
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
            "volume_num_min": volume_floor,
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_all_markets(volume_floor: float) -> list[dict]:
    """
    Page through the Gamma API and return all active markets
    above the hard volume floor.
    """
    markets: list[dict] = []
    limit = 100
    offset = 0

    while True:
        log.info(f"Fetching page offset={offset}")
        batch = fetch_markets_page(offset, limit, volume_floor)

        if not batch:
            break

        markets.extend(batch)
        log.info(f"  Got {len(batch)} markets (total so far: {len(markets)})")

        if len(batch) < limit:
            break

        offset += limit
        time.sleep(INTER_PAGE_SLEEP)

    return markets


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_percentile_filter(markets: list[dict], percentile_floor: float) -> list[dict]:
    """
    Drop markets below the Nth percentile of fetched universe by volume.
    percentile_floor is expressed as a decimal (e.g., 0.20 = 20th percentile).
    """
    if not markets:
        return markets

    volumes = [float(m.get("volume", 0)) for m in markets]
    threshold = float(np.percentile(volumes, percentile_floor * 100))
    log.info(
        f"Percentile filter: {percentile_floor * 100:.0f}th percentile "
        f"= ${threshold:,.2f} USDC"
    )
    filtered = [m for m in markets if float(m.get("volume", 0)) >= threshold]
    log.info(f"Markets after percentile filter: {len(filtered)} (removed {len(markets) - len(filtered)})")
    return filtered


# ---------------------------------------------------------------------------
# Outcome parsing
# ---------------------------------------------------------------------------

def _parse_json_field(value) -> list:
    """Gamma API sometimes returns JSON arrays as strings, sometimes as lists."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def parse_outcomes(market: dict) -> list[dict]:
    """
    Extract per-outcome data from a Gamma API market record.

    Returns list of dicts:
        outcome_id  — Polymarket CLOB token ID (falls back to market_id_index)
        label       — outcome label string ("Yes", "No", "Biden", etc.)
        probability — current price / implied probability (float 0–1)
        is_binary   — True if market has exactly 2 outcomes
    """
    labels     = _parse_json_field(market.get("outcomes", "[]"))
    prices     = _parse_json_field(market.get("outcomePrices", "[]"))
    token_ids  = _parse_json_field(market.get("clobTokenIds", "[]"))

    is_binary = len(labels) == 2
    results = []

    for i, label in enumerate(labels):
        # Prefer the CLOB token ID; fall back to a deterministic synthetic ID
        outcome_id = (
            str(token_ids[i])
            if i < len(token_ids) and token_ids[i]
            else f"{market['id']}__outcome_{i}"
        )

        try:
            probability = float(prices[i]) if i < len(prices) else None
        except (ValueError, TypeError):
            probability = None

        results.append({
            "outcome_id": outcome_id,
            "label":      label,
            "probability": probability,
            "is_binary":  is_binary,
        })

    return results


# ---------------------------------------------------------------------------
# Upserts & inserts
# ---------------------------------------------------------------------------

def upsert_market(cur, market: dict) -> None:
    cur.execute(
        """
        INSERT INTO markets
            (market_id, question, category, subcategory, resolution_date, is_active)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (market_id) DO UPDATE SET
            question        = EXCLUDED.question,
            category        = EXCLUDED.category,
            subcategory     = EXCLUDED.subcategory,
            resolution_date = EXCLUDED.resolution_date,
            is_active       = EXCLUDED.is_active
        """,
        (
            str(market["id"]),
            market.get("question") or "",
            market.get("category"),
            market.get("subcategory"),
            market.get("endDate"),       # ISO string or None
            bool(market.get("active", True)),
        ),
    )


def upsert_outcome(cur, outcome_id: str, market_id: str, label: str, is_binary: bool) -> None:
    cur.execute(
        """
        INSERT INTO market_outcomes (outcome_id, market_id, outcome_label, is_binary)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (outcome_id) DO NOTHING
        """,
        (outcome_id, market_id, label, is_binary),
    )


def get_last_snapshot(cur, outcome_id: str) -> Optional[dict]:
    cur.execute(
        """
        SELECT cumulative_volume_usdc
        FROM market_snapshots
        WHERE outcome_id = %s
        ORDER BY snapshot_at DESC
        LIMIT 1
        """,
        (outcome_id,),
    )
    row = cur.fetchone()
    return {"cumulative_volume_usdc": float(row[0])} if row else None


def insert_snapshot(
    cur,
    outcome_id: str,
    market_id: str,
    snapshot_at: datetime,
    probability: Optional[float],
    cumulative_volume: float,
    period_volume: float,
    liquidity: float,
) -> None:
    cur.execute(
        """
        INSERT INTO market_snapshots
            (outcome_id, market_id, snapshot_at, probability,
             cumulative_volume_usdc, period_volume_usdc, liquidity_usdc)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (outcome_id, snapshot_at) DO NOTHING
        """,
        (
            outcome_id,
            market_id,
            snapshot_at,
            probability,
            round(cumulative_volume, 2),
            round(max(period_volume, 0.0), 2),   # clamp negatives from API quirks
            round(liquidity, 2),
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== ingest.py starting ===")
    snapshot_at = datetime.now(timezone.utc)

    conn = get_connection()
    cur  = conn.cursor()

    # -- Load runtime config from DB ------------------------------------------
    config = load_config(cur)
    volume_floor      = float(config["volume_floor_usdc"])
    percentile_floor  = float(config["volume_percentile_floor"])

    log.info(f"Config: hard floor=${volume_floor}, percentile floor={percentile_floor}")

    # -- Fetch & filter -------------------------------------------------------
    all_markets = fetch_all_markets(volume_floor)
    log.info(f"Total markets above hard floor: {len(all_markets)}")

    markets = apply_percentile_filter(all_markets, percentile_floor)

    # -- Process each market --------------------------------------------------
    markets_processed  = 0
    outcomes_processed = 0
    snapshots_inserted = 0
    errors             = 0

    for market in markets:
        market_id = str(market.get("id", ""))

        if not market_id:
            log.warning("Market record missing id — skipping")
            continue

        try:
            upsert_market(cur, market)

            market_volume = float(market.get("volume", 0))
            liquidity     = float(market.get("liquidity", 0))
            outcomes      = parse_outcomes(market)

            if not outcomes:
                log.warning(f"No outcomes parsed for market {market_id} — skipping")
                conn.commit()
                continue

            for outcome in outcomes:
                upsert_outcome(
                    cur,
                    outcome["outcome_id"],
                    market_id,
                    outcome["label"],
                    outcome["is_binary"],
                )

                # V1 approximation: distribute market volume by outcome probability.
                # Probability-weighted because in a binary market, $100 bet on Yes
                # at 0.60 represents ~$60 of yes-side volume.
                # Replace with CLOB per-token volume in V2.
                prob = outcome["probability"] if outcome["probability"] is not None else 0.5
                outcome_cumulative = market_volume * prob

                last = get_last_snapshot(cur, outcome["outcome_id"])
                period_volume = (
                    outcome_cumulative - last["cumulative_volume_usdc"]
                    if last
                    else 0.0
                )

                insert_snapshot(
                    cur,
                    outcome_id=outcome["outcome_id"],
                    market_id=market_id,
                    snapshot_at=snapshot_at,
                    probability=outcome["probability"],
                    cumulative_volume=outcome_cumulative,
                    period_volume=period_volume,
                    liquidity=liquidity,
                )
                outcomes_processed += 1
                snapshots_inserted += 1

            conn.commit()
            markets_processed += 1

        except Exception as exc:
            log.error(f"Error processing market {market_id}: {exc}", exc_info=True)
            conn.rollback()
            errors += 1
            continue

    log.info(
        f"=== ingest.py complete ===

    # -- Summary --------------------------------------------------------------
    log.info(
        f"=== ingest.py complete === "
        f"markets={markets_processed} outcomes={outcomes_processed} "
        f"snapshots={snapshots_inserted} errors={errors}"
    )

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
