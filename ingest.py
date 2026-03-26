"""
ingest.py — Job 1 of 3
Polls Polymarket Gamma API every 15 minutes via GitHub Actions cron.
Upserts markets + outcomes, inserts per-outcome snapshots.

V1.1 change: batch all inserts into three executemany calls instead of
per-market commits. Reduces 6,000+ round trips to ~3, cutting runtime
from 20+ minutes to under 2 minutes.

Known V1 limitation:
    Polymarket Gamma API returns volume at the market level, not per-outcome.
    Per-outcome cumulative_volume is approximated as:
        market_volume * outcome_probability
    Replace with CLOB per-token volume in V2 without schema changes.

Required environment variables:
    DATABASE_URL  — Supabase PostgreSQL connection string (pooler URL)
                    format: postgresql://postgres.[ref]:[password]@[host]:6543/postgres
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
REQUEST_TIMEOUT = 30
INTER_PAGE_SLEEP = 0.5


def get_connection():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise EnvironmentError("DATABASE_URL environment variable is not set")
    return psycopg2.connect(url)


def load_config(cur) -> dict:
    cur.execute("SELECT key, value FROM pipeline_config")
    return {row[0]: row[1] for row in cur.fetchall()}


def fetch_markets_page(offset: int, limit: int, volume_floor: float) -> list[dict]:
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


def apply_percentile_filter(markets: list[dict], percentile_floor: float) -> list[dict]:
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


def _parse_json_field(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def parse_outcomes(market: dict) -> list[dict]:
    labels    = _parse_json_field(market.get("outcomes", "[]"))
    prices    = _parse_json_field(market.get("outcomePrices", "[]"))
    token_ids = _parse_json_field(market.get("clobTokenIds", "[]"))

    is_binary = len(labels) == 2
    results = []

    for i, label in enumerate(labels):
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


def get_last_cumulative_volumes(cur, outcome_ids: list[str], chunk_size: int = 1000) -> dict[str, float]:
    """
    Fetch most recent cumulative_volume_usdc per outcome in chunks
    to avoid statement timeouts on large IN/ANY queries.
    Uses a 25-second statement timeout per chunk.
    """
    if not outcome_ids:
        return {}

    results = {}
    chunks = [outcome_ids[i:i + chunk_size] for i in range(0, len(outcome_ids), chunk_size)]
    log.info(f"Fetching prior volumes in {len(chunks)} chunks of up to {chunk_size}")

    for chunk in chunks:
        cur.execute("SET LOCAL statement_timeout = '25s'")
        cur.execute(
            """
            SELECT DISTINCT ON (outcome_id)
                outcome_id,
                cumulative_volume_usdc
            FROM market_snapshots
            WHERE outcome_id = ANY(%s)
            ORDER BY outcome_id, snapshot_at DESC
            """,
            (chunk,),
        )
        for row in cur.fetchall():
            results[row[0]] = float(row[1])

    log.info(f"Prior volumes fetched: {len(results)} outcomes")
    return results


def main() -> None:
    log.info("=== ingest.py starting ===")
    snapshot_at = datetime.now(timezone.utc)

    conn = get_connection()
    cur  = conn.cursor()

    config = load_config(cur)
    volume_floor     = float(config["volume_floor_usdc"])
    percentile_floor = float(config["volume_percentile_floor"])

    log.info(f"Config: hard floor=${volume_floor}, percentile floor={percentile_floor}")

    all_markets = fetch_all_markets(volume_floor)
    log.info(f"Total markets above hard floor: {len(all_markets)}")

    markets = apply_percentile_filter(all_markets, percentile_floor)

    # -- Parse all markets into flat row lists --------------------------------
    market_rows         = []
    outcome_rows        = []
    snapshot_candidates = []
    parse_errors        = 0

    for market in markets:
        market_id = str(market.get("id", ""))
        if not market_id:
            continue

        try:
            market_rows.append((
                market_id,
                market.get("question") or "",
                market.get("category"),
                market.get("subcategory"),
                market.get("endDate"),
                bool(market.get("active", True)),
            ))

            market_volume = float(market.get("volume", 0))
            liquidity     = float(market.get("liquidity", 0))
            outcomes      = parse_outcomes(market)

            if not outcomes:
                continue

            for outcome in outcomes:
                outcome_rows.append((
                    outcome["outcome_id"],
                    market_id,
                    outcome["label"],
                    outcome["is_binary"],
                ))

                prob = outcome["probability"] if outcome["probability"] is not None else 0.5
                outcome_cumulative = market_volume * prob

                snapshot_candidates.append({
                    "outcome_id":        outcome["outcome_id"],
                    "market_id":         market_id,
                    "probability":       outcome["probability"],
                    "cumulative_volume": outcome_cumulative,
                    "liquidity":         liquidity,
                })

        except Exception as exc:
            log.error(f"Error parsing market {market_id}: {exc}", exc_info=True)
            parse_errors += 1
            continue

    log.info(
        f"Parsed: {len(market_rows)} markets, {len(outcome_rows)} outcomes, "
        f"{len(snapshot_candidates)} snapshot candidates, {parse_errors} parse errors"
    )

    # -- Batch upsert markets -------------------------------------------------
    log.info("Upserting markets...")
    psycopg2.extras.execute_batch(
        cur,
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
        market_rows,
        page_size=500,
    )
    conn.commit()
    log.info(f"Markets upserted: {len(market_rows)}")

    # -- Batch upsert outcomes ------------------------------------------------
    log.info("Upserting outcomes...")
    psycopg2.extras.execute_batch(
        cur,
        """
        INSERT INTO market_outcomes (outcome_id, market_id, outcome_label, is_binary)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (outcome_id) DO NOTHING
        """,
        outcome_rows,
        page_size=500,
    )
    conn.commit()
    log.info(f"Outcomes upserted: {len(outcome_rows)}")

    # -- Fetch prior cumulative volumes for period_volume delta ---------------
    log.info("Fetching prior cumulative volumes...")
    all_outcome_ids = [s["outcome_id"] for s in snapshot_candidates]
    prior_volumes   = get_last_cumulative_volumes(cur, all_outcome_ids)
    log.info(f"Prior volumes found for {len(prior_volumes)} outcomes")

    # -- Build snapshot rows --------------------------------------------------
    snapshot_rows = []
    for s in snapshot_candidates:
        oid   = s["outcome_id"]
        cum   = s["cumulative_volume"]
        prior = prior_volumes.get(oid)
        period = max(cum - prior, 0.0) if prior is not None else 0.0

        snapshot_rows.append((
            oid,
            s["market_id"],
            snapshot_at,
            s["probability"],
            round(cum, 2),
            round(period, 2),
            round(s["liquidity"], 2),
        ))

    # -- Batch insert snapshots -----------------------------------------------
    log.info(f"Inserting {len(snapshot_rows)} snapshot rows...")
    psycopg2.extras.execute_batch(
        cur,
        """
        INSERT INTO market_snapshots
            (outcome_id, market_id, snapshot_at, probability,
             cumulative_volume_usdc, period_volume_usdc, liquidity_usdc)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (outcome_id, snapshot_at) DO NOTHING
        """,
        snapshot_rows,
        page_size=500,
    )
    conn.commit()
    log.info(f"Snapshots inserted: {len(snapshot_rows)}")

    log.info(
        f"=== ingest.py complete === "
        f"markets={len(market_rows)} outcomes={len(outcome_rows)} "
        f"snapshots={len(snapshot_rows)} parse_errors={parse_errors}"
    )

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()