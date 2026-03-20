"""
rollup.py — Job 3 of 3
Runs daily at 00:05 UTC via GitHub Actions cron.

Three operations in order:
  1. Aggregate prior day's market_snapshots into market_daily (per outcome)
  2. Delete market_snapshots older than snapshot_ttl_days (7-day TTL)
  3. Mark markets as inactive if they have no snapshots in the last 2 days

Volume note — double-counting risk:
  market_daily stores one row per outcome per day.
  V1 volume is approximated from market-level data (probability-weighted).
  Both Yes and No outcomes of a binary market carry volume derived from
  the same underlying market_volume figure.

  volume_is_market_level = TRUE flags this on every row.
  Dashboard queries must use MAX(daily_volume_usdc) per market_id, NOT SUM,
  when computing total market volume. SUM across outcomes double-counts.

  This flag becomes FALSE when V2 replaces approximated volume with
  per-token CLOB volume — at which point SUM becomes correct.

Required environment variable:
    DATABASE_URL — Supabase PostgreSQL connection string
"""

import logging
import os
from datetime import datetime, timedelta, timezone

import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)


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
# Schema migration — add volume_is_market_level if not present
# Idempotent: safe to run repeatedly
# ---------------------------------------------------------------------------

def ensure_schema(cur) -> None:
    """
    Add volume_is_market_level column to market_daily if it doesn't exist.
    This handles deployments where the original schema was created before
    this column was added.
    """
    cur.execute(
        """
        ALTER TABLE market_daily
        ADD COLUMN IF NOT EXISTS volume_is_market_level BOOLEAN DEFAULT TRUE
        """
    )
    log.info("Schema check: volume_is_market_level column confirmed")


# ---------------------------------------------------------------------------
# Operation 1: Aggregate snapshots into market_daily
# ---------------------------------------------------------------------------

def rollup_prior_day(cur, target_date) -> int:
    """
    Aggregate all snapshots for target_date into market_daily.
    One row per outcome per day.
    Upserts so re-running for the same date is safe.

    Returns number of outcome-day rows written.
    """
    cur.execute(
        """
        INSERT INTO market_daily (
            outcome_id,
            market_id,
            date,
            open_probability,
            close_probability,
            high_probability,
            low_probability,
            delta_probability,
            daily_volume_usdc,
            avg_liquidity_usdc,
            snapshot_count,
            volume_is_market_level
        )
        SELECT
            ms.outcome_id,
            ms.market_id,
            DATE(ms.snapshot_at)                                    AS date,

            -- Open: earliest snapshot probability of the day
            (ARRAY_AGG(ms.probability ORDER BY ms.snapshot_at ASC))[1]
                                                                    AS open_probability,
            -- Close: latest snapshot probability of the day
            (ARRAY_AGG(ms.probability ORDER BY ms.snapshot_at DESC))[1]
                                                                    AS close_probability,

            MAX(ms.probability)                                     AS high_probability,
            MIN(ms.probability)                                     AS low_probability,

            -- Delta: close minus open
            (ARRAY_AGG(ms.probability ORDER BY ms.snapshot_at DESC))[1]
            - (ARRAY_AGG(ms.probability ORDER BY ms.snapshot_at ASC))[1]
                                                                    AS delta_probability,

            -- Sum of period volumes = total outcome volume for the day
            -- NOTE: See module docstring re: double-counting on binary markets
            SUM(ms.period_volume_usdc)                              AS daily_volume_usdc,

            AVG(ms.liquidity_usdc)                                  AS avg_liquidity_usdc,
            COUNT(*)                                                AS snapshot_count,

            -- V1: always TRUE until CLOB per-token volume is implemented
            TRUE                                                    AS volume_is_market_level

        FROM market_snapshots ms
        WHERE DATE(ms.snapshot_at AT TIME ZONE 'UTC') = %s
          AND ms.period_volume_usdc IS NOT NULL
        GROUP BY ms.outcome_id, ms.market_id, DATE(ms.snapshot_at)

        ON CONFLICT (outcome_id, date) DO UPDATE SET
            open_probability      = EXCLUDED.open_probability,
            close_probability     = EXCLUDED.close_probability,
            high_probability      = EXCLUDED.high_probability,
            low_probability       = EXCLUDED.low_probability,
            delta_probability     = EXCLUDED.delta_probability,
            daily_volume_usdc     = EXCLUDED.daily_volume_usdc,
            avg_liquidity_usdc    = EXCLUDED.avg_liquidity_usdc,
            snapshot_count        = EXCLUDED.snapshot_count,
            volume_is_market_level = EXCLUDED.volume_is_market_level
        """,
        (target_date,),
    )

    rows_written = cur.rowcount
    log.info(f"Rollup: {rows_written} outcome-day rows written for {target_date}")
    return rows_written


# ---------------------------------------------------------------------------
# Operation 2: TTL purge of market_snapshots
# ---------------------------------------------------------------------------

def purge_old_snapshots(cur, cutoff: datetime) -> int:
    """
    Delete market_snapshots older than the TTL cutoff.
    Uses the idx_snapshots_ttl index on snapshot_at for performance.
    Returns number of rows deleted.
    """
    cur.execute(
        "DELETE FROM market_snapshots WHERE snapshot_at < %s",
        (cutoff,),
    )
    deleted = cur.rowcount
    log.info(f"TTL purge: {deleted} snapshot rows deleted (older than {cutoff.date()})")
    return deleted


# ---------------------------------------------------------------------------
# Operation 3: Deactivate stale markets
# ---------------------------------------------------------------------------

def deactivate_stale_markets(cur, stale_cutoff: datetime) -> int:
    """
    Mark markets inactive if their most recent snapshot predates stale_cutoff.
    Stale = no ingest activity in the last 2 days.
    This catches markets that resolved, were delisted, or fell below the
    volume floor and stopped being ingested.

    Uses a subquery against market_snapshots rather than an API call —
    cheaper and consistent with what the pipeline actually observed.
    """
    cur.execute(
        """
        UPDATE markets
        SET is_active = FALSE
        WHERE is_active = TRUE
          AND market_id NOT IN (
              SELECT DISTINCT market_id
              FROM market_snapshots
              WHERE snapshot_at >= %s
          )
        """,
        (stale_cutoff,),
    )
    deactivated = cur.rowcount
    log.info(f"Deactivated {deactivated} stale markets (no snapshots since {stale_cutoff.date()})")
    return deactivated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== rollup.py starting ===")
    now_utc = datetime.now(timezone.utc)

    conn = get_connection()
    cur  = conn.cursor()

    config = load_config(cur)
    ttl_days = int(config["snapshot_ttl_days"])     # default 7

    # Target date for rollup = yesterday UTC
    target_date = (now_utc - timedelta(days=1)).date()

    # TTL cutoff = now minus retention window
    ttl_cutoff = now_utc - timedelta(days=ttl_days)

    # Stale market cutoff = 2 days ago (lenient — handles weekend/holiday gaps)
    stale_cutoff = now_utc - timedelta(days=2)

    log.info(
        f"Target rollup date: {target_date} | "
        f"TTL cutoff: {ttl_cutoff.date()} | "
        f"Stale market cutoff: {stale_cutoff.date()}"
    )

    # -- Step 1: Ensure schema is current -----------------------------------
    try:
        ensure_schema(cur)
        conn.commit()
    except Exception as exc:
        log.error(f"Schema migration failed: {exc}", exc_info=True)
        conn.rollback()
        raise   # fatal — don't proceed if schema is uncertain

    # -- Step 2: Rollup prior day -------------------------------------------
    try:
        rows = rollup_prior_day(cur, target_date)
        conn.commit()
        log.info(f"Rollup committed: {rows} rows")
    except Exception as exc:
        log.error(f"Rollup failed for {target_date}: {exc}", exc_info=True)
        conn.rollback()
        # Non-fatal: log and continue to purge
        # Rollup can be re-run manually via workflow_dispatch

    # -- Step 3: TTL purge -------------------------------------------------
    # Runs AFTER rollup commit so we never purge data that wasn't yet aggregated
    try:
        deleted = purge_old_snapshots(cur, ttl_cutoff)
        conn.commit()
        log.info(f"TTL purge committed: {deleted} rows deleted")
    except Exception as exc:
        log.error(f"TTL purge failed: {exc}", exc_info=True)
        conn.rollback()

    # -- Step 4: Deactivate stale markets ----------------------------------
    try:
        deactivated = deactivate_stale_markets(cur, stale_cutoff)
        conn.commit()
        log.info(f"Stale market deactivation committed: {deactivated} markets")
    except Exception as exc:
        log.error(f"Stale market deactivation failed: {exc}", exc_info=True)
        conn.rollback()

    log.info("=== rollup.py complete ===")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
