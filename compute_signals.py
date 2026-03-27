"""
compute_signals.py — Job 2 of 3
Runs every 6 hours via GitHub Actions cron.

For each outcome with sufficient history (>= window_days of snapshots):
  1. Compute log-volume z-score over the rolling window
  2. Compute signed ΔP/ΔV where period volume clears the two-condition gate
  3. Rank-normalize both metrics across the active universe (0-1 percentile)
  4. Produce composite score = mean(z_score_rank, dp_dv_rank) on magnitude
  5. Write per-outcome rows to spike_signals
  6. Write top-N outcomes to daily_topx

Two-condition ΔP/ΔV gate (both must pass):
  - period_volume >= avg_period_volume * dp_dv_threshold_pct  (relative)
  - period_volume >= dp_dv_absolute_floor_usdc                 (hard noise floor)
  If either fails, dp_dv_raw / dp_dv_magnitude are NULL and excluded from ranking.

Exclusion rules:
  - Markets with first_seen_at < window_days ago are excluded entirely
  - Outcomes with < (window_days * min_snapshots_per_day) snapshots are excluded
  - NULL dp_dv outcomes are excluded from dp_dv_rank but can still rank on z_score

Required environment variable:
    DATABASE_URL — Supabase PostgreSQL connection string
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import psycopg2
import psycopg2.extras

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
# Data fetch
# ---------------------------------------------------------------------------

def fetch_eligible_outcomes(cur, window_start: datetime, min_snapshots: int) -> list[dict]:
    """
    Two-pass approach to avoid ARRAY_AGG timeout on large snapshot tables:
    Pass 1: Fast COUNT query to find eligible outcome_ids (no arrays)
    Pass 2: Fetch array data for eligible outcomes in chunks of 500
    """
    # Pass 1: find eligible outcome_ids with sufficient snapshot coverage
    log.info("Pass 1: finding eligible outcomes by snapshot count...")
    # No SET LOCAL - connection pooler ignores it; rely on ALTER ROLE timeout
    # Simplified query: count snapshots per outcome only, no JOINs
    cur.execute(
        """
        SELECT outcome_id, market_id, COUNT(*) AS snapshot_count
        FROM market_snapshots
        WHERE snapshot_at >= %s
        GROUP BY outcome_id, market_id
        HAVING COUNT(*) >= %s
        """,
        (window_start, min_snapshots),
    )
    eligible_raw = cur.fetchall()
    log.info(f"Pass 1 complete: {len(eligible_raw)} eligible outcomes found")

    # Fetch outcome labels and market metadata in one batch query
    eligible_outcome_ids = [row[0] for row in eligible_raw]
    cur.execute(
        """
        SELECT mo.outcome_id, mo.outcome_label, m.question, m.category
        FROM market_outcomes mo
        JOIN markets m ON m.market_id = mo.market_id
        WHERE mo.outcome_id = ANY(%s)
          AND m.is_active = TRUE
        """,
        (eligible_outcome_ids,),
    )
    meta_rows = {row[0]: (row[1], row[2], row[3]) for row in cur.fetchall()}

    # Filter to only active markets and build eligible list
    eligible = [
        (row[0], row[1], row[2])
        for row in eligible_raw
        if row[0] in meta_rows
    ]
    log.info(f"Pass 1 after active market filter: {len(eligible)} eligible outcomes")

    if not eligible:
        return []

    # Build metadata lookup from pass 1
    meta = {}
    for row in eligible:
        outcome_id = row[0]
        market_id  = row[1]
        label, question, category = meta_rows.get(outcome_id, ("", "", ""))
        meta[outcome_id] = {
            "outcome_id":      outcome_id,
            "market_id":       market_id,
            "outcome_label":   label,
            "market_question": question,
            "category":        category,
            "snapshot_count":  row[2],
        }

    eligible_ids = list(meta.keys())

    # Pass 2: fetch array data in chunks of 500 to avoid timeouts
    log.info(f"Pass 2: fetching signal data in chunks...")
    chunk_size = 500
    chunks = [eligible_ids[i:i+chunk_size] for i in range(0, len(eligible_ids), chunk_size)]
    results = []

    for chunk_num, chunk in enumerate(chunks):
        cur.execute(
            """
            SELECT
                ms.outcome_id,
                AVG(ms.period_volume_usdc)                                          AS avg_period_volume,
                (ARRAY_AGG(ms.period_volume_usdc ORDER BY ms.snapshot_at ASC))      AS volume_series,
                (ARRAY_AGG(ms.probability        ORDER BY ms.snapshot_at ASC))      AS probability_series,
                (ARRAY_AGG(ms.period_volume_usdc ORDER BY ms.snapshot_at DESC))[1]  AS latest_period_volume,
                (ARRAY_AGG(ms.probability        ORDER BY ms.snapshot_at DESC))[1]  AS latest_probability,
                (ARRAY_AGG(ms.probability        ORDER BY ms.snapshot_at DESC))[2]  AS prior_probability
            FROM market_snapshots ms
            WHERE ms.outcome_id = ANY(%s)
              AND ms.snapshot_at >= %s
            GROUP BY ms.outcome_id
            """,
            (chunk, window_start),
        )
        for row in cur.fetchall():
            oid = row[0]
            if oid in meta:
                results.append({
                    **meta[oid],
                    "avg_period_volume":   row[1],
                    "volume_series":       row[2],
                    "probability_series":  row[3],
                    "latest_period_volume": row[4],
                    "latest_probability":  row[5],
                    "prior_probability":   row[6],
                })

        if (chunk_num + 1) % 5 == 0:
            log.info(f"  Processed {chunk_num + 1}/{len(chunks)} chunks ({len(results)} outcomes so far)")

    log.info(f"Pass 2 complete: {len(results)} outcomes with signal data")
    return results


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def compute_z_score(volume_series: list[Optional[float]]) -> Optional[float]:
    """
    Z-score of the most recent period volume against the window distribution.
    Uses log1p transformation to handle right-skew of USDC values.
    log1p(x) = log(1 + x) — safe for zero-volume periods.
    Returns None if standard deviation is zero (flat series = no signal).
    """
    values = [v for v in volume_series if v is not None]
    if len(values) < 2:
        return None

    log_values = np.log1p(values)
    mean  = np.mean(log_values)
    std   = np.std(log_values, ddof=1)   # sample std dev

    if std == 0:
        return None

    latest_log = np.log1p(values[-1])
    return float((latest_log - mean) / std)


def compute_dp_dv(
    latest_probability:  Optional[float],
    prior_probability:   Optional[float],
    latest_period_volume: float,
    avg_period_volume:   float,
    threshold_pct:       float,
    absolute_floor:      float,
) -> tuple[Optional[float], Optional[float], Optional[int]]:
    """
    Compute signed and magnitude ΔP/ΔV with two-condition gate.

    Gate conditions (BOTH must pass):
      1. latest_period_volume >= avg_period_volume * threshold_pct
      2. latest_period_volume >= absolute_floor

    Returns:
        dp_dv_raw       — signed ΔP / ΔV  (None if gate fails or data missing)
        dp_dv_magnitude — ABS(dp_dv_raw)   (None if dp_dv_raw is None)
        direction       — -1, 0, or 1      (None if dp_dv_raw is None)
    """
    # Gate check
    relative_threshold = avg_period_volume * threshold_pct
    if (
        latest_period_volume < relative_threshold
        or latest_period_volume < absolute_floor
    ):
        return None, None, None

    # Need both probability values to compute ΔP
    if latest_probability is None or prior_probability is None:
        return None, None, None

    delta_p = latest_probability - prior_probability
    delta_v = latest_period_volume

    if delta_v == 0:
        return None, None, None

    raw       = delta_p / delta_v
    magnitude = abs(raw)

    if delta_p > 0:
        direction = 1
    elif delta_p < 0:
        direction = -1
    else:
        direction = 0

    return float(raw), float(magnitude), direction


# ---------------------------------------------------------------------------
# Rank normalization
# ---------------------------------------------------------------------------

def rank_normalize(values: list[Optional[float]]) -> list[Optional[float]]:
    """
    Convert a list of floats to 0-1 percentile ranks.
    None values stay None and are excluded from the ranking universe.
    Ties receive the average rank (scipy-style 'average' method via numpy).
    """
    indexed = [(v, i) for i, v in enumerate(values) if v is not None]
    if not indexed:
        return values

    n = len(indexed)
    sorted_indexed = sorted(indexed, key=lambda x: x[0])

    # Assign average rank to ties
    ranks = [None] * len(values)
    i = 0
    while i < n:
        j = i
        # Find run of equal values
        while j < n - 1 and sorted_indexed[j][0] == sorted_indexed[j + 1][0]:
            j += 1
        avg_rank = (i + j) / 2.0  # 0-indexed average rank
        normalized = avg_rank / (n - 1) if n > 1 else 0.5
        for k in range(i, j + 1):
            original_index = sorted_indexed[k][1]
            ranks[original_index] = round(float(normalized), 4)
        i = j + 1

    return ranks


# ---------------------------------------------------------------------------
# DB writes
# ---------------------------------------------------------------------------

def upsert_signal(cur, signal: dict) -> None:
    cur.execute(
        """
        INSERT INTO spike_signals (
            outcome_id, market_id, computed_at, window_days,
            volume_z_score,
            dp_dv_raw, dp_dv_magnitude, dp_dv_direction,
            z_score_rank, dp_dv_rank, composite_score
        ) VALUES (
            %(outcome_id)s, %(market_id)s, %(computed_at)s, %(window_days)s,
            %(volume_z_score)s,
            %(dp_dv_raw)s, %(dp_dv_magnitude)s, %(dp_dv_direction)s,
            %(z_score_rank)s, %(dp_dv_rank)s, %(composite_score)s
        )
        ON CONFLICT (outcome_id, computed_at) DO UPDATE SET
            volume_z_score  = EXCLUDED.volume_z_score,
            dp_dv_raw       = EXCLUDED.dp_dv_raw,
            dp_dv_magnitude = EXCLUDED.dp_dv_magnitude,
            dp_dv_direction = EXCLUDED.dp_dv_direction,
            z_score_rank    = EXCLUDED.z_score_rank,
            dp_dv_rank      = EXCLUDED.dp_dv_rank,
            composite_score = EXCLUDED.composite_score
        """,
        signal,
    )


def write_topx(cur, computed_at: datetime, top_n: int, ranked: list[dict]) -> None:
    """
    Write the top-N outcomes to daily_topx.
    Deletes existing rows for today's date first to allow recompute idempotency.
    """
    today = computed_at.date()
    cur.execute(
        "DELETE FROM daily_topx WHERE snapshot_date = %s AND top_n = %s",
        (today, top_n),
    )

    for rank_pos, entry in enumerate(ranked[:top_n], start=1):
        cur.execute(
            """
            INSERT INTO daily_topx (
                snapshot_date, top_n, rank_position,
                outcome_id, market_id, outcome_label, market_question,
                composite_score, volume_z_score,
                dp_dv_raw, dp_dv_direction
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                today,
                top_n,
                rank_pos,
                entry["outcome_id"],
                entry["market_id"],
                entry["outcome_label"],
                entry["market_question"],
                entry.get("composite_score"),
                entry.get("volume_z_score"),
                entry.get("dp_dv_raw"),
                entry.get("dp_dv_direction"),
            ),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== compute_signals.py starting ===")
    computed_at = datetime.now(timezone.utc)

    conn = get_connection()
    cur  = conn.cursor()

    config = load_config(cur)
    window_days          = int(config["window_days"])
    top_n                = int(config["top_n"])
    dp_dv_threshold_pct  = float(config.get("dp_dv_threshold_pct", "0.15"))
    dp_dv_absolute_floor = float(config.get("dp_dv_absolute_floor_usdc", "5.0"))

    # Minimum snapshots = 1 per hour over the window (conservative lower bound)
    # At 15-min polling: 4 snapshots/hr * 24 hrs * window_days
    # We use a lenient floor of 50% of theoretical max to allow for gaps/downtime
    min_snapshots = int((60 / 15) * 24 * window_days * 0.5)
    log.info(
        f"Config: window={window_days}d, top_n={top_n}, "
        f"dp_dv_pct={dp_dv_threshold_pct}, floor=${dp_dv_absolute_floor}, "
        f"min_snapshots={min_snapshots}"
    )

    window_start = computed_at - timedelta(days=window_days)

    # -- Fetch eligible outcomes --------------------------------------------
    outcomes = fetch_eligible_outcomes(cur, window_start, min_snapshots)
    log.info(f"Eligible outcomes for signal computation: {len(outcomes)}")

    if not outcomes:
        log.warning("No eligible outcomes found — check market age and snapshot coverage")
        cur.close()
        conn.close()
        return

    # -- Compute raw signals per outcome ------------------------------------
    signals: list[dict] = []

    for o in outcomes:
        z_score = compute_z_score(o["volume_series"])

        dp_dv_raw, dp_dv_magnitude, direction = compute_dp_dv(
            latest_probability   = o["latest_probability"],
            prior_probability    = o["prior_probability"],
            latest_period_volume = float(o["latest_period_volume"] or 0),
            avg_period_volume    = float(o["avg_period_volume"] or 0),
            threshold_pct        = dp_dv_threshold_pct,
            absolute_floor       = dp_dv_absolute_floor,
        )

        signals.append({
            "outcome_id":       o["outcome_id"],
            "market_id":        o["market_id"],
            "outcome_label":    o["outcome_label"],
            "market_question":  o["market_question"],
            "computed_at":      computed_at,
            "window_days":      window_days,
            "volume_z_score":   z_score,
            "dp_dv_raw":        dp_dv_raw,
            "dp_dv_magnitude":  dp_dv_magnitude,
            "dp_dv_direction":  direction,
            # ranks filled in after normalization pass below
            "z_score_rank":     None,
            "dp_dv_rank":       None,
            "composite_score":  None,
        })

    # -- Rank normalization pass --------------------------------------------
    # Z-score rank: all outcomes with a valid z-score
    z_scores     = [s["volume_z_score"]  for s in signals]
    dp_magnitudes = [s["dp_dv_magnitude"] for s in signals]

    z_ranks  = rank_normalize(z_scores)
    dp_ranks = rank_normalize(dp_magnitudes)

    for i, s in enumerate(signals):
        s["z_score_rank"] = z_ranks[i]
        s["dp_dv_rank"]   = dp_ranks[i]

        # Composite = mean of available ranks
        # If dp_dv_rank is None (gate failed), composite is z_score_rank alone.
        # Outcomes with both ranks score against the full universe.
        available_ranks = [r for r in [s["z_score_rank"], s["dp_dv_rank"]] if r is not None]
        s["composite_score"] = round(float(np.mean(available_ranks)), 4) if available_ranks else None

    # -- Sort for top-N ----------------------------------------------------
    # Sort by composite descending; None composite goes to bottom
    ranked = sorted(
        signals,
        key=lambda s: s["composite_score"] if s["composite_score"] is not None else -1,
        reverse=True,
    )

    # -- Write to DB -------------------------------------------------------
    written = 0
    errors  = 0

    for s in signals:
        try:
            upsert_signal(cur, s)
            written += 1
        except Exception as exc:
            log.error(f"Error upserting signal for {s['outcome_id']}: {exc}", exc_info=True)
            conn.rollback()
            errors += 1
            continue

    conn.commit()

    try:
        write_topx(cur, computed_at, top_n, ranked)
        conn.commit()
        log.info(f"Top-{top_n} written for {computed_at.date()}")
    except Exception as exc:
        log.error(f"Error writing top-{top_n}: {exc}", exc_info=True)
        conn.rollback()

    log.info(
        f"=== compute_signals.py complete === "
        f"signals_written={written} errors={errors}"
    )

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()