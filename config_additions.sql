-- Additional pipeline_config entries for compute_signals.py
-- Run this after the initial schema creation

INSERT INTO pipeline_config VALUES
    ('dp_dv_threshold_pct',       '0.15'),   -- 15% of avg period volume (relative gate)
    ('dp_dv_absolute_floor_usdc', '5.0')     -- hard noise floor in USDC (absolute gate)
ON CONFLICT (key) DO NOTHING;
