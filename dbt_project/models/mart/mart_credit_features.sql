{{ config(materialized='table') }}

WITH stg AS (
    SELECT * FROM {{ ref('stg_credit_applications') }}
)

SELECT
    id,
    serious_dlqin2yrs                                               AS label,
    revolving_utilization,
    age,
    debt_ratio,
    monthly_income,
    open_credit_lines,
    times_90_days_late,
    real_estate_loans,
    times_60_89_days_late,
    dependents,
    -- Derived: total late payments across all buckets
    (times_30_59_days_late + times_60_89_days_late + times_90_days_late)
                                                                    AS total_late_payments,
    times_30_59_days_late,
    -- Derived: debt-to-income (debt_ratio already expresses this ratio)
    debt_ratio                                                      AS dti,
    -- Derived: utilization segment
    CASE
        WHEN revolving_utilization <= 0.3 THEN 'low'
        WHEN revolving_utilization <= 0.7 THEN 'medium'
        ELSE 'high'
    END                                                             AS utilization_segment,
    ingested_at
FROM stg
