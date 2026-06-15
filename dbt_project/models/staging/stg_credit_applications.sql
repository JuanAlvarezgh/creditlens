{{ config(materialized='view') }}

WITH source AS (
    SELECT * FROM {{ source('creditlens', 'raw_credit_applications') }}
),

cleaned AS (
    SELECT
        id,
        COALESCE(serious_dlqin2yrs, 0)              AS serious_dlqin2yrs,
        CASE
            WHEN revolving_utilization > 1 THEN 1.0
            WHEN revolving_utilization < 0 THEN 0.0
            ELSE revolving_utilization
        END                                          AS revolving_utilization,
        CASE
            WHEN age < 18 OR age > 100 THEN NULL
            ELSE age
        END                                          AS age,
        GREATEST(times_30_59_days_late, 0)           AS times_30_59_days_late,
        GREATEST(debt_ratio, 0)                      AS debt_ratio,
        CASE
            WHEN monthly_income <= 0 OR monthly_income > 1000000 THEN NULL
            ELSE monthly_income
        END                                          AS monthly_income,
        GREATEST(open_credit_lines, 0)               AS open_credit_lines,
        GREATEST(times_90_days_late, 0)              AS times_90_days_late,
        GREATEST(real_estate_loans, 0)               AS real_estate_loans,
        GREATEST(times_60_89_days_late, 0)           AS times_60_89_days_late,
        GREATEST(COALESCE(dependents, 0), 0)         AS dependents,
        source,
        ingested_at
    FROM source
)

SELECT * FROM cleaned
WHERE age IS NOT NULL
  AND monthly_income IS NOT NULL
