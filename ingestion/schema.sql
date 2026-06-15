CREATE TABLE IF NOT EXISTS raw_credit_applications (
    id                      SERIAL PRIMARY KEY,
    serious_dlqin2yrs       INTEGER,
    revolving_utilization   FLOAT,
    age                     INTEGER,
    times_30_59_days_late   INTEGER,
    debt_ratio              FLOAT,
    monthly_income          FLOAT,
    open_credit_lines       INTEGER,
    times_90_days_late      INTEGER,
    real_estate_loans       INTEGER,
    times_60_89_days_late   INTEGER,
    dependents              INTEGER,
    source                  VARCHAR(20) DEFAULT 'kafka',
    ingested_at             TIMESTAMP DEFAULT NOW()
);
