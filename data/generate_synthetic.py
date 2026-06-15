"""
Generates synthetic credit application data matching the schema of the
Kaggle 'Give Me Some Credit' dataset.

Use this if you don't want to (or can't) download the real Kaggle CSV.
The schema, column names, and reasonable value ranges match the original.

Usage:
    python data/generate_synthetic.py
"""

import numpy as np
import pandas as pd

N_ROWS = 30_000
RNG = np.random.default_rng(42)


def generate() -> pd.DataFrame:
    age = RNG.integers(21, 80, N_ROWS)
    monthly_income = RNG.lognormal(mean=8.4, sigma=0.7, size=N_ROWS).clip(500, 50_000)
    debt_ratio = RNG.beta(2, 5, N_ROWS).clip(0, 2)
    revolving_utilization = RNG.beta(2, 4, N_ROWS).clip(0, 1.5)
    open_credit_lines = RNG.poisson(8, N_ROWS).clip(0, 30)
    real_estate_loans = RNG.poisson(1, N_ROWS).clip(0, 6)
    dependents = RNG.poisson(0.8, N_ROWS).clip(0, 8)

    times_30_59 = RNG.poisson(0.3, N_ROWS).clip(0, 15)
    times_60_89 = RNG.poisson(0.15, N_ROWS).clip(0, 10)
    times_90 = RNG.poisson(0.1, N_ROWS).clip(0, 10)

    risk_score = (
        0.45 * revolving_utilization
        + 0.25 * debt_ratio
        + 0.20 * (times_30_59 + 2 * times_60_89 + 3 * times_90) / 5
        - 0.15 * (age - 40) / 40
        - 0.10 * np.log1p(monthly_income) / 12
    )
    prob_default = 1 / (1 + np.exp(-(risk_score - 0.4) * 5))
    serious_dlqin2yrs = (RNG.random(N_ROWS) < prob_default).astype(int)

    df = pd.DataFrame(
        {
            "SeriousDlqin2yrs": serious_dlqin2yrs,
            "RevolvingUtilizationOfUnsecuredLines": revolving_utilization,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": times_30_59,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
            "NumberOfTimes90DaysLate": times_90,
            "NumberRealEstateLoansOrLines": real_estate_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": times_60_89,
            "NumberOfDependents": dependents,
        }
    )
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/cs-training.csv", index=False)
    print(f"Generated {len(df):,} rows -> data/cs-training.csv")
    print(f"Default rate: {df['SeriousDlqin2yrs'].mean():.2%}")
