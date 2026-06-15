from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import settings


def get_engine() -> Engine:
    return create_engine(settings.postgres_url, pool_pre_ping=True)


def insert_application(record: dict, engine: Engine) -> None:
    sql = text(
        """
        INSERT INTO raw_credit_applications (
            serious_dlqin2yrs, revolving_utilization, age,
            times_30_59_days_late, debt_ratio, monthly_income,
            open_credit_lines, times_90_days_late, real_estate_loans,
            times_60_89_days_late, dependents, source
        ) VALUES (
            :serious_dlqin2yrs, :revolving_utilization, :age,
            :times_30_59_days_late, :debt_ratio, :monthly_income,
            :open_credit_lines, :times_90_days_late, :real_estate_loans,
            :times_60_89_days_late, :dependents, :source
        )
    """
    )
    with engine.begin() as conn:
        conn.execute(sql, {**record, "source": record.get("source", "kafka")})
