import json
import time

import pandas as pd
from confluent_kafka import Producer
from loguru import logger

from config import settings

COLUMN_MAP = {
    "SeriousDlqin2yrs": "serious_dlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "times_30_59_days_late",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "times_90_days_late",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "times_60_89_days_late",
    "NumberOfDependents": "dependents",
}


def _delivery_report(err, msg):
    if err:
        logger.error(f"Delivery failed: {err}")
    else:
        logger.debug(f"Delivered to {msg.topic()} [{msg.partition()}]")


def produce_from_csv(csv_path: str = "data/cs-training.csv", delay: float = 0.01) -> int:
    df = pd.read_csv(csv_path).rename(columns=COLUMN_MAP)
    df = df[list(COLUMN_MAP.values())].dropna()

    producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})
    count = 0

    for _, row in df.iterrows():
        payload = json.dumps(row.to_dict()).encode("utf-8")
        producer.produce(settings.kafka_topic, value=payload, callback=_delivery_report)
        producer.poll(0)
        count += 1
        if delay:
            time.sleep(delay)

    producer.flush()
    logger.info(f"Produced {count} messages to topic '{settings.kafka_topic}'")
    return count


if __name__ == "__main__":
    produce_from_csv()
