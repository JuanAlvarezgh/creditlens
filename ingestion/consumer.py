import json

from confluent_kafka import Consumer
from loguru import logger
from sqlalchemy.engine import Engine

from config import settings
from ingestion.db import get_engine, insert_application


def process_message(msg, engine: Engine) -> None:
    if msg.error():
        logger.warning(f"Consumer error: {msg.error()}")
        return

    try:
        record = json.loads(msg.value().decode("utf-8"))
        insert_application(record, engine)
        logger.debug(f"Inserted record age={record.get('age')}")
    except Exception as e:
        logger.error(f"Failed to process message: {e}")


def consume(max_messages: int | None = None) -> None:
    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": "creditlens-consumer",
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([settings.kafka_topic])
    engine = get_engine()
    count = 0

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            process_message(msg, engine)
            count += 1
            if max_messages and count >= max_messages:
                break
    finally:
        consumer.close()
        logger.info(f"Consumed {count} messages")
