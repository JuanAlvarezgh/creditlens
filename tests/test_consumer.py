import json
from unittest.mock import MagicMock, patch


def test_process_message_valid_record():
    from ingestion.consumer import process_message

    record = {
        "serious_dlqin2yrs": 0,
        "revolving_utilization": 0.5,
        "age": 35,
        "times_30_59_days_late": 0,
        "debt_ratio": 0.2,
        "monthly_income": 5000.0,
        "open_credit_lines": 4,
        "times_90_days_late": 0,
        "real_estate_loans": 1,
        "times_60_89_days_late": 0,
        "dependents": 2,
    }
    mock_msg = MagicMock()
    mock_msg.value.return_value = json.dumps(record).encode("utf-8")
    mock_msg.error.return_value = None
    mock_engine = MagicMock()

    with patch("ingestion.consumer.insert_application") as mock_insert:
        process_message(mock_msg, mock_engine)
        mock_insert.assert_called_once()
        called_record = mock_insert.call_args[0][0]
        assert called_record["age"] == 35
        assert called_record["revolving_utilization"] == 0.5


def test_process_message_skips_on_kafka_error():
    from ingestion.consumer import process_message

    mock_msg = MagicMock()
    mock_msg.error.return_value = Exception("broker error")
    mock_engine = MagicMock()

    with patch("ingestion.consumer.insert_application") as mock_insert:
        process_message(mock_msg, mock_engine)
        mock_insert.assert_not_called()
