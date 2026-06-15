-- Creates the airflow database alongside the main creditlens database
CREATE DATABASE airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO creditlens;
