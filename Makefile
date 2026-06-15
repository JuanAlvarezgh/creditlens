.PHONY: setup start stop test lint format

setup:
	pip install -r requirements.txt
	pre-commit install
	python -c "import shutil, os; shutil.copyfile('.env.example', '.env') if not os.path.exists('.env') else print('.env exists, skipping')"

start:
	docker compose up -d

stop:
	docker compose down -v

test:
	pytest tests/ -v

lint:
	ruff check .
	black --check .
	isort --check-only .

format:
	black .
	isort .
	ruff check --fix .
