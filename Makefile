.PHONY: help install-bot install-api install-dashboard install-all setup-db train test clean

help:
	@echo "RLTrade - Available Commands:"
	@echo "  make install-all       - Install all dependencies (bot, api, dashboard)"
	@echo "  make install-bot       - Install bot dependencies"
	@echo "  make install-api       - Install API dependencies"
	@echo "  make install-dashboard - Install dashboard dependencies"
	@echo "  make setup-db          - Initialize database schema"
	@echo "  make train             - Start bot training"
	@echo "  make train-quick       - Quick training run (1k episodes)"
	@echo "  make dev-api           - Start API in development mode"
	@echo "  make dev-dashboard     - Start dashboard in development mode"
	@echo "  make test              - Run all tests"
	@echo "  make clean             - Clean build artifacts"

install-bot:
	cd bot && pip install -r requirements.txt

install-api:
	cd api && pip install -r requirements.txt

install-dashboard:
	cd dashboard && npm install

install-all: install-bot install-api install-dashboard
	@echo "All dependencies installed!"

setup-db:
	cd api && alembic upgrade head
	@echo "Database schema initialized"

train:
	cd bot && python main.py train --episodes 100000

train-quick:
	cd bot && python main.py train --episodes 1000

eval:
	cd bot && python main.py evaluate

dev-api:
	cd api && uvicorn main:app --reload --port 8000

dev-dashboard:
	cd dashboard && npm run dev

test:
	pytest tests/ -v

test-bot:
	pytest bot/tests/ -v

test-api:
	pytest api/tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	cd dashboard && rm -rf .next out node_modules
