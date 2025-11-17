# Makefile for WebWork Python

.PHONY: help install dev test lint format clean docker-build docker-up docker-down

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies
	pip install -e .
	cd webwork_api && pip install -r requirements_v2.txt
	cd webwork-frontend && npm install

dev-backend:  ## Run backend in development mode
	cd webwork_api && python main_v2.py

dev-frontend:  ## Run frontend in development mode
	cd webwork-frontend && npm run dev

test-backend:  ## Run backend tests
	cd webwork_api && pytest

test-frontend:  ## Run frontend tests
	cd webwork-frontend && npm test

test:  ## Run all tests
	$(MAKE) test-backend
	$(MAKE) test-frontend

lint-backend:  ## Lint backend code
	cd webwork_api && ruff check .
	cd webwork_api && mypy .

lint-frontend:  ## Lint frontend code
	cd webwork-frontend && npm run lint

lint:  ## Lint all code
	$(MAKE) lint-backend
	$(MAKE) lint-frontend

format-backend:  ## Format backend code
	cd webwork_api && black .
	cd webwork_api && ruff check --fix .

format-frontend:  ## Format frontend code
	cd webwork-frontend && npm run format

format:  ## Format all code
	$(MAKE) format-backend
	$(MAKE) format-frontend

clean:  ## Clean build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	cd webwork-frontend && rm -rf node_modules dist

docker-build:  ## Build Docker images
	docker-compose build

docker-up:  ## Start Docker containers
	docker-compose up -d

docker-down:  ## Stop Docker containers
	docker-compose down

docker-logs:  ## View Docker logs
	docker-compose logs -f

docker-shell-api:  ## Open shell in API container
	docker-compose exec api /bin/bash

docker-shell-db:  ## Open shell in database container
	docker-compose exec db psql -U webwork

docker-restart:  ## Restart Docker containers
	docker-compose restart

# Production commands
build-frontend:  ## Build frontend for production
	cd webwork-frontend && npm run build

deploy:  ## Deploy to production (customize as needed)
	@echo "Deploy to your production environment"
	$(MAKE) test
	$(MAKE) build-frontend
	@echo "Ready for deployment"
