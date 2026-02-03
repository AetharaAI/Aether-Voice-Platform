.PHONY: build start stop logs test clean verify rebuild dc-build dc-start

# Load environment variables from .env file if it exists
ifneq (,$(wildcard .env))
    include .env
    export
endif

# Use docker compose (v2) if available, fallback to docker-compose
DC := $(shell if command -v docker-compose >/dev/null 2>&1; then echo docker-compose; else echo docker compose; fi)

COMPOSE=$(DC) -f docker-compose.yml

# -----------------------------------------------------------------------------
# Make-based commands (convenience wrappers)
# -----------------------------------------------------------------------------

build:
	@echo "🏗️ Building Aether Voice Platform..."
	$(COMPOSE) build

dc-build:
	@echo "🏗️ Building (no cache)..."
	$(COMPOSE) build --no-cache

start:
	@echo "🚀 Starting services..."
	$(COMPOSE) up -d
	@echo "⏳ Waiting for health checks..."
	@sleep 30
	$(COMPOSE) ps

dc-start:
	@echo "🚀 Starting services (force recreate)..."
	$(COMPOSE) up -d --force-recreate

stop:
	@echo "🛑 Stopping services..."
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=100

logs-asr:
	$(COMPOSE) logs -f asr-service

logs-tts:
	$(COMPOSE) logs -f tts-service

logs-omni:
	$(COMPOSE) logs -f omni-service

logs-gateway:
	$(COMPOSE) logs -f gateway

verify:
	@echo "🔍 Verifying model files..."
	bash scripts/verify-models.sh

test-asr:
	python scripts/test-asr.py

test-tts:
	python scripts/test-tts.py

test-omni:
	python scripts/test-omni.py

benchmark:
	python scripts/benchmark-latency.py

shell-asr:
	$(COMPOSE) exec asr-service bash

shell-tts:
	$(COMPOSE) exec tts-service bash

shell-omni:
	$(COMPOSE) exec omni-service bash

clean:
	@echo "🧹 Cleaning up containers and volumes..."
	$(COMPOSE) down -v --remove-orphans

dc-clean:
	@echo "🧹 Full cleanup including images..."
	$(COMPOSE) down -v --remove-orphans --rmi all

rebuild: clean dc-clean build start

status:
	$(COMPOSE) ps

# -----------------------------------------------------------------------------
# Pure Docker commands (no make needed)
# Copy-paste these directly if make fails
# -----------------------------------------------------------------------------

docker-help:
	@echo ""
	@echo "=== Pure Docker Commands (if make fails) ==="
	@echo ""
	@echo "Build:"
	@echo "  docker-compose build"
	@echo ""
	@echo "Start:"
	@echo "  docker-compose up -d"
	@echo ""
	@echo "Stop:"
	@echo "  docker-compose down"
	@echo ""
	@echo "View logs:"
	@echo "  docker-compose logs -f"
	@echo ""
	@echo "Check status:"
	@echo "  docker-compose ps"
	@echo ""
