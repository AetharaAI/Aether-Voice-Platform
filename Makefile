.PHONY: build start stop logs test clean verify

# Configuration
export $(shell sed 's/=.*//' .env 2>/dev/null || echo "")
COMPOSE=docker-compose -f docker-compose.yml

build:
	@echo "🏗️ Building Aether Voice Platform..."
	$(COMPOSE) build --no-cache

start:
	@echo "🚀 Starting services..."
	$(COMPOSE) up -d
	@echo "⏳ Waiting for health checks..."
	@sleep 30
	$(COMPOSE) ps

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
	@echo "🧹 Cleaning up..."
	$(COMPOSE) down -v
	docker system prune -f

restart: stop start

status:
	$(COMPOSE) ps
