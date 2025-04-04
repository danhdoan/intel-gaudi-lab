SHELL := /bin/bash
include .env

define show_header
	@echo "============================================================"
	@echo $(1)
	@echo "============================================================"
endef

# ==============================================================================

all: clean format lint test

# ==============================================================================

clean:
	$(call show_header, "Cleaning Source code...")
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type f -name ".DS_Store" -delete
	rm -rf temp/*

# ==============================================================================

test:
	$(call show_header, "Testing...")
	pytest -v

# ==============================================================================

format:
	$(call show_header, "Formatting Source Code...")
	black .

# ==============================================================================

lint:
	$(call show_header, "Linting Source Code...")
	ruff check --fix .

# ==============================================================================

# * DOCKER COMMANDS *

.PHONY: up
up:
	@if [ ! -f .env ]; then \
		echo "WARNING: .env file does not exist! 'example.env' copied to '.env'. Please update the configurations in the .env file running this target."; \
		cp example.env .env; \
        exit 1; \
	fi
	docker compose up -d;

.PHONY: down
down:
	docker compose down -v
	@if [[ "$(docker ps -q -f name=${DOCKER_CONTAINER})" ]]; then \
		echo "Terminating running container..."; \
		docker rm ${DOCKER_CONTAINER}; \
	fi

.PHONY: stop
stop:
	docker compose stop

.PHONY: connect
connect:
	docker exec -it ${DOCKER_CONTAINER} bash

# ==============================================================================
