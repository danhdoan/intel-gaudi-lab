SHELL := /bin/bash
-include .env

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
	ruff --fix .

# ==============================================================================

# * DOCKER COMMANDS *

# ==============================================================================

.PHONY: image
image:
	@if [ ! -f .env ]; then \
		echo "WARNING: .env file not existed! Creating '.env' from sample 'example.env' file..."; \
		cp example.env .env; \
		echo "Please update the configurations in the '.env' file for this repo."; \
        exit 1; \
	fi

	@echo "Building '${DOCKER_IMAGE_NAME}' image for this application..."
	@docker build \
		--build-arg WORKING_DIR=${WORKING_DIR} \
		-t ${DOCKER_IMAGE_NAME} \
		-f ./Dockerfile .

# ==============================================================================

.PHONY: up
up:
	@if [ ! -f .env ]; then \
		echo "WARNING: .env file not existed! Creating '.env' from sample 'example.env' file..."; \
		cp example.env .env; \
		echo "Please update the configurations in the '.env' file for this repo."; \
        exit 1; \
	fi

	@echo "Booting from '${DOCKER_IMAGE_NAME}' image...";
	@if docker compose up -d; then \
		echo "App Container Booted!"; \
	else \
		echo "Booting failed, please run 'make image' command before creating app container."; \
	fi

# ==============================================================================

.PHONY: down
down:
	docker compose down -v
	@if [[ "$(docker ps -q -f name=${DOCKER_CONTAINER})" ]]; then \
		echo "Terminating ${DOCKER_CONTAINER} container..."; \
		docker rm ${DOCKER_CONTAINER}; \
	fi

# ==============================================================================

.PHONY: clean-image
clean-image:
	@echo "Remove ${DOCKER_IMAGE_NAME} built image."
	docker rmi ${DOCKER_IMAGE_NAME}

# ==============================================================================

.PHONY: stop
stop:
	docker compose stop

# ==============================================================================

.PHONY: connect
connect:
	docker exec -it ${DOCKER_CONTAINER} bash

# ==============================================================================
