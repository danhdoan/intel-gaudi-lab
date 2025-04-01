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
