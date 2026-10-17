UV ?= uv
RUFF = $(UV) run --no-project --with ruff==0.16.7 ruff
PYTHON = $(UV) run --no-project python
QUALITY_PATHS = \
	src/wnn/training \
	scripts/train_rnn.py \
	scripts/e540_cli_preflight.py \
	scripts/check_training_structure.py \
	tests/test_training_structure_checker.py

.PHONY: quality lint format-check training-structure quality-tests

quality: lint format-check training-structure

lint:
	$(RUFF) check $(QUALITY_PATHS)

format-check:
	$(RUFF) format --check $(QUALITY_PATHS)

training-structure:
	$(PYTHON) scripts/check_training_structure.py

quality-tests:
	$(PYTHON) -m unittest tests.test_training_structure_checker
