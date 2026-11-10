UV ?= uv
RUFF = $(UV) run --no-project --with ruff==0.16.7 ruff
PYTHON = $(UV) run --no-project python
TEST_PYTHON ?= .venv/bin/python
QUALITY_PATHS = \
	research/submission_candidates/tl3-lstm-complement-20261110 \
	src/wnn/training \
	scripts/train_rnn.py \
	scripts/train_sequence.py \
	scripts/predict_sequence.py \
	scripts/score_sequence.py \
	scripts/check_sequence_ddp.py \
	tests/test_b300_sequence_models.py \
	tests/test_b300_sequence_contracts.py \
	scripts/e540_cli_preflight.py \
	scripts/check_training_structure.py \
	tests/test_training_structure_checker.py \
	tests/test_training_compatibility.py \
	tests/test_training_input_allocation.py \
	src/wnn/prediction_safety.py \
	tests/test_prediction_safety.py
TRAINING_TESTS = \
	tests/test_training_compatibility.py \
	tests/test_training_input_allocation.py \
	tests/test_training_structure_checker.py \
	tests/test_trainer_resume_integration.py \
	tests/test_features.py \
	tests/test_causal_moment_features.py \
	tests/test_field_age_features.py \
	tests/test_prediction_safety.py

.PHONY: quality lint format-check training-structure quality-tests training-tests

quality: lint format-check training-structure

lint:
	$(RUFF) check $(QUALITY_PATHS)

format-check:
	$(RUFF) format --check $(QUALITY_PATHS)

training-structure:
	$(PYTHON) scripts/check_training_structure.py

quality-tests:
	$(PYTHON) -m unittest tests.test_training_structure_checker

training-tests:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WNN_THREADS=1 \
		PYTHONPATH=src $(TEST_PYTHON) -m pytest -q $(TRAINING_TESTS)
