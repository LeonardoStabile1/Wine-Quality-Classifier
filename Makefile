PYTHON = python
PIP = python -m pip

PROJECT = src

.PHONY: help install lint test train run_api clean

help:
	@echo "Available targets:"
	@echo ""
	@echo "  install    Install project dependencies"
	@echo "  lint       Run flake8 linting"
	@echo "  test       Run pytest test suite"
	@echo "  train      Execute training pipeline"
	@echo "              Example: make train ARGS=\"--simple\""
	@echo "  run_api    Start FastAPI development server"
	@echo "  clean      Remove temporary files"

install:
	$(PIP) install -r requirements.txt

lint:
	$(PYTHON) -m flake8 $(PROJECT) tests

test:
	$(PYTHON) -m pytest

train:
	$(PYTHON) -m src.pipelines.training_pipeline $(ARGS)
	$(MAKE) clean

run_api:
	uvicorn src.api.main:app --reload

clean:
	$(PYTHON) -c "import pathlib, shutil; \
	for pattern in ('*.pyc', '*.pyo', '*.log'): \
		[p.unlink() for p in pathlib.Path('.').rglob(pattern) if p.is_file()]; \
	[shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]"