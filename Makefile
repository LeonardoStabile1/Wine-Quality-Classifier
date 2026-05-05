PYTHON = .venv/bin/python
PIP = .venv/bin/pip

PROJECT = src

.PHONY: help install lint test run clean

help:
	@echo "Targets disponíveis:"
	@echo "  install   - Instala dependências"
	@echo "  lint      - Roda flake8"
	@echo "  test      - Executa pytest"
	@echo "  run       - Executa pipeline de treino"
	@echo "  clean     - Remove arquivos temporários"

install:
	$(PIP) install -r requirements.txt

lint:
	$(PYTHON) -m flake8 $(PROJECT) tests

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) -m src.pipelines.training_pipeline

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache