PYTHON = python
PIP = python -m pip

PROJECT = src

.PHONY: help install lint test run clean

help:
	@echo "Targets disponíveis:"
	@echo ""
	@echo "  install            Instala dependências"
	@echo "  lint               Roda flake8"
	@echo "  test               Executa pytest"
	@echo "  run                Executa pipeline de treino"
	@echo "                     Exemplo: make run ARGS=\"--simple\""
	@echo "  clean              Remove arquivos temporários"

install:
	$(PIP) install -r requirements.txt

lint:
	$(PYTHON) -m flake8 $(PROJECT) tests

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) -m src.pipelines.training_pipeline $(ARGS)

clean:
	$(PYTHON) -c "import pathlib; \
[ p.unlink() for p in pathlib.Path('.').rglob('*.pyc') ]; \
[ p.unlink() for p in pathlib.Path('.').rglob('*.pyo') ]; \
[ p.unlink() for p in pathlib.Path('.').rglob('*.log') ]; \
[ p.rmdir() for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir() ]"