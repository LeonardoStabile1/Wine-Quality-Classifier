# Running the Project

This project can be executed either with the provided `Makefile` commands or by running the commands manually.

---

# Option 1 — Using Makefile Commands

The repository includes a `Makefile` to simplify common development tasks such as dependency installation, model training, testing, and API execution.

## Create the Virtual Environment

```bash
python -m venv .venv
```

## Activate the Environment

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

## Install Dependencies

```bash
make install
```

## Train the Model

```bash
make train ARGS="--simple"
```

## Run the API

```bash
make run_api
```

## Additional Commands

Run tests:

```bash
make test
```

Run linting:

```bash
make lint
```

Remove temporary files:

```bash
make clean
```

Display all available commands:

```bash
make help
```

---

# Option 2 — Running Commands Manually

If `make` is not available in your environment, the project can also be executed manually.

## Create the Virtual Environment

```bash
python -m venv .venv
```

## Activate the Environment

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python -m src.pipelines.training_pipeline --simple
```

## Run the API

```bash
uvicorn src.api.main:app --reload
```

---

# API Access

Application URL:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```