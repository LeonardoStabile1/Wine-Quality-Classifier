# Running the Project

Instructions to run this project:

### Create the environment

```bash
python -m venv .venv
```

### Activate the environment

#### Linux / macOS

```bash
source .venv/bin/activate
```

#### Windows

```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
make install
```

Or:

```bash
pip install -r requirements.txt
```

### Train the model

```bash
make train ARGS="--simple"
```

Or:

```bash
python -m src.pipelines.training_pipeline --simple
```

### Run the API

```bash
make run_api
```

Or:

```bash
uvicorn src.api.main:app --reload
```

API URL:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```