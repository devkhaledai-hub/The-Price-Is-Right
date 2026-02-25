# The Price Is Right Project

An agentic AI system that scans online deal feeds, estimates product fair value using an ensemble of models, and surfaces high-discount opportunities in a Gradio app.
<img width="1349" height="625" alt="image" src="https://github.com/user-attachments/assets/3fe23641-ff9a-4467-9d10-2e4bb3e101b0" />

## What this project does

- Scans deal RSS feeds (DealNews categories).
- Uses an agent framework to:
  - select candidate deals,
  - estimate true value with multiple pricing agents,
  - compute discount opportunities,
  - optionally send push alerts.
- Stores retrieval data in a local Chroma vector DB (`products_vectorstore/`).
- Persists discovered opportunities in `memory.json`.
- Visualizes activity and opportunities in a Gradio UI.

## Project structure

- `price_is_right.py`: Main Gradio app entrypoint.
- `deal_agent_framework.py`: Orchestrates memory, Chroma, and planning flow.
- `agents/`: Scanner, planner, ensemble, frontier, neural network, messaging, and data models.
- `app.ipynb`: Notebook walkthrough and runner (`!uv run price_is_right.py`).
- `products_vectorstore/`: Local Chroma persistent storage.
- `memory.json`: Stored opportunities from previous runs.

## Prerequisites

- Python 3.11+
- `uv` installed (`pip install uv`)
- Internet access (RSS feeds + model/provider APIs)

## 1) Create and activate environment

Using `uv` (recommended):

```powershell
uv venv
.\.venv\Scripts\activate
```

Or with Python:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

## 2) Install dependencies

`pyproject.toml` currently has no dependency list, so install manually:

```powershell
uv pip install `
  gradio plotly python-dotenv chromadb scikit-learn numpy `
  litellm groq openai sentence-transformers `
  pydantic beautifulsoup4 feedparser requests tqdm `
  pandas datasets torch modal
```

If you want to run notebooks:

```powershell
uv pip install notebook ipykernel
```

## 3) Configure `.env`

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key
PUSHOVER_USER=your_pushover_user
PUSHOVER_TOKEN=your_pushover_token

# Optional
OPENAI_API_KEY=your_openai_key
HF_TOKEN=your_huggingface_token
PRICER_PREPROCESSOR_MODEL=ollama/llama3.2
```

Notes:

- `GROQ_API_KEY` is required by current scanner/frontier/autonomous agents.
- `PUSHOVER_*` is needed for push notifications.
- `HF_TOKEN` is only needed for workflows that use Hugging Face access.
- `OPENAI_API_KEY` is optional for current code paths (some OpenAI paths are commented/legacy).

## 4) Run the app

Direct run:

```powershell
uv run price_is_right.py
```

Then open the local Gradio URL shown in terminal (typically `http://127.0.0.1:7860`).

## 5) Run from notebook (`app.ipynb`)

1. Start Jupyter:
```powershell
uv run jupyter notebook
```
2. Open `app.ipynb`.
3. Run cells in order.
4. The notebook ends with:
```python
!uv run price_is_right.py
```
5. Open the displayed local Gradio URL.

## 6) Reset memory (optional)

To keep only the first two stored opportunities:

```python
from deal_agent_framework import DealAgentFramework
DealAgentFramework.reset_memory()
```

## Common issues

- Missing API key errors:
  - Ensure `.env` exists in project root.
  - Ensure `load_dotenv(override=True)` is executed (already done in app/framework modules).
- First run can be slow:
  - Chroma retrieval + model warm-up may take time.
- No deal alerts sent:
  - Check `PUSHOVER_USER` / `PUSHOVER_TOKEN`.
  - A discount must exceed planner threshold (`DEAL_THRESHOLD = 50` in `agents/planning_agent.py`).

## Security note

Do not commit real API keys/tokens to Git. Keep `.env` out of version control and rotate any key that has been exposed.
