## Setup

```bash
pip install -r requirements.txt
```

## Environment Variables

```bash
GOOGLE_API_KEY=***
LLM_API_KEY=***  # custom LLM endpoint
LLM_API_URL=***  # custom LLM endpoint
LLM_TIMEOUT=10
LLM_MAX_RETRIES=2
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## Run

```bash
streamlit run chat_ui.py
```