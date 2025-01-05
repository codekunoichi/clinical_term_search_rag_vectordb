# clinical_term_search_rag_vectordb
Experiments with RAG, VectorDB, Semantic Search for Clinical Terms

## Environment Setup
Clone the repository.

Set up the virtual environment and install the required libraries:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the proper file path for running the app:

```
export PYTHONPATH=$(pwd)
```

Token Credits and API Key Setup

Before running the application, you need to purchase token credits for OpenAI and/or Anthropic if required.

OpenAI Credits: Purchase token credits from OpenAI.
Anthropic Credits: Purchase token credits from Anthropic.
After purchasing credits, set the following environment variables for your API keys:

```
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

