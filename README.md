---
title: GitLab Handbook RAG Chatbot
emoji: 📚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# RAG Knowledge Base Chatbot

Deployable RAG chatbot for the GitLab Handbook assignment. The app ingests Markdown files, chunks them, embeds them with an open-source model, stores vectors in local ChromaDB, retrieves relevant context, and answers with citations through a Gradio chat UI.

## What is included

- Markdown ingestion pipeline
- ChromaDB persistent vector store
- Retrieval with relevance filtering
- LiteLLM-based answer generation
- Golden dataset evaluation runner
- Gradio app with chat history and source citations
- Hugging Face Spaces deployment guidance

## Project layout

```text
.
|-- app.py
|-- requirements.txt
|-- .env.example
|-- src/rag_chatbot/
|   |-- __init__.py
|   |-- config.py
|   |-- ingest.py
|   |-- retrieval.py
|   |-- chatbot.py
|   `-- evaluate.py
|-- data/
|   |-- handbook/
|   |   `-- README.md
|   `-- golden_dataset.json
|-- chroma_db/
|   `-- .gitkeep
`-- evaluation_results/
    `-- .gitkeep
```

## Setup

1. Create a virtual environment and install dependencies.
2. Copy `.env.example` to `.env` and fill in your LiteLLM model and API key.
   If you are using Google Gemini through LiteLLM, set `GOOGLE_API_KEY` or `GEMINI_API_KEY`.
3. For local development, either place the GitLab Handbook Markdown files under `data/handbook/` or set `HANDBOOK_DIR` in `.env` to an external dataset path.
4. Run ingestion:

```bash
python -m src.rag_chatbot.ingest
```

5. Run evaluation:

```bash
python -m src.rag_chatbot.evaluate
```

6. Launch the app:

```bash
python app.py
```

## Golden dataset

Edit `data/golden_dataset.json` after you manually review the handbook files. The assignment requires 5 to 10 QA pairs, including at least 1 negative example that should be refused because the answer is not in the knowledge base.

## Hugging Face Spaces deployment notes

- Set the Hugging Face Space SDK to `gradio`.
- Upload the full repository.
- Keep the generated `chroma_db/` folder in the Space repository. This is required because the app expects a local persistent Chroma database at runtime.
- Add your Gemini secret in the Space settings, for example `GEMINI_API_KEY`.
- In the deployed Space, set `LITELLM_MODEL=gemini/gemini-2.5-flash`.
- In the deployed Space, set `HANDBOOK_DIR=./data/handbook`.
- If you kept the handbook Markdown files outside the repo locally, do not use the Windows path in Space settings. Switch `HANDBOOK_DIR` back to the repo-relative path above and include any required dataset files in the repo if you want ingestion to run there.

## Suggested workflow once the dataset arrives

1. Review a sample of handbook files and write the final golden dataset.
2. Run ingestion to build `chroma_db/`.
3. Run evaluation and inspect `evaluation_results/`.
4. Tune `RELEVANCE_THRESHOLD`, chunking, and prompt wording.
5. Deploy the app to Hugging Face Spaces.
