# Learn Retrieval Augmented Generation (RAG)

This repository contains the companion files and starter project (`rag-search-engine`) used throughout the **Learn Retrieval Augmented Generation** course on [Boot.dev](https://www.boot.dev).

The goal of this project is to build a full-featured RAG pipeline and search engine for movie data—moving from basic keyword matching and inverted indexes all the way to semantic search with vector embeddings, chunking, re-ranking, and LLM-powered context generation.

---

## 🛠 Features

- **Text Processing & Inverted Indexing**: Custom tokenization, stopword removal, stemming (`PorterStemmer`), TF-IDF scoring, and fast lookup using inverted indexes.
- **Vector Search & Embeddings**: Dense vector generation using Hugging Face models (`all-MiniLM-L6-v2`) and multimodal models (such as `clip-ViT-B-32`).
- **Hybrid Search**: Combining keyword search and dense semantic search scores using techniques like Reciprocal Rank Fusion (RRF).
- **RAG Generation**: Context formatting and prompt augmentation to send relevant document chunks to LLMs for grounded Q&A.
- **CLI Commands**: Command-line interfaces to run search queries, inspect TF-IDF scores, generate embeddings, and evaluate responses.

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.10+** (Python 3.11 recommended)
- **uv** (Fast Python package installer and venv manager)
- **Boot.dev CLI** (For submitting lesson checks)

---

### 1. Project Initialization

Clone the repository and set up your project directory:

```bash
git clone [https://github.com/hasheddev/rag.git](https://github.com/hasheddev/rag.git)
cd rag
uv venv

# Activate on Linux/macOS
source .venv/bin/activate
mkdir -p data
```

Place movies.json into the data/ folder:

rag-search-engine/
├── data/
│ └── movies.json
├── main.py
├── requirements.txt
└── README.md

Add openrouter apikey to .env file

```bash
OPENROUTER_API_KEY=
```

### 2.💻 Usage

Running Starter Script
Verify your Python virtual environment and code execution:

Bash
uv run test_llm.py

Run carious clis with

```bash
uv run  **.cli.py
```
