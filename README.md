# LangChain RAG Framework Evaluator (Pass 2)

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)

## What It Is
This project is Pass 2 of a two-part Retrieval-Augmented Generation (RAG) engineering study. While Pass 1 built the pipeline from scratch using raw HTTP clients and manual vector math, this iteration rebuilds the exact same architecture utilizing **LangChain** and **Ragas**. The objective is to evaluate what these orchestration frameworks abstract, optimize, and obfuscate during data ingestion, retrieval, and automated LLM-as-a-judge evaluation.

## Architecture

* **Data Ingestion:** `TextLoader` for document parsing and metadata wrapping.
* **Chunking Strategy:** `RecursiveCharacterTextSplitter` configured for semantic boundary preservation (`chunk_size=1000`, `chunk_overlap=200`).
* **Embeddings:** `GoogleGenerativeAIEmbeddings` (`gemini-embedding-2`, 768-dimensions) integrated with `CacheBackedEmbeddings` via local file system storage to prevent redundant API calls.
* **Vector Storage:** LangChain's `Chroma` wrapper for persistent vector management and implicit query embedding.
* **Generation:** `ChatGoogleGenerativeAI` (`gemini-3.5-flash-lite`) configured with `temperature=0.0` for deterministic, grounded answering.
* **Evaluation Suite:** `ragas` framework replacing manual cross-encoders. Computes `Faithfulness` (hallucination check) and `AnswerRelevancy` (context alignment) using LangChain wrappers.

---

## What LangChain & Ragas Abstract

Rebuilding the Pass 1 pipeline with LangChain exposed exactly how much boilerplate the framework handles under the hood. Here is the direct mapping of custom mechanics to framework abstractions:

* **File Handling & Parsing:** 
  * *Pass 1:* `open(file).read()`
  * *Pass 2:* `TextLoader`
  * *Abstraction:* Automates file encoding, error handling, and wraps text in standardized `Document` objects with source metadata.
* **Text Chunking:**
  * *Pass 1:* Custom `fixed_size_chunks()` via string splitting and loops.
  * *Pass 2:* `RecursiveCharacterTextSplitter`
  * *Abstraction:* Replaces blunt token counts with a hierarchy of separators (`\n\n`, `\n`, ` `) to keep paragraphs and sentences intact before enforcing the hard character limit.
* **Embedding Generation:**
  * *Pass 1:* Custom HTTP client with `asyncio.TaskGroup`, explicit `httpx` rate-limit handling, and backoff math.
  * *Pass 2:* `GoogleGenerativeAIEmbeddings`
  * *Abstraction:* Silently handles network retries, batching, and payload formatting natively.
* **Vector Database Operations:**
  * *Pass 1:* Raw `chromadb.PersistentClient`, manual MD5 ID generation, and explicit embedding calls prior to upsertion.
  * *Pass 2:* `Chroma` + `CacheBackedEmbeddings`
  * *Abstraction:* Syncs embeddings and storage in one step. The caching layer intercepts repeated document hashing to save API costs without writing explicit database sync logic.
* **Retrieval & Search:**
  * *Pass 1:* Raw `collection.query()`, parsing nested lists of distances and metadata.
  * *Pass 2:* `vectorstore.similarity_search()`
  * *Abstraction:* Automatically embeds the raw string query, executes the vector math, and returns a clean list of `Document` objects.
* **LLM Prompting & Generation:**
  * *Pass 1:* HTTP POST requests mapping JSON schemas to `v1beta/models/...:generateContent`.
  * *Pass 2:* `ChatGoogleGenerativeAI.invoke()`
  * *Abstraction:* Manages message schemas, unwraps the nested response payload, and handles connection timeouts internally.
* **Automated Evaluation:**
  * *Pass 1:* Custom deterministic cross-encoder scoring and manual LLM-as-a-judge prompting.
  * *Pass 2:* `ragas` metrics (`Faithfulness`, `AnswerRelevancy`)
  * *Abstraction:* Replaces single-pass manual prompts with complex, multi-step reverse-engineering chains to mathematically score relevance and extract grounded facts using strict JSON parsing.

---

## Evaluation Results: Pass 1 vs. Pass 2

| Metric | Pass 1 (Raw) | Pass 2 (LangChain) |
| :--- | :--- | :--- |
| **Total Queries** | 15 | 15 |
| **Average Relevance** | **1.00** | **0.98** |
| **Faithfulness** | **100%** | **100%** |
| **Eval Methodology** | CrossEncoder + LLM-as-a-judge | Ragas Framework |

> **Methodology Note on Relevance Variance:**
> The slight delta in relevance scores (1.00 vs. 0.98) reflects differing measurement techniques rather than a degradation in retrieval quality. Pass 1 used a localized `CrossEncoder` model (`ms-marco-MiniLM-L-6-v2`) computing raw pairwise semantic similarity scores passed through a sigmoid function. In contrast, Ragas evaluates `AnswerRelevancy` by using an LLM to reverse-engineer synthetic candidate questions directly from the generated answer and computing cosine similarity against the original query vector.

---

## How to Run

1. **Clone repository and set up environment:**

    ```bash
    git clone [https://github.com/yourusername/Langchain-RAG-Engine.git](https://github.com/yourusername/Langchain-RAG-Engine.git)
    cd Langchain-RAG-Engine
    cp .env.example .env
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run evaluation:**

    ```bash
    python main.py --data text.txt --queries queries.txt --output eval_results.md
    ```

## What I Learned / What LangChain Abstracts

* [x] **Document loaders vs. manual file reading:** Standardizes text and metadata encapsulation over raw filesystem I/O.
* [x] **TextSplitter abstractions vs. explicit NLTK / sliding window loops:** Implements recursive separator fallback rather than blunt token slices.
* [x] **Vector Store caching vs. raw upserts:** Manages local embedding key-value stores to prevent redundant external API round-trips.
* [x] **Built-in evaluators vs. custom Cross-Encoder / LLM judge loops:** Replaces ad-hoc string matching with structured metric frameworks (`ragas`).
* [x] **Framework overhead, execution latency, and debugging visibility:** Balances rapid prototyping against reduced visibility into lower-level network retries and schema mismatches.