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

## Evaluation Results

| Metric | Score |
| :--- | :--- |
| **Total Queries Evaluated** | 15 |
| **Average Relevance Score** |  0.9837 |
| **Faithfulness Rate** |100% |

---

## How to Run

1. **Set up environment variables:**
   Ensure your Google Gemini API key is configured in the `.env` file.
   ```bash
   GEMINI_API_KEY="your_api_key_here"