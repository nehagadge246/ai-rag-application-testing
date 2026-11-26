# AI & RAG Application Testing

A hands-on QA engineering project demonstrating how to evaluate, test, and validate Retrieval-Augmented Generation (RAG) applications using Python, RAGAs, DeepEval, Ollama, and locally hosted embedding and language models.

## Project Overview

This project explores how QA engineers can validate the quality, reliability, relevance, accuracy, security, and performance of AI-powered RAG applications.

It includes practical testing exercises covering the complete RAG workflow, from document ingestion and embedding generation to retrieval, response evaluation, security checks, and production-readiness assessments.

The project uses locally hosted models through Ollama, including Qwen 2.5 Coder, Nomic Embed Text, and All-MiniLM.

## Technology Stack

- Python 3.14
- RAGAs
- DeepEval
- Ollama
- Qwen 2.5 Coder 3B
- Nomic Embed Text
- All-MiniLM
- Pandas
- Streamlit
- CSV-based reporting
- Git and GitHub

## Evaluation Coverage

The project includes experiments with the following evaluation metrics and techniques:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall
- Factual Correctness
- Semantic Similarity
- Noise Sensitivity
- Context Utilization
- Context Entity Recall
- Response Groundedness
- Answer Accuracy
- Answer Correctness
- Context Relevance
- Topic Adherence
- Aspect Critic
- Summary Score
- Domain-specific Rubrics
- Instance-specific Rubrics

## Advanced RAG Testing Coverage

### 1. Document Processing and Retrieval
- Document ingestion and chunking
- Embedding generation and semantic retrieval
- Top-K retrieval evaluation
- Retrieval threshold calibration and optimization
- Search relevance evaluation
- Hybrid search testing
- Query preprocessing and rewriting
- Query expansion and multi-query retrieval
- Reranking evaluation
- Embedding model comparison
- Chunking optimization
- Context window and token-limit experiments

### 2. Response Quality and Reliability
- Hallucination detection
- Unsupported-answer rejection
- Answer accuracy validation
- Response consistency testing
- Context relevance and grounding
- End-to-end RAG regression testing

### 3. Security Testing
- Direct prompt injection testing
- Instruction override attempts
- System prompt extraction attempts
- Credential and sensitive information exposure scenarios
- Role manipulation scenarios
- Indirect prompt injection through retrieved documents
- Untrusted document detection and quarantine

### 4. Performance and Robustness
- RAG pipeline robustness
- Failure injection testing
- Performance and load testing
- Stress and scalability testing
- Retrieval latency measurement
- Configuration and dependency checks

### 5. Production Readiness
- Ollama service health checks
- Model availability validation
- Dataset integrity checks
- Embedding generation checks
- Retrieval quality checks
- Unsupported-query handling
- Security filter verification
- Production-readiness reporting

## RAG Evaluation and Reporting

The project generates CSV reports containing test-level results and evaluation summaries.

Reports support analysis of:

- Test case pass and failure status
- Retrieval relevance and ranking
- Precision, recall, and MRR
- Similarity scores and threshold behavior
- Retrieval and embedding latency
- Hallucination and unsupported-answer detection
- Security findings and quarantined documents
- Production-readiness review results

## Interactive Dashboard

A Streamlit dashboard is available to review RAG evaluation results.

Run it from the project directory:

```powershell
streamlit run ragas_dashboard.py
```

## Running Individual Tests

Activate the Python virtual environment before running tests:

```powershell
.\ragas-venv\Scripts\Activate.ps1
```

Ensure Ollama is running and the required models are available.

Example commands:

```powershell
python test_rag_hybrid_search.py
python test_rag_reranking.py
python test_rag_security.py
python test_rag_hallucination.py
python test_rag_end_to_end_regression.py
python test_rag_production_readiness.py
```

Individual scripts generate CSV reports in the project directory.

## Important Testing Notes

This repository contains practical QA experiments using controlled datasets and local testing utilities.

Some tests use embeddings and deterministic validation rules rather than a live, end-to-end LLM-generated response. Security and prompt-injection checks demonstrate simulated safeguards, not a complete production security implementation.

Results such as pass rates and retrieval metrics apply to their respective test datasets and configurations. They should not be interpreted as guarantees of production performance or security.

## Project Goals

- Build practical knowledge of RAG application testing.
- Understand AI evaluation metrics and their limitations.
- Validate retrieval quality and response reliability.
- Identify hallucinations and unsupported responses.
- Explore prompt-injection and document-poisoning scenarios.
- Measure retrieval performance and latency.
- Develop repeatable QA reports and testing workflows.
- Apply software testing principles to AI and LLM applications.

## Repository

GitHub: https://github.com/nehagadge246/ai-rag-application-testing
