# AI & RAG Application Testing

A hands-on QA project demonstrating evaluation and testing of Retrieval-Augmented Generation (RAG) applications using RAGAs, DeepEval, Python, and locally hosted LLMs.

## Project Overview

This project explores how QA engineers can validate the quality, reliability, relevance, and factual accuracy of AI-generated responses in RAG applications.

The evaluation environment uses Ollama with the qwen2.5-coder:3b model to run local LLM-based evaluations.

## Technology Stack

- Python 3.14
- RAGAs
- DeepEval
- Ollama
- qwen2.5-coder:3b
- nomic-embed-text
- Pandas
- Streamlit
- CSV reporting
- Git and GitHub

## Evaluation Coverage

The project includes experiments with the following evaluation metrics:

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

## Advanced Testing Capabilities

- Integrated RAGAs evaluation suite
- Multiple dataset evaluation
- Positive and negative test scenarios
- Automated dataset validation
- Metric threshold validation
- CSV-based evaluation reports
- Interactive Streamlit dashboard
- Scenario-wise evaluation analysis
- Failed scenario identification
- QA recommendations based on evaluation results

## Dashboard

Run the dashboard locally:

```powershell
streamlit run ragas_dashboard.py