

\# AI \& RAG Application Testing



A practical QA automation project focused on testing AI and LLM-based applications using Python, Ollama, and DeepEval.



\## Technologies Used



\- Python

\- Ollama

\- Qwen 2.5 Coder 3B

\- DeepEval

\- RAG Testing

\- LLM Evaluation



\## Test Coverage



\- Basic LLM response validation

\- Answer Relevancy

\- Faithfulness

\- Contextual Precision

\- Contextual Recall

\- Hallucination Testing

\- Bias Testing



\## Project Structure



\- `main.py` - Basic LLM integration

\- `test\_llm.py` - Answer relevancy testing

\- `test\_faithfulness.py` - Faithfulness evaluation

\- `test\_rag.py` - RAG evaluation

\- `test\_contextual\_precision.py` - Contextual precision

\- `test\_contextual\_recall.py` - Contextual recall

\- `test\_hallucination.py` - Hallucination detection

\- `test\_bias.py` - Bias evaluation



\## Setup



1\. Install Python and Ollama.

2\. Download the model:



&#x20;  `ollama pull qwen2.5-coder:3b`



3\. Create and activate a virtual environment.

4\. Install dependencies:



&#x20;  `pip install ollama deepeval requests`



5\. Configure DeepEval to use Ollama:



&#x20;  `deepeval set-ollama --model=qwen2.5-coder:3b --base-url=http://localhost:11434`



6\. Run individual tests:



&#x20;  `python test\_rag.py`



\## Project Goal



To explore practical QA techniques for validating AI-generated responses, RAG applications, and LLM evaluation metrics.



