
# Lesson 18: RAG End-to-End Testing
# Actual answer generation using Ollama + DeepEval evaluation

import ollama

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.models import OllamaModel


# 1. Knowledge base
documents = [
    "Employees receive 20 annual leave days per year.",
    "Employees receive 10 sick leave days per year.",
    "Employees can work remotely twice a week.",
    "Office parking is available for employees."
]


# 2. User question
question = "How many annual leave days do employees receive?"


# 3. Retrieve relevant documents
retrieved_context = [
    doc for doc in documents
    if "annual leave" in doc.lower()
]


# 4. Generate actual answer using Ollama
prompt = f"""
Answer the question using only the provided context.

Context:
{chr(10).join(retrieved_context)}

Question:
{question}

Provide a clear and concise answer.
"""

response = ollama.generate(
    model="qwen2.5-coder:3b",
    prompt=prompt
)

actual_output = response["response"].strip()


# 5. Create DeepEval test case
test_case = LLMTestCase(
    input=question,
    actual_output=actual_output,
    retrieval_context=retrieved_context
)


# 6. Configure Ollama evaluation model
eval_model = OllamaModel(
    model="qwen2.5-coder:3b",
    base_url="http://localhost:11434",
    temperature=0
)


# 7. Configure Answer Relevancy metric
answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=eval_model
)


# 8. Configure Faithfulness metric
faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=eval_model
)


# 9. Display RAG workflow results
print("\n===== RAG END-TO-END TEST =====")

print("\nUser Question:")
print(question)

print("\nRetrieved Context:")
for index, doc in enumerate(retrieved_context, start=1):
    print(f"{index}. {doc}")

print("\nGenerated Answer:")
print(actual_output)


# 10. Evaluate Answer Relevancy
print("\n===== ANSWER RELEVANCY =====")

answer_relevancy_metric.measure(test_case)

print("Score:", answer_relevancy_metric.score)

if answer_relevancy_metric.score >= 0.7:
    print("PASS: Answer is relevant.")
else:
    print("FAIL: Answer is not sufficiently relevant.")


# 11. Evaluate Faithfulness
print("\n===== FAITHFULNESS =====")

faithfulness_metric.measure(test_case)

print("Score:", faithfulness_metric.score)

if faithfulness_metric.score >= 0.7:
    print("PASS: Answer is supported by retrieved context.")
else:
    print("FAIL: Answer contains unsupported information.")


# 12. Final evaluation summary
print("\n===== FINAL RAG TEST SUMMARY =====")

if (
    answer_relevancy_metric.score >= 0.7
    and faithfulness_metric.score >= 0.7
):
    print("OVERALL RESULT: PASS")
else:
    print("OVERALL RESULT: FAIL")
