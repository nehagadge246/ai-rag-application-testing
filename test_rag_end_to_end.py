
# Lesson 18: RAG End-to-End Testing
# Positive and Negative Testing using Ollama + DeepEval

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

positive_output = response["response"].strip()


# 5. Configure local Ollama evaluation model
eval_model = OllamaModel(
    model="qwen2.5-coder:3b",
    base_url="http://localhost:11434",
    temperature=0
)


# 6. Evaluation function
def run_evaluation(test_name, actual_output):

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        retrieval_context=retrieved_context
    )

    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=eval_model
    )

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model=eval_model
    )

    print(f"\n===== {test_name} =====")

    print("\nGenerated Answer:")
    print(actual_output)

    # Answer Relevancy
    print("\n--- Answer Relevancy ---")

    answer_relevancy_metric.measure(test_case)

    relevancy_score = answer_relevancy_metric.score

    print("Score:", relevancy_score)

    # Faithfulness
    print("\n--- Faithfulness ---")

    faithfulness_metric.measure(test_case)

    faithfulness_score = faithfulness_metric.score

    print("Score:", faithfulness_score)

    return relevancy_score, faithfulness_score


# 7. Display retrieved context
print("\n===== RAG END-TO-END TEST =====")

print("\nUser Question:")
print(question)

print("\nRetrieved Context:")

for index, doc in enumerate(retrieved_context, start=1):
    print(f"{index}. {doc}")


# 8. Positive test scenario
positive_relevancy, positive_faithfulness = run_evaluation(
    "POSITIVE TEST",
    positive_output
)

if (
    positive_relevancy >= 0.7
    and positive_faithfulness >= 0.7
):
    print("\nPASS: Positive test passed.")
    positive_test_passed = True
else:
    print("\nFAIL: Positive test failed.")
    positive_test_passed = False


# 9. Negative test scenario
# Deliberately incorrect answer
negative_output = (
    "Employees receive 30 annual leave days per year."
)

negative_relevancy, negative_faithfulness = run_evaluation(
    "NEGATIVE TEST",
    negative_output
)

# The answer is relevant to the question,
# but its information contradicts the retrieved context.

if negative_faithfulness < 0.7:
    print("\nPASS: Negative test detected unsupported information.")
    negative_test_passed = True
else:
    print("\nFAIL: Negative test did not detect unsupported information.")
    negative_test_passed = False


# 10. Final evaluation summary
print("\n===== FINAL RAG TEST SUMMARY =====")

print("Positive Test:", "PASS" if positive_test_passed else "FAIL")
print("Negative Test:", "PASS" if negative_test_passed else "FAIL")

if positive_test_passed and negative_test_passed:
    print("\nOVERALL RESULT: PASS")
else:
    print("\nOVERALL RESULT: FAIL")
