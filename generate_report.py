
import csv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.models import OllamaModel

# Configure local evaluation model
eval_model = OllamaModel(
    model="qwen2.5-coder:3b",
    base_url="http://localhost:11434",
    temperature=0
)

# RAG retrieval context
context = [
    "Employees receive 20 annual leave days per year.",
    "Employees receive 10 sick leave days per year."
]

# Multiple test scenarios
test_cases = [
    LLMTestCase(
        input="How many annual leave days do employees receive?",
        actual_output="Employees receive 20 annual leave days per year.",
        retrieval_context=context
    ),
    LLMTestCase(
        input="How many annual leave days do employees receive?",
        actual_output="Employees receive 30 annual leave days per year.",
        retrieval_context=context
    ),
    LLMTestCase(
        input="How many sick leave days do employees receive?",
        actual_output="Employees receive 10 sick leave days per year.",
        retrieval_context=context
    ),
    LLMTestCase(
        input="How many annual leave days do employees receive?",
        actual_output="Python is a programming language.",
        retrieval_context=context
    )
]

# Evaluation metrics
metrics = [
    AnswerRelevancyMetric(
        threshold=0.7,
        model=eval_model
    ),
    FaithfulnessMetric(
        threshold=0.7,
        model=eval_model
    )
]

# Execute evaluation
result = evaluate(
    test_cases=test_cases,
    metrics=metrics
)

# Export results to CSV
with open(
    "rag_evaluation_report.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Test Case",
        "Metric",
        "Score",
        "Threshold",
        "Status",
        "Reason"
    ])

    for index, test_result in enumerate(result.test_results):
        for metric in test_result.metrics_data:
            writer.writerow([
                f"Test Case {index + 1}",
                metric.name,
                metric.score,
                metric.threshold,
                "PASS" if metric.success else "FAIL",
                metric.reason
            ])

print("\nEvaluation report generated successfully!")
print("File: rag_evaluation_report.csv")
