
import csv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.models import OllamaModel


# ==========================================
# 1. Configure local evaluation model
# ==========================================

eval_model = OllamaModel(
    model="qwen2.5-coder:3b",
    base_url="http://localhost:11434",
    temperature=0
)


# ==========================================
# 2. Define RAG retrieval context
# ==========================================

context = [
    "Employees receive 20 annual leave days per year.",
    "Employees receive 10 sick leave days per year."
]


# ==========================================
# 3. Create multiple test scenarios
# ==========================================

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


# ==========================================
# 4. Configure evaluation metrics
# ==========================================

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


# ==========================================
# 5. Execute evaluation
# ==========================================

print("\nStarting RAG evaluation...\n")

result = evaluate(
    test_cases=test_cases,
    metrics=metrics
)


# ==========================================
# 6. Generate detailed CSV report
# ==========================================

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


# ==========================================
# 7. Calculate evaluation statistics
# ==========================================

total_cases = len(result.test_results)

passed_cases = sum(
    all(metric.success for metric in test.metrics_data)
    for test in result.test_results
)

failed_cases = total_cases - passed_cases

all_metrics = [
    metric
    for test in result.test_results
    for metric in test.metrics_data
]

total_metrics = len(all_metrics)

passed_metrics = sum(
    metric.success for metric in all_metrics
)

failed_metrics = total_metrics - passed_metrics

case_pass_rate = (
    (passed_cases / total_cases) * 100
    if total_cases else 0
)

metric_pass_rate = (
    (passed_metrics / total_metrics) * 100
    if total_metrics else 0
)


# ==========================================
# 8. Generate summary text report
# ==========================================

with open(
    "rag_evaluation_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("====================================\n")
    file.write("       RAG EVALUATION SUMMARY       \n")
    file.write("====================================\n\n")

    file.write("TEST CASE SUMMARY\n")
    file.write("------------------------------\n")
    file.write(f"Total Test Cases: {total_cases}\n")
    file.write(f"Passed Test Cases: {passed_cases}\n")
    file.write(f"Failed Test Cases: {failed_cases}\n")
    file.write(f"Test Case Pass Rate: {case_pass_rate:.2f}%\n\n")

    file.write("METRIC EVALUATION SUMMARY\n")
    file.write("------------------------------\n")
    file.write(f"Total Metric Evaluations: {total_metrics}\n")
    file.write(f"Passed Metrics: {passed_metrics}\n")
    file.write(f"Failed Metrics: {failed_metrics}\n")
    file.write(f"Metric Pass Rate: {metric_pass_rate:.2f}%\n")

    file.write("\n====================================\n")
    file.write("         END OF REPORT             \n")
    file.write("====================================\n")


# ==========================================
# 9. Display final summary in terminal
# ==========================================

print("\n====================================")
print("       RAG EVALUATION SUMMARY       ")
print("====================================")

print(f"Total Test Cases: {total_cases}")
print(f"Passed Test Cases: {passed_cases}")
print(f"Failed Test Cases: {failed_cases}")
print(f"Test Case Pass Rate: {case_pass_rate:.2f}%")

print("\nTotal Metric Evaluations:", total_metrics)
print("Passed Metrics:", passed_metrics)
print("Failed Metrics:", failed_metrics)
print(f"Metric Pass Rate: {metric_pass_rate:.2f}%")

print("\nEvaluation reports generated successfully!")
print("1. rag_evaluation_report.csv")
print("2. rag_evaluation_summary.txt")
