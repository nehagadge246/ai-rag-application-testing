
import csv
from datetime import datetime

from openai import OpenAI, AsyncOpenAI
from instructor import Mode

from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    ContextRelevance
)


# =====================================================
# 1. CONFIGURATION
# =====================================================

print("\n===== INTEGRATED RAGAS EVALUATION SUITE =====")

start_time = datetime.now()

llm_client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

embedding_client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

evaluator_llm = llm_factory(
    model="qwen2.5-coder:3b",
    provider="openai",
    client=llm_client,
    mode=Mode.MD_JSON
)

evaluator_embeddings = OpenAIEmbeddings(
    client=embedding_client,
    model="nomic-embed-text:latest"
)


# =====================================================
# 2. INITIALIZE METRICS
# =====================================================

metrics = {
    "Faithfulness": Faithfulness(
        llm=evaluator_llm
    ),

    "Answer Relevancy": AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    ),

    "Context Precision": ContextPrecision(
        llm=evaluator_llm
    ),

    "Context Recall": ContextRecall(
        llm=evaluator_llm
    ),

    "Factual Correctness": FactualCorrectness(
        llm=evaluator_llm
    ),

    "Context Relevance": ContextRelevance(
        llm=evaluator_llm
    )
}


# =====================================================
# 3. TEST SCENARIOS AND EXPECTATIONS
# =====================================================

test_cases = [

    {
        "scenario": "Positive",

        "user_input": (
            "How many annual leave days do employees receive?"
        ),

        "reference": (
            "Employees receive 20 annual leave days per year."
        ),

        "response": (
            "Employees receive 20 annual leave days per year."
        ),

        "retrieved_contexts": [
            "Employees receive 20 annual leave days per year.",
            "Employees receive 10 sick leave days per year."
        ],

        "expectations": {
            "Faithfulness": {"min": 0.70},
            "Answer Relevancy": {"min": 0.70},
            "Context Precision": {"min": 0.70},
            "Context Recall": {"min": 0.50},
            "Factual Correctness": {"min": 0.70},
            "Context Relevance": {"min": 0.70}
        }
    },

    {
        "scenario": "Negative",

        "user_input": (
            "How many annual leave days do employees receive?"
        ),

        "reference": (
            "Employees receive 20 annual leave days per year."
        ),

        "response": (
            "Employees receive 30 annual leave days per year."
        ),

        "retrieved_contexts": [
            "The company organizes annual sports competitions.",
            "Employees participate in football and cricket tournaments."
        ],

        "expectations": {
            "Faithfulness": {"max": 0.20},
            "Answer Relevancy": {"min": 0.70},
            "Context Precision": {"max": 0.20},
            "Context Recall": {"max": 0.20},
            "Factual Correctness": {"max": 0.20},
            "Context Relevance": {"max": 0.20}
        }
    }
]


# =====================================================
# 4. METRIC VALIDATION FUNCTION
# =====================================================

def validate_metric(metric_name, score, expectations):

    if score is None:
        return "FAIL"

    rule = expectations[metric_name]

    if "min" in rule:
        return "PASS" if score >= rule["min"] else "FAIL"

    if "max" in rule:
        return "PASS" if score <= rule["max"] else "FAIL"

    return "FAIL"


# =====================================================
# 5. EVALUATION FUNCTION
# =====================================================

results = []


def evaluate_scenario(test_case):

    print("\n" + "=" * 65)
    print("SCENARIO:", test_case["scenario"])
    print("=" * 65)

    print("Question:", test_case["user_input"])
    print("Reference:", test_case["reference"])
    print("Generated Answer:", test_case["response"])

    print("\nRetrieved Contexts:")

    for context in test_case["retrieved_contexts"]:
        print("-", context)

    scenario_scores = {}
    scenario_statuses = {}

    for metric_name, metric in metrics.items():

        print(f"\nRunning {metric_name}...")

        try:

            if metric_name == "Faithfulness":

                result = metric.score(
                    user_input=test_case["user_input"],
                    response=test_case["response"],
                    retrieved_contexts=test_case["retrieved_contexts"]
                )

            elif metric_name == "Answer Relevancy":

                result = metric.score(
                    user_input=test_case["user_input"],
                    response=test_case["response"]
                )

            elif metric_name == "Context Precision":

                result = metric.score(
                    user_input=test_case["user_input"],
                    reference=test_case["reference"],
                    retrieved_contexts=test_case["retrieved_contexts"]
                )

            elif metric_name == "Context Recall":

                result = metric.score(
                    user_input=test_case["user_input"],
                    reference=test_case["reference"],
                    retrieved_contexts=test_case["retrieved_contexts"]
                )

            elif metric_name == "Factual Correctness":

                result = metric.score(
                    response=test_case["response"],
                    reference=test_case["reference"]
                )

            elif metric_name == "Context Relevance":

                result = metric.score(
                    user_input=test_case["user_input"],
                    retrieved_contexts=test_case["retrieved_contexts"]
                )

            score = float(result.value)

            status = validate_metric(
                metric_name,
                score,
                test_case["expectations"]
            )

            scenario_scores[metric_name] = score
            scenario_statuses[metric_name] = status

            print(f"Score: {score:.4f}")
            print(f"Validation: {status}")

        except Exception as error:

            scenario_scores[metric_name] = None
            scenario_statuses[metric_name] = "ERROR"

            print(f"ERROR: {error}")

    passed_metrics = sum(
        1 for status in scenario_statuses.values()
        if status == "PASS"
    )

    failed_metrics = sum(
        1 for status in scenario_statuses.values()
        if status == "FAIL"
    )

    error_metrics = sum(
        1 for status in scenario_statuses.values()
        if status == "ERROR"
    )

    scenario_status = (
        "PASS"
        if passed_metrics == len(metrics)
        else "FAIL"
    )

    successful_scores = [
        score for score in scenario_scores.values()
        if score is not None
    ]

    average_score = (
        sum(successful_scores) / len(successful_scores)
        if successful_scores else None
    )

    print("\n----- SCENARIO VALIDATION SUMMARY -----")
    print("Passed Metrics:", passed_metrics)
    print("Failed Metrics:", failed_metrics)
    print("Metric Errors:", error_metrics)
    print("Scenario Result:", scenario_status)

    results.append({

        "scenario": test_case["scenario"],
        "scores": scenario_scores,
        "statuses": scenario_statuses,
        "passed_metrics": passed_metrics,
        "failed_metrics": failed_metrics,
        "error_metrics": error_metrics,
        "scenario_status": scenario_status,
        "average_score": average_score
    })


# =====================================================
# 6. RUN ALL TEST SCENARIOS
# =====================================================

for test_case in test_cases:
    evaluate_scenario(test_case)


# =====================================================
# 7. CONSOLIDATED EXECUTION SUMMARY
# =====================================================

print("\n\n===== FINAL RAGAS EXECUTION SUMMARY =====")

total_scenarios = len(results)

passed_scenarios = sum(
    1 for result in results
    if result["scenario_status"] == "PASS"
)

failed_scenarios = total_scenarios - passed_scenarios

for result in results:

    print("\nScenario:", result["scenario"])
    print("Result:", result["scenario_status"])

    for metric_name in metrics:

        score = result["scores"][metric_name]
        status = result["statuses"][metric_name]

        if score is not None:
            print(
                f"{metric_name}: {score:.4f} - {status}"
            )
        else:
            print(f"{metric_name}: ERROR")

print("\nTotal Scenarios:", total_scenarios)
print("Passed Scenarios:", passed_scenarios)
print("Failed Scenarios:", failed_scenarios)


# =====================================================
# 8. EXPORT DETAILED CSV REPORT
# =====================================================

report_file = "ragas_integrated_report.csv"

metric_names = list(metrics.keys())

with open(
    report_file,
    mode="w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    headers = ["Scenario"]

    for metric_name in metric_names:
        headers.extend([
            f"{metric_name} Score",
            f"{metric_name} Status"
        ])

    headers.extend([
        "Average Score",
        "Passed Metrics",
        "Failed Metrics",
        "Metric Errors",
        "Scenario Result",
        "Execution Time"
    ])

    writer.writerow(headers)

    execution_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    for result in results:

        row = [result["scenario"]]

        for metric_name in metric_names:

            row.extend([
                result["scores"].get(metric_name),
                result["statuses"].get(metric_name)
            ])

        row.extend([
            result["average_score"],
            result["passed_metrics"],
            result["failed_metrics"],
            result["error_metrics"],
            result["scenario_status"],
            execution_time
        ])

        writer.writerow(row)


# =====================================================
# 9. EXECUTION COMPLETION
# =====================================================

end_time = datetime.now()

duration = (end_time - start_time).total_seconds()

print("\nCSV Report Generated:", report_file)
print(f"Total Execution Time: {duration:.2f} seconds")

print("\n===== RAGAS EVALUATION COMPLETED =====")
