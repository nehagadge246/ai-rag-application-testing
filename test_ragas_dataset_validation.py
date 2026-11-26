
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

print("\n===== RAGAS AUTOMATED DATASET VALIDATION =====")

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
    "Faithfulness": Faithfulness(llm=evaluator_llm),

    "Answer Relevancy": AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    ),

    "Context Precision": ContextPrecision(llm=evaluator_llm),

    "Context Recall": ContextRecall(llm=evaluator_llm),

    "Factual Correctness": FactualCorrectness(
        llm=evaluator_llm
    ),

    "Context Relevance": ContextRelevance(
        llm=evaluator_llm
    )
}


# =====================================================
# 3. MULTIPLE TEST DATASETS
# =====================================================

test_cases = [

    {
        "scenario": "Correct Answer",
        "question": "How many annual leave days do employees receive?",
        "reference": "Employees receive 20 annual leave days per year.",
        "response": "Employees receive 20 annual leave days per year.",
        "contexts": [
            "Employees receive 20 annual leave days per year."
        ]
    },

    {
        "scenario": "Incorrect Answer",
        "question": "How many annual leave days do employees receive?",
        "reference": "Employees receive 20 annual leave days per year.",
        "response": "Employees receive 30 annual leave days per year.",
        "contexts": [
            "Employees receive 20 annual leave days per year."
        ]
    },

    {
        "scenario": "Irrelevant Context",
        "question": "How many annual leave days do employees receive?",
        "reference": "Employees receive 20 annual leave days per year.",
        "response": "Employees receive 20 annual leave days per year.",
        "contexts": [
            "The company organizes annual sports competitions.",
            "Employees participate in football tournaments."
        ]
    },

    {
        "scenario": "Hallucinated Answer",
        "question": "How many annual leave days do employees receive?",
        "reference": "Employees receive 20 annual leave days per year.",
        "response": "Employees receive 20 annual leave days and 15 bonus leave days.",
        "contexts": [
            "Employees receive 20 annual leave days per year."
        ]
    },

    {
        "scenario": "Missing Information",
        "question": "How many annual leave days do employees receive?",
        "reference": "Employees receive 20 annual leave days per year.",
        "response": "The available information does not specify the annual leave entitlement.",
        "contexts": [
            "Employees can request leave through the HR portal.",
            "Leave requests require manager approval."
        ]
    }
]


# =====================================================
# 4. SCENARIO-SPECIFIC VALIDATION RULES
# =====================================================

# "min" means the score must be >= threshold.
# "max" means the score must be <= threshold.
#
# Context Recall is diagnostic-only because its
# reliability is still under investigation.

validation_rules = {

    "Correct Answer": {
        "Faithfulness": ("min", 0.70),
        "Answer Relevancy": ("min", 0.70),
        "Context Precision": ("min", 0.70),
        "Factual Correctness": ("min", 0.70),
        "Context Relevance": ("min", 0.70)
    },

    "Incorrect Answer": {
        "Faithfulness": ("max", 0.20),
        "Factual Correctness": ("max", 0.30)
    },

    "Irrelevant Context": {
        "Context Precision": ("max", 0.30),
        "Context Relevance": ("max", 0.30)
    },

    "Hallucinated Answer": {
        "Faithfulness": ("max", 0.30),
        "Factual Correctness": ("max", 0.50)
    },

    "Missing Information": {
        "Faithfulness": ("max", 0.20),
        "Answer Relevancy": ("max", 0.50),
        "Context Precision": ("max", 0.30),
        "Context Relevance": ("max", 0.30),
        "Factual Correctness": ("max", 0.30)
    }
}

diagnostic_metrics = ["Context Recall"]


# =====================================================
# 5. EVALUATION FUNCTION
# =====================================================

results = []


def evaluate_dataset(test_case):

    scenario = test_case["scenario"]
    rules = validation_rules[scenario]

    print("\n" + "=" * 65)
    print("SCENARIO:", scenario)
    print("=" * 65)

    print("Question:", test_case["question"])
    print("Reference:", test_case["reference"])
    print("Response:", test_case["response"])

    print("\nRetrieved Contexts:")

    for context in test_case["contexts"]:
        print("-", context)

    scores = {}

    # -------------------------------------------------
    # RUN ALL RAGAS METRICS
    # -------------------------------------------------

    for metric_name, metric in metrics.items():

        print(f"\nRunning {metric_name}...")

        try:

            if metric_name == "Faithfulness":

                result = metric.score(
                    user_input=test_case["question"],
                    response=test_case["response"],
                    retrieved_contexts=test_case["contexts"]
                )

            elif metric_name == "Answer Relevancy":

                result = metric.score(
                    user_input=test_case["question"],
                    response=test_case["response"]
                )

            elif metric_name in [
                "Context Precision",
                "Context Recall"
            ]:

                result = metric.score(
                    user_input=test_case["question"],
                    reference=test_case["reference"],
                    retrieved_contexts=test_case["contexts"]
                )

            elif metric_name == "Factual Correctness":

                result = metric.score(
                    response=test_case["response"],
                    reference=test_case["reference"]
                )

            elif metric_name == "Context Relevance":

                result = metric.score(
                    user_input=test_case["question"],
                    retrieved_contexts=test_case["contexts"]
                )

            score = float(result.value)

            scores[metric_name] = score

            print(f"{metric_name}: {score:.4f}")

        except Exception as error:

            scores[metric_name] = None

            print(f"{metric_name} ERROR: {error}")

    # -------------------------------------------------
    # AUTOMATED METRIC-LEVEL VALIDATION
    # -------------------------------------------------

    metric_statuses = {}

    for metric_name, score in scores.items():

        # Diagnostic metrics are not used for verdicts.
        if metric_name in diagnostic_metrics:

            metric_statuses[metric_name] = "REVIEW"

        elif metric_name not in rules:

            metric_statuses[metric_name] = "NOT APPLICABLE"

        elif score is None:

            metric_statuses[metric_name] = "REVIEW"

        else:

            condition, threshold = rules[metric_name]

            if condition == "min":
                passed = score >= threshold

            elif condition == "max":
                passed = score <= threshold

            else:
                passed = False

            metric_statuses[metric_name] = (
                "PASS" if passed else "FAIL"
            )

    # -------------------------------------------------
    # OVERALL SCENARIO VALIDATION
    # -------------------------------------------------

    # Only metrics with configured rules affect the
    # overall result. Context Recall remains diagnostic.

    applicable_statuses = [
        metric_statuses[name]
        for name in rules
    ]

    if "FAIL" in applicable_statuses:

        overall_status = "FAIL"

    elif "REVIEW" in applicable_statuses:

        overall_status = "REVIEW"

    elif applicable_statuses and all(
        status == "PASS"
        for status in applicable_statuses
    ):

        overall_status = "PASS"

    else:

        overall_status = "REVIEW"

    # -------------------------------------------------
    # CALCULATE AVERAGE SCORE
    # -------------------------------------------------

    successful_scores = [
        score
        for score in scores.values()
        if score is not None
    ]

    average_score = (
        sum(successful_scores) / len(successful_scores)
        if successful_scores else None
    )

    # -------------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------------

    print("\n----- AUTOMATED VALIDATION -----")

    for metric_name, status in metric_statuses.items():

        score = scores[metric_name]

        if score is not None:
            print(
                f"{metric_name}: {score:.4f} | {status}"
            )
        else:
            print(f"{metric_name}: ERROR | {status}")

    print("\nOverall Scenario Status:", overall_status)

    results.append({
        "scenario": scenario,
        "scores": scores,
        "metric_statuses": metric_statuses,
        "overall_status": overall_status,
        "average_score": average_score
    })


# =====================================================
# 6. RUN ALL DATASETS
# =====================================================

for test_case in test_cases:
    evaluate_dataset(test_case)


# =====================================================
# 7. FINAL VALIDATION SUMMARY
# =====================================================

print("\n===== FINAL AUTOMATED VALIDATION SUMMARY =====")

for result in results:

    print("\nScenario:", result["scenario"])

    for metric_name, status in result["metric_statuses"].items():

        score = result["scores"].get(metric_name)

        if score is not None:

            print(
                f"{metric_name}: {score:.4f} | {status}"
            )

        else:

            print(f"{metric_name}: ERROR | {status}")

    print("Average Score:", result["average_score"])
    print("Overall Status:", result["overall_status"])


# =====================================================
# 8. VALIDATION STATISTICS
# =====================================================

passed_scenarios = sum(
    1 for result in results
    if result["overall_status"] == "PASS"
)

failed_scenarios = sum(
    1 for result in results
    if result["overall_status"] == "FAIL"
)

review_scenarios = sum(
    1 for result in results
    if result["overall_status"] == "REVIEW"
)

print("\n===== VALIDATION STATISTICS =====")

print("Total Scenarios:", len(results))
print("Passed Scenarios:", passed_scenarios)
print("Failed Scenarios:", failed_scenarios)
print("Review Scenarios:", review_scenarios)


# =====================================================
# 9. EXPORT VALIDATION CSV REPORT
# =====================================================

report_file = "ragas_dataset_validation_report.csv"

metric_names = list(metrics.keys())

with open(
    report_file,
    mode="w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Scenario",

        *[
            f"{name} Score"
            for name in metric_names
        ],

        *[
            f"{name} Status"
            for name in metric_names
        ],

        "Average Score",
        "Overall Status",
        "Execution Time"
    ])

    execution_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    for result in results:

        writer.writerow([
            result["scenario"],

            *[
                result["scores"].get(name)
                for name in metric_names
            ],

            *[
                result["metric_statuses"].get(name)
                for name in metric_names
            ],

            result["average_score"],
            result["overall_status"],
            execution_time
        ])


# =====================================================
# 10. EXECUTION COMPLETION
# =====================================================

duration = (
    datetime.now() - start_time
).total_seconds()

print("\nCSV Report Generated:", report_file)

print(f"Total Execution Time: {duration:.2f} seconds")

print("\n===== AUTOMATED DATASET VALIDATION COMPLETED =====")
