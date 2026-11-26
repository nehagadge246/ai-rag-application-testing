
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

print("\n===== RAGAS MULTIPLE DATASET EVALUATION =====")

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
# 4. EVALUATION FUNCTION
# =====================================================

results = []


def evaluate_dataset(test_case):

    print("\n" + "=" * 65)
    print("SCENARIO:", test_case["scenario"])
    print("=" * 65)

    print("Question:", test_case["question"])
    print("Reference:", test_case["reference"])
    print("Response:", test_case["response"])

    print("\nRetrieved Contexts:")

    for context in test_case["contexts"]:
        print("-", context)

    scores = {}

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

    successful_scores = [
        score for score in scores.values()
        if score is not None
    ]

    average_score = (
        sum(successful_scores) / len(successful_scores)
        if successful_scores else None
    )

    results.append({
        "scenario": test_case["scenario"],
        "scores": scores,
        "average_score": average_score
    })

    print("\nAverage Score:", average_score)


# =====================================================
# 5. RUN ALL DATASETS
# =====================================================

for test_case in test_cases:
    evaluate_dataset(test_case)


# =====================================================
# 6. FINAL SUMMARY
# =====================================================

print("\n===== MULTIPLE DATASET SUMMARY =====")

for result in results:

    print("\nScenario:", result["scenario"])

    for metric_name, score in result["scores"].items():

        if score is not None:
            print(f"{metric_name}: {score:.4f}")
        else:
            print(f"{metric_name}: ERROR")

    print("Average Score:", result["average_score"])


# =====================================================
# 7. EXPORT CSV REPORT
# =====================================================

report_file = "ragas_multiple_datasets_report.csv"

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
        *metric_names,
        "Average Score",
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
            result["average_score"],
            execution_time
        ])


# =====================================================
# 8. EXECUTION COMPLETION
# =====================================================

duration = (
    datetime.now() - start_time
).total_seconds()

print("\nCSV Report Generated:", report_file)
print(f"Total Execution Time: {duration:.2f} seconds")

print("\n===== MULTIPLE DATASET EVALUATION COMPLETED =====")
