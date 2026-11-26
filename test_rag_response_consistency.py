
import requests
import time
import csv
import re
from difflib import SequenceMatcher
from datetime import datetime

# ==========================================
# LESSON 51: RAG RESPONSE CONSISTENCY TESTING
# ==========================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:3b"

TOTAL_RUNS = 5
SIMILARITY_THRESHOLD = 0.70
REQUEST_TIMEOUT = 120

QUESTION = "How many annual leave days do employees receive?"

PROMPT = f"""
You are an HR policy assistant.

Answer the following question clearly and consistently.
If the policy information is not available, say that it is not specified.

Question: {QUESTION}

Provide a concise answer.
"""

REPORT_FILE = "rag_response_consistency_report.csv"
SUMMARY_FILE = "rag_response_consistency_summary.csv"

results = []
answers = []

print("=" * 65)
print("RAG RESPONSE CONSISTENCY AND REPEATABILITY TESTING")
print("=" * 65)

print("\nModel:", MODEL)
print("Question:", QUESTION)
print("Total test runs:", TOTAL_RUNS)


# ==========================================
# STEP 1: CHECK OLLAMA CONNECTION
# ==========================================

print("\n===== STEP 1: CHECK OLLAMA CONNECTION =====")

try:
    response = requests.get(
        "http://localhost:11434/api/tags",
        timeout=10
    )
    response.raise_for_status()

    available_models = [
        model.get("name", "")
        for model in response.json().get("models", [])
    ]

    if not any(
        name == MODEL or name.startswith(MODEL + ":")
        for name in available_models
    ):
        raise RuntimeError(
            f"Model {MODEL} is not available in Ollama."
        )

    print("Ollama connection: PASS")
    print("Model availability: PASS")

except Exception as error:
    print("Ollama connection: FAIL")
    print("Error:", error)
    raise SystemExit(1)


# ==========================================
# STEP 2: WARM-UP REQUEST
# ==========================================

print("\n===== STEP 2: MODEL WARM-UP =====")

try:
    warmup_response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": "Reply with the word READY.",
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        },
        timeout=REQUEST_TIMEOUT
    )

    warmup_response.raise_for_status()

    warmup_answer = warmup_response.json().get(
        "response", ""
    ).strip()

    if not warmup_answer:
        raise ValueError("Warm-up returned an empty response.")

    print("Warm-up: PASS")
    print("Warm-up response:", warmup_answer)

except Exception as error:
    print("Warm-up: FAIL")
    print("Error:", error)
    raise SystemExit(1)


# ==========================================
# STEP 3: REPEATED RESPONSE GENERATION
# ==========================================

print("\n===== STEP 3: REPEATED RESPONSE TESTING =====")

for run_number in range(1, TOTAL_RUNS + 1):

    print(f"\n----- TEST RUN {run_number} -----")

    start_time = time.perf_counter()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "seed": 42
                }
            },
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        response_data = response.json()
        answer = response_data.get("response", "").strip()

        latency = round(
            time.perf_counter() - start_time, 3
        )

        if not answer:
            status = "FAIL"
            error_message = "Empty response"
            response_length = 0
        else:
            status = "PASS"
            error_message = ""
            response_length = len(answer)
            answers.append(answer)

        results.append({
            "Run": run_number,
            "Question": QUESTION,
            "Generated Answer": answer,
            "Response Time (sec)": latency,
            "Response Length": response_length,
            "Similarity (%)": "",
            "Status": status,
            "Error": error_message
        })

        print("Question:", QUESTION)
        print("Answer:", answer)
        print("Response time:", latency, "seconds")
        print("Response length:", response_length)
        print("Status:", status)

    except Exception as error:

        latency = round(
            time.perf_counter() - start_time, 3
        )

        results.append({
            "Run": run_number,
            "Question": QUESTION,
            "Generated Answer": "",
            "Response Time (sec)": latency,
            "Response Length": 0,
            "Similarity (%)": "",
            "Status": "FAIL",
            "Error": str(error)
        })

        print("Status: FAIL")
        print("Error:", error)


# ==========================================
# STEP 4: RESPONSE SIMILARITY ANALYSIS
# ==========================================

print("\n===== STEP 4: RESPONSE CONSISTENCY ANALYSIS =====")

similarity_scores = []

if len(answers) >= 2:

    baseline_answer = answers[0]

    for index, answer in enumerate(answers):

        similarity = SequenceMatcher(
            None,
            baseline_answer.lower(),
            answer.lower()
        ).ratio()

        similarity_percentage = round(
            similarity * 100, 2
        )

        similarity_scores.append(similarity_percentage)

        print(
            f"Response {index + 1} similarity to baseline: "
            f"{similarity_percentage}%"
        )

    average_similarity = round(
        sum(similarity_scores) / len(similarity_scores),
        2
    )

    minimum_similarity = round(
        min(similarity_scores), 2
    )

    # Baseline is compared with itself, so exclude it
    # when calculating repeat-run consistency.
    repeat_scores = similarity_scores[1:]

    average_repeat_similarity = round(
        sum(repeat_scores) / len(repeat_scores),
        2
    )

else:
    average_similarity = 0
    minimum_similarity = 0
    average_repeat_similarity = 0

    print("Not enough successful responses for comparison.")


# Add similarity values to successful run records.
successful_results = [
    item for item in results
    if item["Status"] == "PASS"
]

for index, item in enumerate(successful_results):

    if index < len(similarity_scores):
        item["Similarity (%)"] = similarity_scores[index]


# ==========================================
# STEP 5: CONSISTENCY STATUS
# ==========================================

print("\n===== STEP 5: CONSISTENCY VALIDATION =====")

passed_runs = sum(
    1 for item in results
    if item["Status"] == "PASS"
)

failed_runs = TOTAL_RUNS - passed_runs

if passed_runs == TOTAL_RUNS:
    response_status = "PASS"
else:
    response_status = "REVIEW"

if (
    passed_runs >= 2
    and average_repeat_similarity
    >= SIMILARITY_THRESHOLD * 100
):
    consistency_status = "PASS"
else:
    consistency_status = "REVIEW"

print("Total runs:", TOTAL_RUNS)
print("Successful runs:", passed_runs)
print("Failed runs:", failed_runs)
print("Average repeat similarity:", average_repeat_similarity, "%")
print("Minimum similarity:", minimum_similarity, "%")
print("Response validity:", response_status)
print("Consistency status:", consistency_status)


# ==========================================
# STEP 6: GENERATE DETAILED CSV REPORT
# ==========================================

print("\n===== STEP 6: GENERATING DETAILED REPORT =====")

fieldnames = [
    "Run",
    "Question",
    "Generated Answer",
    "Response Time (sec)",
    "Response Length",
    "Similarity (%)",
    "Status",
    "Error"
]

with open(
    REPORT_FILE,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)

print("Detailed report generated:", REPORT_FILE)


# ==========================================
# STEP 7: GENERATE SUMMARY CSV REPORT
# ==========================================

print("\n===== STEP 7: GENERATING SUMMARY REPORT =====")

latencies = [
    item["Response Time (sec)"]
    for item in results
    if item["Status"] == "PASS"
]

average_latency = (
    round(sum(latencies) / len(latencies), 3)
    if latencies else 0
)

summary = {
    "Execution Date": datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
    "Model": MODEL,
    "Question": QUESTION,
    "Total Runs": TOTAL_RUNS,
    "Passed Runs": passed_runs,
    "Failed Runs": failed_runs,
    "Average Latency (sec)": average_latency,
    "Average Repeat Similarity (%)": average_repeat_similarity,
    "Minimum Similarity (%)": minimum_similarity,
    "Response Validity": response_status,
    "Consistency Status": consistency_status
}

with open(
    SUMMARY_FILE,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(summary.keys())
    )

    writer.writeheader()
    writer.writerow(summary)

print("Summary report generated:", SUMMARY_FILE)


# ==========================================
# FINAL RESULTS
# ==========================================

print("\n" + "=" * 65)
print("RAG RESPONSE CONSISTENCY TESTING COMPLETED")
print("=" * 65)

print("Total test runs:", TOTAL_RUNS)
print("Passed:", passed_runs)
print("Failed:", failed_runs)
print("Average latency:", average_latency, "seconds")
print(
    "Average repeat similarity:",
    average_repeat_similarity,
    "%"
)
print("Response validity:", response_status)
print("Consistency status:", consistency_status)

print("\nGenerated files:")
print(REPORT_FILE)
print(SUMMARY_FILE)

print("\nNOTE:")
print(
    "Similarity uses text comparison (SequenceMatcher), "
    "not semantic similarity."
)
print(
    "The test measures repeated local LLM generation, "
    "not a complete production RAG retrieval pipeline."
)
print("=" * 65)
