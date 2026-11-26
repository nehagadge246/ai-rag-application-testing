
import requests
import pandas as pd
import time
import math

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ==================================================
# CONFIGURATION
# ==================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_HEALTH_URL = "http://localhost:11434/api/tags"

MODEL = "qwen2.5-coder:3b"

REQUEST_TIMEOUT = 120

CONCURRENCY_LEVELS = [1, 2, 3]
REQUESTS_PER_WORKER = 2

LATENCY_REVIEW_THRESHOLD = 60

DETAIL_REPORT = "rag_stress_scalability_report.csv"
SUMMARY_REPORT = "rag_stress_scalability_summary.csv"

all_results = []
summary_results = []


# ==================================================
# FUNCTION 1: GENERATE LLM RESPONSE
# ==================================================

def generate_response(request_id, concurrency_level):

    prompt = (
        "Answer in two short sentences: "
        "Explain how Retrieval Augmented Generation works."
    )

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 80
        }
    }

    start = time.perf_counter()

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get("response", "")

        elapsed = time.perf_counter() - start

        if not isinstance(answer, str) or not answer.strip():

            status = "FAIL"
            actual = "Empty or invalid LLM response"

        elif elapsed > LATENCY_REVIEW_THRESHOLD:

            status = "REVIEW"
            actual = "Valid response received but latency exceeded threshold"

        else:

            status = "PASS"
            actual = "Valid LLM response received"

        return {
            "Concurrency Level": concurrency_level,
            "Request ID": request_id,
            "Response Time (Seconds)": round(elapsed, 3),
            "Status": status,
            "Actual Result": actual
        }

    except requests.Timeout:

        elapsed = time.perf_counter() - start

        return {
            "Concurrency Level": concurrency_level,
            "Request ID": request_id,
            "Response Time (Seconds)": round(elapsed, 3),
            "Status": "FAIL",
            "Actual Result": "Request timed out"
        }

    except requests.RequestException as error:

        elapsed = time.perf_counter() - start

        return {
            "Concurrency Level": concurrency_level,
            "Request ID": request_id,
            "Response Time (Seconds)": round(elapsed, 3),
            "Status": "FAIL",
            "Actual Result": str(error)
        }

    except (ValueError, TypeError) as error:

        elapsed = time.perf_counter() - start

        return {
            "Concurrency Level": concurrency_level,
            "Request ID": request_id,
            "Response Time (Seconds)": round(elapsed, 3),
            "Status": "FAIL",
            "Actual Result": f"Invalid response: {error}"
        }

    except Exception as error:

        elapsed = time.perf_counter() - start

        return {
            "Concurrency Level": concurrency_level,
            "Request ID": request_id,
            "Response Time (Seconds)": round(elapsed, 3),
            "Status": "FAIL",
            "Actual Result": f"Unexpected error: {error}"
        }


# ==================================================
# FUNCTION 2: CHECK OLLAMA SERVICE
# ==================================================

print("\n===== RAG STRESS AND SCALABILITY TESTING =====")

print("\n===== STEP 1: CHECK OLLAMA SERVICE =====")

try:

    health = requests.get(
        OLLAMA_HEALTH_URL,
        timeout=5
    )

    health.raise_for_status()

    print("Ollama service: AVAILABLE")
    print("Model:", MODEL)

except requests.RequestException as error:

    print("Ollama connection failed:", error)
    raise SystemExit(1)


# ==================================================
# STEP 2: MODEL WARM-UP
# ==================================================

print("\n===== STEP 2: MODEL WARM-UP =====")

warmup = generate_response(
    "WARMUP",
    0
)

print("Warm-up status:", warmup["Status"])

print(
    "Warm-up response time:",
    warmup["Response Time (Seconds)"],
    "seconds"
)

if warmup["Status"] == "FAIL":

    print("Warm-up failed. Stopping tests.")
    raise SystemExit(1)


# ==================================================
# STEP 3: STRESS TEST AT DIFFERENT LOAD LEVELS
# ==================================================

for concurrency in CONCURRENCY_LEVELS:

    print(
        f"\n===== STRESS TEST: {concurrency} CONCURRENT WORKER(S) ====="
    )

    total_requests = concurrency * REQUESTS_PER_WORKER

    print("Concurrent workers:", concurrency)
    print("Total requests:", total_requests)

    level_start = time.perf_counter()

    level_results = []

    with ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = []

        for number in range(1, total_requests + 1):

            future = executor.submit(
                generate_response,
                number,
                concurrency
            )

            futures.append(future)

        for future in as_completed(futures):

            result = future.result()

            level_results.append(result)
            all_results.append(result)

            print(
                f"Request {result['Request ID']}: "
                f"{result['Response Time (Seconds)']} seconds | "
                f"{result['Status']}"
            )

    level_total_time = time.perf_counter() - level_start

    # ----------------------------------------------
    # CALCULATE LEVEL METRICS
    # ----------------------------------------------

    latencies = [
        item["Response Time (Seconds)"]
        for item in level_results
    ]

    sorted_latencies = sorted(latencies)

    average_latency = sum(latencies) / len(latencies)

    minimum_latency = min(latencies)

    maximum_latency = max(latencies)

    p95_index = max(
        0,
        math.ceil(0.95 * len(sorted_latencies)) - 1
    )

    p95_latency = sorted_latencies[p95_index]

    passed = sum(
        1 for item in level_results
        if item["Status"] == "PASS"
    )

    reviewed = sum(
        1 for item in level_results
        if item["Status"] == "REVIEW"
    )

    failed = sum(
        1 for item in level_results
        if item["Status"] == "FAIL"
    )

    throughput = (
        len(level_results) / level_total_time
        if level_total_time > 0
        else 0
    )

    success_rate = (
        passed / len(level_results) * 100
        if level_results
        else 0
    )

    summary_results.append({

        "Concurrency Level": concurrency,
        "Total Requests": len(level_results),
        "Passed": passed,
        "Review": reviewed,
        "Failed": failed,
        "Success Rate (%)": round(success_rate, 2),
        "Average Latency (Seconds)": round(average_latency, 3),
        "Minimum Latency (Seconds)": round(minimum_latency, 3),
        "Maximum Latency (Seconds)": round(maximum_latency, 3),
        "P95 Latency (Seconds)": round(p95_latency, 3),
        "Total Execution Time (Seconds)": round(level_total_time, 3),
        "Throughput (Requests/Second)": round(throughput, 3)

    })

    print("\nLevel Summary:")

    print("Average latency:", round(average_latency, 3), "seconds")
    print("Minimum latency:", round(minimum_latency, 3), "seconds")
    print("Maximum latency:", round(maximum_latency, 3), "seconds")
    print("P95 latency:", round(p95_latency, 3), "seconds")
    print("Passed:", passed)
    print("Review:", reviewed)
    print("Failed:", failed)
    print("Success rate:", round(success_rate, 2), "%")
    print("Throughput:", round(throughput, 3), "requests/second")


# ==================================================
# STEP 4: GENERATE CSV REPORTS
# ==================================================

print("\n===== STEP 4: GENERATING REPORTS =====")

detail_df = pd.DataFrame(all_results)

summary_df = pd.DataFrame(summary_results)

detail_df.to_csv(
    DETAIL_REPORT,
    index=False
)

summary_df.to_csv(
    SUMMARY_REPORT,
    index=False
)

print(
    "Detailed report:",
    Path(DETAIL_REPORT).resolve()
)

print(
    "Summary report:",
    Path(SUMMARY_REPORT).resolve()
)


# ==================================================
# STEP 5: FINAL TEST SUMMARY
# ==================================================

total_tests = len(all_results)

passed_tests = sum(
    1 for item in all_results
    if item["Status"] == "PASS"
)

review_tests = sum(
    1 for item in all_results
    if item["Status"] == "REVIEW"
)

failed_tests = sum(
    1 for item in all_results
    if item["Status"] == "FAIL"
)

print("\n===== FINAL STRESS TEST SUMMARY =====")

print("Concurrency levels:", CONCURRENCY_LEVELS)
print("Total requests:", total_tests)
print("Passed:", passed_tests)
print("Review:", review_tests)
print("Failed:", failed_tests)

print("\n===== SCALABILITY COMPARISON =====")

comparison_columns = [
    "Concurrency Level",
    "Average Latency (Seconds)",
    "P95 Latency (Seconds)",
    "Throughput (Requests/Second)",
    "Success Rate (%)"
]

print(
    summary_df[comparison_columns].to_string(index=False)
)

print("\n===== RAG STRESS AND SCALABILITY TESTING COMPLETED =====")
