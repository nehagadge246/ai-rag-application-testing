
import requests
import pandas as pd
import time
import statistics

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:3b"

REQUEST_TIMEOUT = 120

SEQUENTIAL_REQUESTS = 5
CONCURRENT_REQUESTS = 4
MAX_WORKERS = 2

LATENCY_TARGET_SECONDS = 60

REPORT_FILE = "rag_performance_load_report.csv"

results = []


# --------------------------------------------------
# GENERATE LLM RESPONSE
# --------------------------------------------------

def generate_response(request_id, test_type):

    prompt = (
        "Answer this question in two short sentences: "
        "What is Retrieval Augmented Generation (RAG)?"
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

        if isinstance(answer, str) and answer.strip():

            status = (
                "PASS"
                if elapsed <= LATENCY_TARGET_SECONDS
                else "REVIEW"
            )

            actual = "Valid LLM response received"

        else:

            status = "FAIL"
            actual = "Empty or invalid LLM response"

        return {
            "Test Type": test_type,
            "Request ID": request_id,
            "Response Time (Seconds)": round(elapsed, 3),
            "Status": status,
            "Actual Result": actual,
            "HTTP Status": response.status_code
        }

    except requests.Timeout:

        elapsed = time.perf_counter() - start

        return {
            "Test Type": test_type,
            "Request ID": request_id,
            "Response Time (Seconds)": round(elapsed, 3),
            "Status": "FAIL",
            "Actual Result": "Request timed out",
            "HTTP Status": None
        }

    except requests.RequestException as error:

        elapsed = time.perf_counter() - start

        return {
            "Test Type": test_type,
            "Request ID": request_id,
            "Response Time (Seconds)": round(elapsed, 3),
            "Status": "FAIL",
            "Actual Result": str(error),
            "HTTP Status": None
        }

    except (ValueError, TypeError) as error:

        elapsed = time.perf_counter() - start

        return {
            "Test Type": test_type,
            "Request ID": request_id,
            "Response Time (Seconds)": round(elapsed, 3),
            "Status": "FAIL",
            "Actual Result": f"Invalid response: {error}",
            "HTTP Status": None
        }


# --------------------------------------------------
# CHECK OLLAMA CONNECTION
# --------------------------------------------------

print("\n===== RAG PERFORMANCE AND LOAD TESTING =====")

print("\n===== STEP 1: CHECK OLLAMA CONNECTION =====")

try:

    health = requests.get(
        "http://localhost:11434/api/tags",
        timeout=5
    )

    health.raise_for_status()

    print("Ollama service: AVAILABLE")
    print("Model:", MODEL)

except requests.RequestException as error:

    print("Ollama connection failed:", error)
    raise SystemExit(1)


# --------------------------------------------------
# STEP 2: WARM-UP REQUEST
# --------------------------------------------------

print("\n===== STEP 2: MODEL WARM-UP =====")

warmup = generate_response(
    "WARMUP",
    "Warm-up"
)

print("Warm-up status:", warmup["Status"])
print("Warm-up time:", warmup["Response Time (Seconds)"], "seconds")

if warmup["Status"] == "FAIL":

    print("Warm-up failed. Stopping performance tests.")
    raise SystemExit(1)


# --------------------------------------------------
# STEP 3: SEQUENTIAL PERFORMANCE TEST
# --------------------------------------------------

print("\n===== STEP 3: SEQUENTIAL PERFORMANCE TEST =====")

sequential_start = time.perf_counter()

for number in range(1, SEQUENTIAL_REQUESTS + 1):

    result = generate_response(
        number,
        "Sequential"
    )

    results.append(result)

    print(
        f"Request {number}: "
        f"{result['Response Time (Seconds)']} seconds | "
        f"{result['Status']}"
    )

sequential_total_time = (
    time.perf_counter() - sequential_start
)


# --------------------------------------------------
# STEP 4: CONCURRENT LOAD TEST
# --------------------------------------------------

print("\n===== STEP 4: CONCURRENT LOAD TEST =====")

print("Concurrent requests:", CONCURRENT_REQUESTS)
print("Maximum workers:", MAX_WORKERS)

concurrent_start = time.perf_counter()

with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    futures = []

    for number in range(1, CONCURRENT_REQUESTS + 1):

        future = executor.submit(
            generate_response,
            number,
            "Concurrent"
        )

        futures.append(future)

    for future in as_completed(futures):

        result = future.result()

        results.append(result)

        print(
            f"Concurrent request {result['Request ID']}: "
            f"{result['Response Time (Seconds)']} seconds | "
            f"{result['Status']}"
        )

concurrent_total_time = (
    time.perf_counter() - concurrent_start
)


# --------------------------------------------------
# STEP 5: CALCULATE PERFORMANCE METRICS
# --------------------------------------------------

print("\n===== STEP 5: PERFORMANCE METRICS =====")

df = pd.DataFrame(results)

latencies = df["Response Time (Seconds)"].tolist()

successful = sum(
    1 for status in df["Status"]
    if status == "PASS"
)

reviewed = sum(
    1 for status in df["Status"]
    if status == "REVIEW"
)

failed = sum(
    1 for status in df["Status"]
    if status == "FAIL"
)

total_requests = len(results)

average_latency = statistics.mean(latencies)

minimum_latency = min(latencies)

maximum_latency = max(latencies)

sorted_latencies = sorted(latencies)

p95_index = max(
    0,
    int(0.95 * len(sorted_latencies) + 0.999999) - 1
)

p95_latency = sorted_latencies[p95_index]

success_rate = (
    successful / total_requests * 100
    if total_requests
    else 0
)

sequential_throughput = (
    SEQUENTIAL_REQUESTS / sequential_total_time
)

concurrent_throughput = (
    CONCURRENT_REQUESTS / concurrent_total_time
)

print("Total Requests:", total_requests)
print("Successful:", successful)
print("Review:", reviewed)
print("Failed:", failed)

print("Success Rate:", round(success_rate, 2), "%")

print("Average Latency:", round(average_latency, 3), "seconds")
print("Minimum Latency:", round(minimum_latency, 3), "seconds")
print("Maximum Latency:", round(maximum_latency, 3), "seconds")
print("P95 Latency:", round(p95_latency, 3), "seconds")

print(
    "Sequential Throughput:",
    round(sequential_throughput, 3),
    "requests/second"
)

print(
    "Concurrent Throughput:",
    round(concurrent_throughput, 3),
    "requests/second"
)


# --------------------------------------------------
# STEP 6: GENERATE CSV REPORT
# --------------------------------------------------

print("\n===== STEP 6: GENERATING REPORT =====")

df.to_csv(
    REPORT_FILE,
    index=False
)

print("Report generated:", Path(REPORT_FILE).resolve())


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n===== RAG PERFORMANCE TEST SUMMARY =====")

print("Sequential Requests:", SEQUENTIAL_REQUESTS)
print("Concurrent Requests:", CONCURRENT_REQUESTS)
print("Concurrent Workers:", MAX_WORKERS)

print("Total Requests:", total_requests)
print("Passed:", successful)
print("Review:", reviewed)
print("Failed:", failed)

print("\n===== TESTING COMPLETED =====")
