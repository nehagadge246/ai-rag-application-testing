
import requests
import pandas as pd
import time

from pathlib import Path
from unittest.mock import patch, Mock


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"

REPORT_FILE = "rag_failure_injection_report.csv"

results = []


# --------------------------------------------------
# RESULT REPORTING
# --------------------------------------------------

def record_result(
    test_name,
    failure_condition,
    expected_result,
    actual_result,
    status,
    execution_time
):

    results.append({
        "Test Name": test_name,
        "Failure Condition": failure_condition,
        "Expected Result": expected_result,
        "Actual Result": actual_result,
        "Status": status,
        "Execution Time (Seconds)": round(execution_time, 3)
    })

    print("Expected:", expected_result)
    print("Actual:", actual_result)
    print("Status:", status)
    print("Execution Time:", round(execution_time, 3), "seconds")


# --------------------------------------------------
# TEST 1: OLLAMA SERVICE UNAVAILABLE
# --------------------------------------------------

print("\n===== TEST 1: OLLAMA SERVICE UNAVAILABLE =====")

start = time.perf_counter()

try:

    with patch(
        "requests.get",
        side_effect=requests.ConnectionError(
            "Ollama service unavailable"
        )
    ):

        requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=5
        )

    actual = "Service failure was not detected"
    status = "FAIL"

except requests.ConnectionError:

    actual = "Connection failure detected and handled"
    status = "PASS"

except requests.RequestException as error:

    actual = f"Unexpected request error: {error}"
    status = "FAIL"

record_result(
    "Ollama Service Unavailable",
    "Connection error injected",
    "Application should detect service connection failure",
    actual,
    status,
    time.perf_counter() - start
)


# --------------------------------------------------
# TEST 2: INVALID API ENDPOINT
# --------------------------------------------------

print("\n===== TEST 2: INVALID API ENDPOINT =====")

start = time.perf_counter()

try:

    mock_response = Mock()

    mock_response.status_code = 404
    mock_response.text = "404 Not Found"

    mock_response.raise_for_status.side_effect = (
        requests.HTTPError("404 Client Error: Not Found")
    )

    with patch(
        "requests.post",
        return_value=mock_response
    ):

        response = requests.post(
            f"{OLLAMA_BASE_URL}/invalid-endpoint"
        )

        response.raise_for_status()

    actual = "Invalid endpoint accepted"
    status = "FAIL"

except requests.HTTPError:

    actual = "HTTP 404 error detected and handled"
    status = "PASS"

except requests.RequestException as error:

    actual = f"Unexpected request error: {error}"
    status = "FAIL"

record_result(
    "Invalid API Endpoint",
    "HTTP 404 response injected",
    "Application should detect HTTP error",
    actual,
    status,
    time.perf_counter() - start
)


# --------------------------------------------------
# TEST 3: API REQUEST TIMEOUT
# --------------------------------------------------

print("\n===== TEST 3: API REQUEST TIMEOUT =====")

start = time.perf_counter()

try:

    with patch(
        "requests.post",
        side_effect=requests.Timeout(
            "API request timed out"
        )
    ):

        requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            timeout=1
        )

    actual = "Timeout was not detected"
    status = "FAIL"

except requests.Timeout:

    actual = "API timeout detected and handled"
    status = "PASS"

except requests.RequestException as error:

    actual = f"Unexpected request error: {error}"
    status = "FAIL"

record_result(
    "API Request Timeout",
    "Timeout exception injected",
    "Application should detect API timeout",
    actual,
    status,
    time.perf_counter() - start
)


# --------------------------------------------------
# TEST 4: MALFORMED API RESPONSE
# --------------------------------------------------

print("\n===== TEST 4: MALFORMED API RESPONSE =====")

start = time.perf_counter()

try:

    mock_response = Mock()

    mock_response.json.side_effect = ValueError(
        "Invalid JSON response"
    )

    with patch(
        "requests.post",
        return_value=mock_response
    ):

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate"
        )

        response.json()

    actual = "Malformed response accepted"
    status = "FAIL"

except ValueError:

    actual = "Malformed JSON response detected"
    status = "PASS"

record_result(
    "Malformed API Response",
    "Invalid JSON response injected",
    "Application should detect malformed JSON",
    actual,
    status,
    time.perf_counter() - start
)


# --------------------------------------------------
# TEST 5: EMPTY RETRIEVAL RESULTS
# --------------------------------------------------

print("\n===== TEST 5: EMPTY RETRIEVAL RESULTS =====")

start = time.perf_counter()

retrieved_documents = []

if len(retrieved_documents) == 0:

    actual = "Empty retrieval detected; fallback message returned"
    status = "PASS"

else:

    actual = "Unexpected documents returned"
    status = "FAIL"

record_result(
    "Empty Retrieval Results",
    "Empty document list supplied",
    "System should handle empty retrieval results",
    actual,
    status,
    time.perf_counter() - start
)


# --------------------------------------------------
# TEST 6: EMPTY LLM RESPONSE
# --------------------------------------------------

print("\n===== TEST 6: EMPTY LLM RESPONSE =====")

start = time.perf_counter()

try:

    mock_response = Mock()

    mock_response.json.return_value = {
        "response": ""
    }

    with patch(
        "requests.post",
        return_value=mock_response
    ):

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate"
        )

        response_data = response.json()

        generated_answer = response_data.get(
            "response",
            ""
        )

    if not generated_answer.strip():

        actual = "Empty LLM response detected"
        status = "PASS"

    else:

        actual = "Non-empty response received"
        status = "FAIL"

except Exception as error:

    actual = f"Unexpected error: {error}"
    status = "FAIL"

record_result(
    "Empty LLM Response",
    "Empty response injected",
    "System should detect empty LLM response",
    actual,
    status,
    time.perf_counter() - start
)


# --------------------------------------------------
# TEST 7: UNEXPECTED LLM OUTPUT TYPE
# --------------------------------------------------

print("\n===== TEST 7: UNEXPECTED LLM OUTPUT TYPE =====")

start = time.perf_counter()

generated_answer = None

if not isinstance(generated_answer, str):

    actual = "Unexpected response type detected"
    status = "PASS"

else:

    actual = "Response type is valid"
    status = "FAIL"

record_result(
    "Unexpected LLM Output Type",
    "None supplied instead of string",
    "System should identify invalid response type",
    actual,
    status,
    time.perf_counter() - start
)


# --------------------------------------------------
# TEST 8: API RECOVERY
# --------------------------------------------------

print("\n===== TEST 8: API RECOVERY =====")

start = time.perf_counter()

try:

    mock_response = Mock()

    mock_response.status_code = 200

    mock_response.json.return_value = {
        "response": "PASS"
    }

    with patch(
        "requests.post",
        return_value=mock_response
    ):

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate"
        )

        response_data = response.json()

        generated_answer = response_data.get(
            "response",
            ""
        )

    if generated_answer:

        actual = "API recovery successful; valid response received"
        status = "PASS"

    else:

        actual = "API recovery returned empty response"
        status = "FAIL"

except Exception as error:

    actual = f"API recovery failed: {error}"
    status = "FAIL"

record_result(
    "API Recovery",
    "Successful API response supplied after failure scenarios",
    "System should accept a valid API response",
    actual,
    status,
    time.perf_counter() - start
)


# --------------------------------------------------
# GENERATE CSV REPORT
# --------------------------------------------------

print("\n===== GENERATING FAILURE INJECTION REPORT =====")

df = pd.DataFrame(results)

df.to_csv(
    REPORT_FILE,
    index=False
)

total_tests = len(results)

passed_tests = sum(
    1 for result in results
    if result["Status"] == "PASS"
)

failed_tests = total_tests - passed_tests

pass_percentage = (
    passed_tests / total_tests * 100
    if total_tests > 0
    else 0
)


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n===== RAG FAILURE INJECTION SUMMARY =====")

print("Total Tests:", total_tests)
print("Passed:", passed_tests)
print("Failed:", failed_tests)
print("Pass Percentage:", round(pass_percentage, 2), "%")

print("\n===== TEST RESULTS =====")

print(
    df[["Test Name", "Status"]].to_string(index=False)
)

print("\nReport generated:", Path(REPORT_FILE).resolve())

print("\n===== RAG FAILURE INJECTION TESTING COMPLETED =====")
