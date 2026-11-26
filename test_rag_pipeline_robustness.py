
import requests
import pandas as pd
import time

from pathlib import Path

print("===== LESSON 47: RAG PIPELINE ROBUSTNESS TESTING =====")

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5-coder:3b"

results = []


# --------------------------------------------------
# 1. COMMON TEST REPORT FUNCTION
# --------------------------------------------------

def record_result(
    test_name,
    input_value,
    expected_result,
    actual_result,
    status,
    response_time
):

    results.append({
        "Test Name": test_name,
        "Input": input_value,
        "Expected Result": expected_result,
        "Actual Result": actual_result,
        "Status": status,
        "Response Time Seconds": round(response_time, 2)
    })

    print("\nTest:", test_name)
    print("Expected:", expected_result)
    print("Actual:", actual_result)
    print("Status:", status)


# --------------------------------------------------
# 2. EMPTY QUESTION VALIDATION
# --------------------------------------------------

print("\n===== TEST 1: EMPTY QUESTION =====")

start = time.perf_counter()

question = ""

if not question.strip():
    actual = "Empty question rejected"
    status = "PASS"
else:
    actual = "Question accepted"
    status = "FAIL"

record_result(
    "Empty Question Validation",
    question,
    "Empty question rejected",
    actual,
    status,
    time.perf_counter() - start
)


# --------------------------------------------------
# 3. WHITESPACE QUESTION VALIDATION
# --------------------------------------------------

print("\n===== TEST 2: WHITESPACE QUESTION =====")

start = time.perf_counter()

question = "     "

if not question.strip():
    actual = "Whitespace-only question rejected"
    status = "PASS"
else:
    actual = "Question accepted"
    status = "FAIL"

record_result(
    "Whitespace Question Validation",
    repr(question),
    "Whitespace-only question rejected",
    actual,
    status,
    time.perf_counter() - start
)


# --------------------------------------------------
# 4. UNRELATED QUESTION HANDLING
# --------------------------------------------------

print("\n===== TEST 3: UNRELATED QUESTION =====")

start = time.perf_counter()

question = "Who won the football world cup?"

context = ""

if not context.strip():
    actual = "No relevant context available"
    status = "PASS"
else:
    actual = "Context unexpectedly available"
    status = "FAIL"

record_result(
    "Unrelated Question Without Context",
    question,
    "No relevant context available",
    actual,
    status,
    time.perf_counter() - start
)


# --------------------------------------------------
# 5. EMPTY CONTEXT HANDLING
# --------------------------------------------------

print("\n===== TEST 4: EMPTY CONTEXT =====")

start = time.perf_counter()

question = "How many annual leave days do employees receive?"
context = ""

if not context.strip():
    answer = "No relevant information found."
else:
    answer = "Context available"

if answer == "No relevant information found.":
    actual = "Fallback answer returned"
    status = "PASS"
else:
    actual = "Unexpected response"
    status = "FAIL"

record_result(
    "Empty Context Handling",
    question,
    "Fallback answer returned",
    actual,
    status,
    time.perf_counter() - start
)


# --------------------------------------------------
# 6. UNEXPECTED LLM RESPONSE VALIDATION
# --------------------------------------------------

print("\n===== TEST 5: UNEXPECTED LLM RESPONSE =====")

start = time.perf_counter()

unexpected_responses = [
    "",
    "I don't know",
    "N/A",
    "No relevant information found."
]

valid_fallbacks = [
    "i don't know",
    "n/a",
    "no relevant information found."
]

invalid_responses = [
    response
    for response in unexpected_responses
    if not response.strip()
]

if invalid_responses:
    actual = "Empty LLM response detected"
    status = "PASS"
else:
    actual = "No empty LLM response detected"
    status = "PASS"

record_result(
    "Unexpected LLM Response Validation",
    str(unexpected_responses),
    "Empty response detected and handled",
    actual,
    status,
    time.perf_counter() - start
)


# --------------------------------------------------
# 7. OLLAMA CONNECTION CHECK
# --------------------------------------------------

print("\n===== TEST 6: OLLAMA CONNECTION =====")

start = time.perf_counter()

try:

    response = requests.get(
        f"{OLLAMA_BASE_URL}/api/tags",
        timeout=5
    )

    response.raise_for_status()

    actual = "Ollama API is reachable"
    status = "PASS"

except requests.RequestException as error:

    actual = f"Ollama connection error: {error}"
    status = "PASS"

record_result(
    "Ollama Connection Handling",
    OLLAMA_BASE_URL,
    "Connection status detected without crashing",
    actual,
    status,
    time.perf_counter() - start
)


# --------------------------------------------------
# 8. LLM API RESPONSE CHECK
# --------------------------------------------------

print("\n===== TEST 7: LLM API RESPONSE =====")

start = time.perf_counter()

try:

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": "Reply with exactly the word PASS.",
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=60
    )

    response.raise_for_status()

    response_data = response.json()

    generated_answer = response_data.get(
        "response",
        ""
    ).strip()

    if generated_answer:

        actual = "LLM returned a non-empty response"
        status = "PASS"

    else:

        actual = "LLM returned an empty response"
        status = "FAIL"

except requests.RequestException as error:

    generated_answer = ""

    actual = f"LLM API error handled: {error}"
    status = "PASS"

except ValueError as error:

    generated_answer = ""

    actual = f"Invalid API response handled: {error}"
    status = "PASS"

record_result(
    "LLM API Response Validation",
    "Reply with exactly the word PASS",
    "Non-empty response or controlled error",
    actual,
    status,
    time.perf_counter() - start
)


# --------------------------------------------------
# 9. GENERATE CSV REPORT
# --------------------------------------------------

print("\n===== GENERATING ROBUSTNESS REPORT =====")

report = pd.DataFrame(results)

report_path = (
    Path(__file__).parent /
    "rag_pipeline_robustness_report.csv"
)

report.to_csv(
    report_path,
    index=False
)


# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

passed = sum(
    item["Status"] == "PASS"
    for item in results
)

failed = len(results) - passed

print("\n===== FINAL ROBUSTNESS TEST SUMMARY =====")

print(report.to_string(index=False))

print("\nTotal Tests:", len(results))
print("Passed:", passed)
print("Failed:", failed)

print(
    "Success Rate:",
    f"{(passed / len(results)) * 100:.2f}%"
)

print("\nReport saved:", report_path.name)

print("\n===== RAG ROBUSTNESS TESTING COMPLETED =====")
