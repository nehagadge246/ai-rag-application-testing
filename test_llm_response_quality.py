
import requests
import time
import pandas as pd
from pathlib import Path

print("===== LOCAL LLM RESPONSE QUALITY TESTING =====")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:3b"

# Test scenarios with expected keywords
test_cases = [
    {
        "id": 1,
        "question": "How many annual leave days do employees receive?",
        "context": "Employees receive 20 annual leave days per year.",
        "expected_keywords": ["20", "annual leave"]
    },
    {
        "id": 2,
        "question": "How many unused leave days can employees carry forward?",
        "context": "Employees can carry forward up to 5 unused leave days.",
        "expected_keywords": ["5", "carry forward"]
    },
    {
        "id": 3,
        "question": "What is the annual leave entitlement?",
        "context": "Employees receive 20 annual leave days per year.",
        "expected_keywords": ["20", "annual leave"]
    }
]

results = []

for test in test_cases:

    print(f"\n===== TEST CASE {test['id']} =====")
    print("Question:", test["question"])

    prompt = f"""
Answer the question using ONLY the provided context.
Do not invent information.
If the answer is unavailable, say that it is not available.

Context: {test['context']}
Question: {test['question']}
Answer:
"""

    start_time = time.perf_counter()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0
                }
            },
            timeout=120
        )

        response.raise_for_status()

        answer = response.json()["response"].strip()

        latency = round(
            time.perf_counter() - start_time, 2
        )

        answer_lower = answer.lower()

        matched_keywords = [
            keyword
            for keyword in test["expected_keywords"]
            if keyword.lower() in answer_lower
        ]

        missing_keywords = [
            keyword
            for keyword in test["expected_keywords"]
            if keyword.lower() not in answer_lower
        ]

        keyword_coverage = (
            len(matched_keywords) /
            len(test["expected_keywords"])
        )

        status = (
            "PASS"
            if keyword_coverage == 1.0
            else "FAIL"
        )

        print("Generated Answer:", answer)
        print("Matched Keywords:", matched_keywords)
        print("Missing Keywords:", missing_keywords)
        print("Keyword Coverage:", f"{keyword_coverage:.2%}")
        print("Response Time:", f"{latency}s")
        print("Status:", status)

        results.append({
            "Test Case": test["id"],
            "Question": test["question"],
            "Generated Answer": answer,
            "Matched Keywords": ", ".join(matched_keywords),
            "Missing Keywords": ", ".join(missing_keywords),
            "Keyword Coverage": round(keyword_coverage, 2),
            "Latency Seconds": latency,
            "Status": status
        })

    except requests.exceptions.RequestException as error:

        print("API Error:", str(error))

        results.append({
            "Test Case": test["id"],
            "Question": test["question"],
            "Generated Answer": "",
            "Matched Keywords": "",
            "Missing Keywords": "",
            "Keyword Coverage": 0,
            "Latency Seconds": "",
            "Status": "ERROR"
        })

# Generate CSV report
report = pd.DataFrame(results)

report_path = (
    Path(__file__).parent /
    "llm_response_quality_report.csv"
)

report.to_csv(report_path, index=False)

print("\n===== FINAL QA SUMMARY =====")
print(report[["Test Case", "Keyword Coverage", "Status"]])

print("\nTotal Tests:", len(results))
print("Passed:", sum(r["Status"] == "PASS" for r in results))
print("Failed:", sum(r["Status"] == "FAIL" for r in results))
print("Errors:", sum(r["Status"] == "ERROR" for r in results))

print("\nReport saved:", report_path.name)
print("===== LLM TESTING COMPLETED =====")
