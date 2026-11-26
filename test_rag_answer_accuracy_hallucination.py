
import requests
import time
import csv
import re
from datetime import datetime

# =====================================================
# LESSON 52: RAG ANSWER ACCURACY AND HALLUCINATION TESTING
# =====================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:3b"
TIMEOUT = 120

REPORT_FILE = "rag_answer_accuracy_hallucination_report.csv"
SUMMARY_FILE = "rag_answer_accuracy_hallucination_summary.csv"

# -----------------------------------------------------
# HR POLICY KNOWLEDGE BASE (TEST CONTEXT)
# -----------------------------------------------------

POLICY_CONTEXT = """
HR POLICY DOCUMENT

1. Annual Leave:
Employees receive 15 days of annual leave per year.

2. Sick Leave:
Employees receive 10 days of sick leave per year.

3. Remote Work:
Employees are allowed to work remotely up to 2 days per week,
subject to manager approval.

4. Parental Leave:
Employees receive 12 weeks of parental leave.

5. Public Holidays:
The organization observes 10 public holidays each year.
"""

# -----------------------------------------------------
# TEST DATASET
# -----------------------------------------------------

TEST_CASES = [
    {
        "id": 1,
        "scenario": "Correct annual leave answer",
        "question": "How many annual leave days do employees receive?",
        "expected": "15 days",
        "keywords": ["15"],
        "should_answer": True
    },
    {
        "id": 2,
        "scenario": "Correct sick leave answer",
        "question": "How many sick leave days are provided?",
        "expected": "10 days",
        "keywords": ["10"],
        "should_answer": True
    },
    {
        "id": 3,
        "scenario": "Remote work policy",
        "question": "How many days can employees work remotely per week?",
        "expected": "2 days per week",
        "keywords": ["2"],
        "should_answer": True
    },
    {
        "id": 4,
        "scenario": "Parental leave accuracy",
        "question": "How many weeks of parental leave are available?",
        "expected": "12 weeks",
        "keywords": ["12"],
        "should_answer": True
    },
    {
        "id": 5,
        "scenario": "Public holiday accuracy",
        "question": "How many public holidays are observed each year?",
        "expected": "10 public holidays",
        "keywords": ["10"],
        "should_answer": True
    },
    {
        "id": 6,
        "scenario": "Unsupported salary question",
        "question": "What is the annual salary of an employee?",
        "expected": "Not specified in the provided context",
        "keywords": [],
        "should_answer": False
    },
    {
        "id": 7,
        "scenario": "Unsupported bonus question",
        "question": "How much annual performance bonus does an employee receive?",
        "expected": "Not specified in the provided context",
        "keywords": [],
        "should_answer": False
    },
    {
        "id": 8,
        "scenario": "Unsupported insurance question",
        "question": "Which health insurance company does the organization use?",
        "expected": "Not specified in the provided context",
        "keywords": [],
        "should_answer": False
    }
]


# -----------------------------------------------------
# HELPER: NORMALIZE TEXT
# -----------------------------------------------------

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -----------------------------------------------------
# HELPER: GENERATE LLM ANSWER
# -----------------------------------------------------

def generate_answer(question):

    prompt = f"""
You are a reliable HR policy assistant.

Answer the question using ONLY the information in the
provided HR policy context.

Rules:
1. Never invent facts.
2. Do not assume missing information.
3. If the answer is unavailable in the context,
   respond exactly:
   "This information is not specified in the provided context."
4. Keep your answer concise.

HR POLICY CONTEXT:
{POLICY_CONTEXT}

QUESTION:
{question}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "seed": 42
            }
        },
        timeout=TIMEOUT
    )

    response.raise_for_status()

    answer = response.json().get("response", "").strip()

    return answer


# -----------------------------------------------------
# HELPER: CHECK ANSWER ACCURACY
# -----------------------------------------------------

def evaluate_answer(answer, test_case):

    normalized_answer = normalize_text(answer)

    should_answer = test_case["should_answer"]
    keywords = test_case["keywords"]

    if not answer.strip():
        return "FAIL", "Empty answer", "FAIL"

    if should_answer:

        missing_keywords = [
            keyword for keyword in keywords
            if keyword.lower() not in normalized_answer
        ]

        if missing_keywords:
            return (
                "FAIL",
                "Expected answer information missing",
                "REVIEW"
            )

        return (
            "PASS",
            "Expected information found in answer",
            "PASS"
        )

    else:

        refusal_phrases = [
            "not specified",
            "not provided",
            "not mentioned",
            "not available",
            "no information",
            "cannot determine",
            "can't determine",
            "does not specify",
            "doesn't specify",
            "unable to determine"
        ]

        if any(
            phrase in normalized_answer
            for phrase in refusal_phrases
        ):
            return (
                "PASS",
                "Unsupported question handled without a factual claim",
                "PASS"
            )

        return (
            "FAIL",
            "Potential unsupported answer or hallucination",
            "FAIL"
        )


# -----------------------------------------------------
# STEP 1: CHECK OLLAMA
# -----------------------------------------------------

print("=" * 70)
print("RAG ANSWER ACCURACY AND HALLUCINATION DETECTION")
print("=" * 70)

print("\nModel:", MODEL)
print("Total test cases:", len(TEST_CASES))

print("\n===== STEP 1: OLLAMA CONNECTION =====")

try:
    connection = requests.get(
        "http://localhost:11434/api/tags",
        timeout=10
    )

    connection.raise_for_status()

    models = [
        item.get("name", "")
        for item in connection.json().get("models", [])
    ]

    if not any(
        name == MODEL or name.startswith(MODEL + ":")
        for name in models
    ):
        raise RuntimeError(f"Model {MODEL} not found")

    print("Ollama connection: PASS")
    print("Model availability: PASS")

except Exception as error:
    print("Ollama connection: FAIL")
    print("Error:", error)
    raise SystemExit(1)


# -----------------------------------------------------
# STEP 2: EXECUTE TEST CASES
# -----------------------------------------------------

print("\n===== STEP 2: ANSWER ACCURACY TESTING =====")

results = []

for test_case in TEST_CASES:

    print("\n" + "-" * 60)
    print(
        f"TEST CASE {test_case['id']}: "
        f"{test_case['scenario']}"
    )
    print("-" * 60)

    start_time = time.perf_counter()

    try:

        answer = generate_answer(test_case["question"])

        response_time = round(
            time.perf_counter() - start_time, 3
        )

        accuracy_status, accuracy_details, hallucination_status = (
            evaluate_answer(answer, test_case)
        )

        if test_case["should_answer"]:
            hallucination_check = (
                "PASS" if accuracy_status == "PASS" else "REVIEW"
            )
        else:
            hallucination_check = hallucination_status

        overall_status = (
            "PASS"
            if accuracy_status == "PASS"
            and hallucination_check == "PASS"
            else "REVIEW"
            if accuracy_status == "REVIEW"
            or hallucination_check == "REVIEW"
            else "FAIL"
        )

        result = {
            "Test ID": test_case["id"],
            "Scenario": test_case["scenario"],
            "Question": test_case["question"],
            "Expected Answer": test_case["expected"],
            "Generated Answer": answer,
            "Response Time (sec)": response_time,
            "Accuracy Status": accuracy_status,
            "Hallucination Check": hallucination_check,
            "Evaluation Details": accuracy_details,
            "Overall Status": overall_status,
            "Error": ""
        }

        print("Question:", test_case["question"])
        print("Expected:", test_case["expected"])
        print("Generated:", answer)
        print("Response time:", response_time, "seconds")
        print("Accuracy:", accuracy_status)
        print("Hallucination check:", hallucination_check)
        print("Overall:", overall_status)

    except Exception as error:

        response_time = round(
            time.perf_counter() - start_time, 3
        )

        result = {
            "Test ID": test_case["id"],
            "Scenario": test_case["scenario"],
            "Question": test_case["question"],
            "Expected Answer": test_case["expected"],
            "Generated Answer": "",
            "Response Time (sec)": response_time,
            "Accuracy Status": "FAIL",
            "Hallucination Check": "FAIL",
            "Evaluation Details": "Test execution error",
            "Overall Status": "FAIL",
            "Error": str(error)
        }

        print("Test execution failed:", error)

    results.append(result)


# -----------------------------------------------------
# STEP 3: CALCULATE RESULTS
# -----------------------------------------------------

print("\n===== STEP 3: TEST SUMMARY =====")

total_tests = len(results)

passed = sum(
    1 for result in results
    if result["Overall Status"] == "PASS"
)

failed = sum(
    1 for result in results
    if result["Overall Status"] == "FAIL"
)

review = sum(
    1 for result in results
    if result["Overall Status"] == "REVIEW"
)

accuracy_passed = sum(
    1 for result in results
    if result["Accuracy Status"] == "PASS"
)

hallucination_passed = sum(
    1 for result in results
    if result["Hallucination Check"] == "PASS"
)

successful_latencies = [
    result["Response Time (sec)"]
    for result in results
    if not result["Error"]
]

average_latency = (
    round(
        sum(successful_latencies) / len(successful_latencies),
        3
    )
    if successful_latencies else 0
)

pass_percentage = round(
    (passed / total_tests) * 100, 2
) if total_tests else 0

print("Total test cases:", total_tests)
print("Passed:", passed)
print("Failed:", failed)
print("Review:", review)
print("Accuracy checks passed:", accuracy_passed)
print("Hallucination checks passed:", hallucination_passed)
print("Average response time:", average_latency, "seconds")
print("Overall pass percentage:", pass_percentage, "%")


# -----------------------------------------------------
# STEP 4: GENERATE DETAILED CSV REPORT
# -----------------------------------------------------

print("\n===== STEP 4: GENERATING DETAILED REPORT =====")

fieldnames = [
    "Test ID",
    "Scenario",
    "Question",
    "Expected Answer",
    "Generated Answer",
    "Response Time (sec)",
    "Accuracy Status",
    "Hallucination Check",
    "Evaluation Details",
    "Overall Status",
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


# -----------------------------------------------------
# STEP 5: GENERATE SUMMARY CSV
# -----------------------------------------------------

print("\n===== STEP 5: GENERATING SUMMARY REPORT =====")

summary = {
    "Execution Date": datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
    "Model": MODEL,
    "Total Test Cases": total_tests,
    "Passed": passed,
    "Failed": failed,
    "Review": review,
    "Accuracy Checks Passed": accuracy_passed,
    "Hallucination Checks Passed": hallucination_passed,
    "Average Response Time (sec)": average_latency,
    "Pass Percentage": pass_percentage
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


# -----------------------------------------------------
# FINAL RESULTS
# -----------------------------------------------------

print("\n" + "=" * 70)
print("RAG ANSWER ACCURACY AND HALLUCINATION TESTING COMPLETED")
print("=" * 70)

print("Total tests:", total_tests)
print("Passed:", passed)
print("Failed:", failed)
print("Review:", review)
print("Pass percentage:", pass_percentage, "%")

print("\nGenerated files:")
print(REPORT_FILE)
print(SUMMARY_FILE)

print("\nIMPORTANT QA NOTE:")
print(
    "This is a rule-based test using a controlled HR policy context."
)
print(
    "It checks expected keywords and abstention phrases."
)
print(
    "It does not provide full semantic factual verification."
)
print(
    "A PASS indicates the configured checks passed, "
    "not that every aspect of the answer is factually correct."
)

print("=" * 70)
