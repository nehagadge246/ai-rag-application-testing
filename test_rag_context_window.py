
import csv
import re
import time
from datetime import datetime

# ============================================================
# LESSON 67: RAG CONTEXT WINDOW AND TOKEN LIMIT TESTING
# Corrected experiment: fixed document dataset, variable limits
# ============================================================

CONTEXT_LIMITS = [100, 200, 400, 800]

DOCUMENTS = {
    "DOC001": """
Annual Leave Policy
Employees receive 15 days of annual leave per year.
Annual leave requests must be submitted to the reporting manager.
Employees should submit leave requests at least five working days
before the planned leave date.
Unused annual leave is handled according to company policy.
""",
    "DOC002": """
Sick Leave Policy
Employees can take 10 days of paid sick leave per year.
A medical certificate may be requested for extended sick leave.
Employees must inform their manager when they cannot attend work.
Sick leave must be recorded in the employee attendance system.
""",
    "DOC003": """
Remote Work Policy
Employees may work remotely up to 3 days per week with manager approval.
Employees must remain available during normal working hours.
Company security policies apply when working remotely.
Employees must use approved devices and secure network connections.
""",
    "DOC004": """
Parental Leave Policy
Eligible employees receive 26 weeks of maternity leave.
Employees should notify HR before the planned leave period.
Supporting documents may be required for leave approval.
Paternity leave is governed by the applicable company policy.
""",
    "DOC005": """
Public Holiday Policy
Employees receive paid leave on officially declared public holidays.
The annual holiday calendar is shared by HR.
Employees required to work on a public holiday should follow
the applicable compensatory leave policy.
""",
    "DOC006": """
IT Security Policy
Employees must use strong passwords and enable
multi-factor authentication.
Passwords must not be shared with other employees.
Employees must report suspected security incidents immediately.
Company devices must have approved security software installed.
""",
    "DOC007": """
Travel Reimbursement Policy
Employees can claim approved business travel expenses with receipts.
Travel must be approved by the reporting manager before booking.
Expense claims should be submitted within 15 days of travel.
Personal expenses are not eligible for reimbursement.
""",
    "DOC008": """
Performance Review Policy
Employee performance is reviewed twice every year by managers.
Reviews cover individual goals, work quality, and development needs.
Managers should discuss feedback with employees.
Performance review records are maintained by HR.
""",
}

TEST_CASES = [
    {
        "id": "TC01",
        "query": "How many annual leave days do employees receive?",
        "expected": "DOC001",
        "keywords": ["15 days", "annual leave"],
    },
    {
        "id": "TC02",
        "query": "How many paid sick leave days are available?",
        "expected": "DOC002",
        "keywords": ["10 days", "paid sick leave"],
    },
    {
        "id": "TC03",
        "query": "How many days per week can employees work remotely?",
        "expected": "DOC003",
        "keywords": ["3 days", "manager approval"],
    },
    {
        "id": "TC04",
        "query": "How many weeks of maternity leave are provided?",
        "expected": "DOC004",
        "keywords": ["26 weeks", "maternity leave"],
    },
    {
        "id": "TC05",
        "query": "Are officially declared public holidays paid?",
        "expected": "DOC005",
        "keywords": ["paid leave", "public holidays"],
    },
    {
        "id": "TC06",
        "query": "What authentication method must employees enable?",
        "expected": "DOC006",
        "keywords": ["multi-factor authentication", "strong passwords"],
    },
    {
        "id": "TC07",
        "query": "What proof is needed for travel reimbursement?",
        "expected": "DOC007",
        "keywords": ["receipts", "approved business travel expenses"],
    },
    {
        "id": "TC08",
        "query": "How often are employee performance reviews conducted?",
        "expected": "DOC008",
        "keywords": ["twice every year", "performance"],
    },
]


def build_context():
    """Build one fixed context containing all eight documents."""
    parts = []

    for doc_id, content in DOCUMENTS.items():
        parts.append(
            f"\nDocument ID: {doc_id}\n{content.strip()}"
        )

    return "\n".join(parts)


def truncate_context(context, word_limit):
    """Apply the configured word limit."""
    words = context.split()
    truncated_words = words[:word_limit]

    return (
        " ".join(truncated_words),
        len(words),
        len(truncated_words),
    )


def extract_expected_document(context, expected_id):
    """Extract the expected document only if its ID is present."""
    pattern = (
        rf"Document ID:\s*{re.escape(expected_id)}\b"
        rf"(.*?)(?=\nDocument ID:|\Z)"
    )

    match = re.search(pattern, context, re.DOTALL)

    if match:
        return match.group(1).strip()

    return ""


def evaluate_context(test, context):
    expected_content = extract_expected_document(
        context,
        test["expected"],
    )

    expected_present = bool(expected_content)

    found_keywords = [
        keyword
        for keyword in test["keywords"]
        if keyword.lower() in expected_content.lower()
    ]

    missing_keywords = [
        keyword
        for keyword in test["keywords"]
        if keyword.lower() not in expected_content.lower()
    ]

    coverage = (
        len(found_keywords) / len(test["keywords"]) * 100
        if test["keywords"] else 0
    )

    if not expected_present:
        status = "FAIL_EXPECTED_DOCUMENT_MISSING"
    elif missing_keywords:
        status = "FAIL_INFORMATION_MISSING"
    else:
        status = "PASS"

    return {
        "Expected Document Present": expected_present,
        "Found Keywords": ", ".join(found_keywords),
        "Missing Keywords": ", ".join(missing_keywords),
        "Keyword Coverage Percent": round(coverage, 2),
        "Status": status,
        "Passed": status == "PASS",
    }


def main():
    print("=" * 70)
    print("RAG CONTEXT WINDOW AND TOKEN LIMIT TESTING")
    print("=" * 70)

    print(f"Execution date: {datetime.now()}")
    print(f"Context limits: {CONTEXT_LIMITS}")
    print(f"Documents: {len(DOCUMENTS)}")
    print(f"Test cases: {len(TEST_CASES)}")

    full_context = build_context()
    original_words = len(full_context.split())

    print(f"Full context size: {original_words} words")

    all_reports = []
    all_summaries = []

    for word_limit in CONTEXT_LIMITS:

        print("\n" + "=" * 70)
        print(f"CONTEXT LIMIT: {word_limit} WORDS")
        print("=" * 70)

        context, total_words, retained_words = truncate_context(
            full_context,
            word_limit,
        )

        truncation_percent = (
            (total_words - retained_words) / total_words * 100
        )

        print(f"Original words: {total_words}")
        print(f"Retained words: {retained_words}")
        print(f"Truncation: {truncation_percent:.2f}%")

        passed = 0
        total_coverage = 0
        total_latency = 0

        for test in TEST_CASES:

            start = time.perf_counter()

            result = evaluate_context(test, context)

            latency = (
                time.perf_counter() - start
            ) * 1000

            total_latency += latency
            total_coverage += result["Keyword Coverage Percent"]

            if result["Passed"]:
                passed += 1

            report = {
                "Context Limit Words": word_limit,
                "Original Context Words": total_words,
                "Retained Context Words": retained_words,
                "Truncation Percent": round(truncation_percent, 2),
                "Test ID": test["id"],
                "Query": test["query"],
                "Expected Document": test["expected"],
                "Expected Document Present": (
                    "Yes" if result["Expected Document Present"]
                    else "No"
                ),
                "Found Keywords": result["Found Keywords"],
                "Missing Keywords": result["Missing Keywords"],
                "Keyword Coverage Percent": (
                    result["Keyword Coverage Percent"]
                ),
                "Evaluation Latency Ms": round(latency, 4),
                "Status": result["Status"],
            }

            all_reports.append(report)

            print(
                f"{test['id']} | {result['Status']} | "
                f"Coverage: {result['Keyword Coverage Percent']}%"
            )

        total_tests = len(TEST_CASES)

        summary = {
            "Context Limit Words": word_limit,
            "Original Context Words": total_words,
            "Retained Context Words": retained_words,
            "Truncation Percent": round(truncation_percent, 2),
            "Total Test Cases": total_tests,
            "Passed": passed,
            "Failed": total_tests - passed,
            "Pass Rate": round(passed / total_tests * 100, 2),
            "Average Keyword Coverage Percent": round(
                total_coverage / total_tests, 2
            ),
            "Average Evaluation Latency Ms": round(
                total_latency / total_tests, 4
            ),
        }

        all_summaries.append(summary)

        print(f"\nPass rate: {summary['Pass Rate']}%")

    report_file = "rag_context_window_report.csv"
    summary_file = "rag_context_window_summary.csv"

    with open(
        report_file, "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_reports[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_reports)

    with open(
        summary_file, "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_summaries)

    print("\n" + "=" * 70)
    print("CONTEXT WINDOW COMPARISON SUMMARY")
    print("=" * 70)

    for summary in all_summaries:
        print(
            f"\nLimit: {summary['Context Limit Words']} words"
        )
        print(
            f"Passed: {summary['Passed']}/"
            f"{summary['Total Test Cases']}"
        )
        print(f"Pass Rate: {summary['Pass Rate']}%")
        print(
            "Average keyword coverage: "
            f"{summary['Average Keyword Coverage Percent']}%"
        )

    print("\nReports generated:")
    print(report_file)
    print(summary_file)

    print("\nLesson 67 corrected execution completed.")


if __name__ == "__main__":
    main()
