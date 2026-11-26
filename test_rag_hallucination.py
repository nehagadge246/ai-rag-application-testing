
import csv
import re
import time
from datetime import datetime

import ollama


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "all-minilm:latest"
BASE_URL = "http://localhost:11434"

TOP_K = 3
SIMILARITY_THRESHOLD = 0.68

REPORT_FILE = "rag_hallucination_report.csv"
SUMMARY_FILE = "rag_hallucination_summary.csv"


# ============================================================
# HR KNOWLEDGE BASE
# ============================================================

DOCUMENTS = {
    "DOC001": {
        "title": "Annual Leave Policy",
        "content": (
            "Employees receive 15 annual leave days per year. "
            "Annual leave requests must be submitted to the manager "
            "for approval."
        ),
    },
    "DOC002": {
        "title": "Sick Leave Policy",
        "content": (
            "Employees receive 10 sick leave days per year. "
            "A medical certificate may be required for extended "
            "sick leave."
        ),
    },
    "DOC003": {
        "title": "Remote Work Policy",
        "content": (
            "Employees may work remotely up to 2 days per week "
            "with manager approval. Employees must follow company "
            "security policies while working remotely."
        ),
    },
    "DOC004": {
        "title": "Parental Leave Policy",
        "content": (
            "Eligible employees receive 26 weeks of maternity leave. "
            "Employees should contact HR to understand eligibility "
            "and application requirements."
        ),
    },
    "DOC005": {
        "title": "Public Holiday Policy",
        "content": (
            "The company publishes an annual public holiday calendar. "
            "Employees should refer to the official calendar for "
            "applicable holidays."
        ),
    },
    "DOC006": {
        "title": "IT Security Policy",
        "content": (
            "Employees must use strong passwords and enable "
            "multi-factor authentication. Company credentials must "
            "never be shared with other individuals."
        ),
    },
    "DOC007": {
        "title": "Travel Reimbursement Policy",
        "content": (
            "Employees can submit eligible business travel expenses "
            "for reimbursement. Expense claims must include valid "
            "receipts and manager approval."
        ),
    },
    "DOC008": {
        "title": "Performance Review Policy",
        "content": (
            "Employee performance reviews are conducted twice a year. "
            "Managers provide feedback and discuss development goals "
            "during the review process."
        ),
    },
}


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    {
        "id": "TC01",
        "category": "Supported",
        "query": "How many annual leave days do employees receive?",
        "expected_doc": "DOC001",
        "expected_keywords": ["15", "annual leave"],
        "expected_answer": "15 annual leave days",
    },
    {
        "id": "TC02",
        "category": "Supported",
        "query": "How many sick leave days are available?",
        "expected_doc": "DOC002",
        "expected_keywords": ["10", "sick leave"],
        "expected_answer": "10 sick leave days",
    },
    {
        "id": "TC03",
        "category": "Supported",
        "query": "How many days can employees work remotely?",
        "expected_doc": "DOC003",
        "expected_keywords": ["2 days", "manager approval"],
        "expected_answer": "2 days per week",
    },
    {
        "id": "TC04",
        "category": "Supported",
        "query": "How long is maternity leave?",
        "expected_doc": "DOC004",
        "expected_keywords": ["26 weeks", "maternity leave"],
        "expected_answer": "26 weeks of maternity leave",
    },
    {
        "id": "TC05",
        "category": "Supported",
        "query": "How often are employee performance reviews conducted?",
        "expected_doc": "DOC008",
        "expected_keywords": ["twice a year", "performance reviews"],
        "expected_answer": "twice a year",
    },
    {
        "id": "TC06",
        "category": "Supported",
        "query": "What is required for travel reimbursement?",
        "expected_doc": "DOC007",
        "expected_keywords": ["receipts", "manager approval"],
        "expected_answer": "valid receipts and manager approval",
    },
    {
        "id": "TC07",
        "category": "Unsupported",
        "query": "How much annual performance bonus does every employee receive?",
        "expected_doc": "",
        "expected_keywords": [],
        "expected_answer": "",
    },
    {
        "id": "TC08",
        "category": "Unsupported",
        "query": "Does the company provide free health insurance for family members?",
        "expected_doc": "",
        "expected_keywords": [],
        "expected_answer": "",
    },
    {
        "id": "TC09",
        "category": "Unsupported",
        "query": "How many work-from-home days are guaranteed without approval?",
        "expected_doc": "",
        "expected_keywords": [],
        "expected_answer": "",
    },
    {
        "id": "TC10",
        "category": "Unsupported",
        "query": "What is the exact salary increment percentage for every employee?",
        "expected_doc": "",
        "expected_keywords": [],
        "expected_answer": "",
    },
    {
        "id": "TC11",
        "category": "Unsupported",
        "query": "How many paid annual leave days are provided?",
        "expected_doc": "DOC001",
        "expected_keywords": ["15", "annual leave"],
        "expected_answer": "15 annual leave days",
        "fabricated_answer": "Employees receive 30 annual leave days.",
    },
    {
        "id": "TC12",
        "category": "Unsupported",
        "query": "How long is maternity leave?",
        "expected_doc": "DOC004",
        "expected_keywords": ["26 weeks", "maternity leave"],
        "expected_answer": "26 weeks of maternity leave",
        "fabricated_answer": "Employees receive 52 weeks of maternity leave.",
    },
]


# ============================================================
# OLLAMA CONNECTION
# ============================================================

client = ollama.Client(host=BASE_URL)


# ============================================================
# EMBEDDINGS
# ============================================================

def get_embedding(text):
    response = client.embeddings(
        model=MODEL,
        prompt=text
    )

    return response["embedding"]


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ============================================================
# DOCUMENT INDEXING
# ============================================================

def build_document_index():
    print("\nBuilding document embeddings...")

    index = {}

    for doc_id, document in DOCUMENTS.items():

        text = (
            document["title"] + " " +
            document["content"]
        )

        index[doc_id] = {
            "embedding": get_embedding(text),
            "content": document["content"],
            "title": document["title"],
        }

    print(
        f"Indexed {len(index)} documents successfully."
    )

    return index


# ============================================================
# DOCUMENT RETRIEVAL
# ============================================================

def retrieve_documents(query, index):

    query_embedding = get_embedding(query)

    results = []

    for doc_id, document in index.items():

        score = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        results.append({
            "doc_id": doc_id,
            "score": score,
            "content": document["content"],
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:TOP_K]


# ============================================================
# ANSWER VALIDATION
# ============================================================

def validate_answer(answer, expected_keywords):

    answer_lower = answer.lower()

    matched = [
        keyword
        for keyword in expected_keywords
        if keyword.lower() in answer_lower
    ]

    coverage = (
        len(matched) / len(expected_keywords) * 100
        if expected_keywords else 0
    )

    return matched, coverage


def normalize_numbers(text):

    return re.findall(
        r"\d+(?:\.\d+)?",
        text
    )


def detect_numeric_conflict(answer, source_text):

    answer_numbers = normalize_numbers(answer)
    source_numbers = normalize_numbers(source_text)

    for number in answer_numbers:

        if number not in source_numbers:
            return True, number

    return False, ""


# ============================================================
# TEST EXECUTION
# ============================================================

def run_tests(index):

    report_rows = []

    for test in TEST_CASES:

        print("\n" + "=" * 65)

        print(
            f"{test['id']} | {test['category']} | "
            f"{test['query']}"
        )

        start_time = time.perf_counter()

        retrieved = retrieve_documents(
            test["query"],
            index
        )

        latency = (
            time.perf_counter() - start_time
        ) * 1000

        top_document = retrieved[0]

        top_score = top_document["score"]

        retrieved_ids = [
            item["doc_id"]
            for item in retrieved
        ]

        source_text = " ".join(
            item["content"]
            for item in retrieved
        )

        expected_doc = test["expected_doc"]

        document_found = (
            expected_doc in retrieved_ids
            if expected_doc else False
        )

        conflict = False
        conflicting_number = ""

        # ----------------------------------------------------
        # UNSUPPORTED AND FABRICATED ANSWER TESTS
        # ----------------------------------------------------

        if test["category"] == "Unsupported":

            if test.get("fabricated_answer"):

                fabricated_answer = test["fabricated_answer"]

                conflict, conflicting_number = (
                    detect_numeric_conflict(
                        fabricated_answer,
                        source_text
                    )
                )

                status = (
                    "HALLUCINATION_DETECTED"
                    if conflict
                    else "REVIEW"
                )

                matched, coverage = validate_answer(
                    fabricated_answer,
                    test["expected_keywords"]
                )

                passed = conflict

                answer_type = "Fabricated numeric answer"

                print(
                    f"Fabricated answer: {fabricated_answer}"
                )

                print(
                    f"Expected answer: {test['expected_answer']}"
                )

                print(
                    f"Numeric conflict: {conflict}"
                )

                if conflict:
                    print(
                        f"Unsupported number: "
                        f"{conflicting_number}"
                    )

            else:

                status = (
                    "REVIEW"
                    if top_score >= SIMILARITY_THRESHOLD
                    else "REJECTED"
                )

                matched = []
                coverage = 0.0

                passed = status == "REJECTED"

                answer_type = "Unsupported question"

        # ----------------------------------------------------
        # SUPPORTED QUESTION TESTS
        # ----------------------------------------------------

        else:

            matched, coverage = validate_answer(
                source_text,
                test["expected_keywords"]
            )

            passed = (
                document_found and
                coverage == 100.0
            )

            status = (
                "PASS"
                if passed
                else "FAIL"
            )

            answer_type = "Supported question"

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print(
            f"Top document: {top_document['doc_id']}"
        )

        print(
            f"Similarity: {top_score:.4f}"
        )

        print(
            f"Retrieved documents: {retrieved_ids}"
        )

        print(
            f"Expected document found: {document_found}"
        )

        print(
            f"Keyword coverage: {coverage:.2f}%"
        )

        print(
            f"Status: {status}"
        )

        print(
            f"Latency: {latency:.2f} ms"
        )

        report_rows.append({

            "Test ID": test["id"],
            "Category": test["category"],
            "Query": test["query"],
            "Answer Type": answer_type,
            "Expected Document": expected_doc,
            "Retrieved Documents": ", ".join(retrieved_ids),
            "Top Similarity": round(top_score, 4),
            "Expected Document Found": document_found,
            "Expected Keywords": ", ".join(
                test["expected_keywords"]
            ),
            "Matched Keywords": ", ".join(matched),
            "Keyword Coverage Percent": round(
                coverage, 2
            ),
            "Fabricated Answer": test.get(
                "fabricated_answer", ""
            ),
            "Numeric Conflict Detected": (
                conflict if test.get("fabricated_answer")
                else ""
            ),
            "Conflicting Number": conflicting_number,
            "Status": status,
            "Passed": passed,
            "Latency Ms": round(latency, 2),

        })

    return report_rows


# ============================================================
# CSV REPORT GENERATION
# ============================================================

def generate_reports(rows):

    fields = list(rows[0].keys())

    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)

    passed = sum(
        row["Passed"]
        for row in rows
    )

    failed = total - passed

    supported = [
        row for row in rows
        if row["Category"] == "Supported"
    ]

    unsupported = [
        row for row in rows
        if row["Category"] == "Unsupported"
    ]

    supported_passed = sum(
        row["Passed"]
        for row in supported
    )

    unsupported_rejected = sum(
        row["Status"] == "REJECTED"
        for row in unsupported
    )

    hallucinations_detected = sum(
        row["Status"] == "HALLUCINATION_DETECTED"
        for row in rows
    )

    review_count = sum(
        row["Status"] == "REVIEW"
        for row in rows
    )

    summary = {

        "Execution Date": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "Embedding Model": MODEL,

        "Total Documents": len(DOCUMENTS),

        "Total Test Cases": total,

        "Supported Test Cases": len(supported),

        "Unsupported Test Cases": len(unsupported),

        "Passed": passed,

        "Failed": failed,

        "Overall Pass Rate": round(
            passed / total * 100, 2
        ),

        "Supported Question Pass Rate": round(
            supported_passed / len(supported) * 100, 2
        ),

        "Unsupported Rejection Rate": round(
            unsupported_rejected / len(unsupported) * 100, 2
        ),

        "Hallucinations Detected": hallucinations_detected,

        "Cases Requiring Review": review_count,

        "Similarity Threshold": SIMILARITY_THRESHOLD,

        "Top K": TOP_K,

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

    print("\n" + "=" * 65)

    print("HALLUCINATION TESTING SUMMARY")

    print("=" * 65)

    for key, value in summary.items():

        print(f"{key}: {value}")

    print("\nReports generated:")

    print(REPORT_FILE)

    print(SUMMARY_FILE)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)

    print(
        "RAG HALLUCINATION AND UNSUPPORTED-ANSWER TESTING"
    )

    print("=" * 65)

    print(f"Embedding model: {MODEL}")

    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")

    print(f"Top K: {TOP_K}")

    print(f"Documents: {len(DOCUMENTS)}")

    print(f"Test cases: {len(TEST_CASES)}")

    try:

        # FIX: Ollama returns model objects, not dictionaries.

        model_response = client.list()

        available_models = [
            model.model
            for model in model_response.models
        ]

        model_available = any(
            name == MODEL or
            name.startswith(MODEL.split(":")[0] + ":")
            for name in available_models
        )

        if not model_available:

            raise RuntimeError(
                f"Model {MODEL} is not available in Ollama."
            )

        print("\nOllama embedding model verified.")

        index = build_document_index()

        rows = run_tests(index)

        generate_reports(rows)

        print("\nLesson 68 execution completed.")

    except Exception as error:

        print("\nERROR:", str(error))

        raise


if __name__ == "__main__":

    main()
