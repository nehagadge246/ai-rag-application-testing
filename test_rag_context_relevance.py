
import requests
import time
import csv
import math
from datetime import datetime

# ==========================================================
# LESSON 53: RAG CONTEXT RELEVANCE AND RETRIEVAL QUALITY
# ==========================================================

OLLAMA_URL = "http://localhost:11434"
EMBED_URL = f"{OLLAMA_URL}/api/embed"
MODEL = "nomic-embed-text:latest"

REPORT_FILE = "rag_context_relevance_report.csv"
SUMMARY_FILE = "rag_context_relevance_summary.csv"

TOP_K = 3
RELEVANCE_THRESHOLD = 0.55

# ==========================================================
# STEP 1: DOCUMENT KNOWLEDGE BASE
# ==========================================================

DOCUMENTS = [
    {
        "id": "DOC001",
        "title": "Annual Leave Policy",
        "content": (
            "Employees receive 15 days of annual leave per year. "
            "Annual leave must be requested and approved by the manager."
        )
    },
    {
        "id": "DOC002",
        "title": "Sick Leave Policy",
        "content": (
            "Employees receive 10 days of sick leave per year. "
            "Employees may need to provide medical documentation."
        )
    },
    {
        "id": "DOC003",
        "title": "Remote Work Policy",
        "content": (
            "Employees can work remotely up to 2 days per week, "
            "subject to manager approval."
        )
    },
    {
        "id": "DOC004",
        "title": "Parental Leave Policy",
        "content": (
            "Employees receive 12 weeks of parental leave. "
            "Employees should contact HR to initiate the process."
        )
    },
    {
        "id": "DOC005",
        "title": "Public Holiday Policy",
        "content": (
            "The organization observes 10 public holidays each year. "
            "The holiday calendar is shared with employees."
        )
    },
    {
        "id": "DOC006",
        "title": "IT Security Policy",
        "content": (
            "Employees must use strong passwords and enable "
            "multi-factor authentication. Company devices must "
            "be protected against unauthorized access."
        )
    },
    {
        "id": "DOC007",
        "title": "Travel Reimbursement Policy",
        "content": (
            "Employees can submit approved business travel expenses "
            "for reimbursement. Receipts must be attached to claims."
        )
    },
    {
        "id": "DOC008",
        "title": "Performance Review Policy",
        "content": (
            "Employee performance reviews are conducted twice a year. "
            "Managers provide feedback and discuss development goals."
        )
    }
]

# ==========================================================
# STEP 2: TEST DATASET
# ==========================================================

TEST_CASES = [
    {
        "id": 1,
        "scenario": "Annual leave retrieval",
        "question": "How many annual leave days do employees receive?",
        "expected_doc": "DOC001",
        "expected_keyword": "15 days"
    },
    {
        "id": 2,
        "scenario": "Sick leave retrieval",
        "question": "What is the yearly sick leave entitlement?",
        "expected_doc": "DOC002",
        "expected_keyword": "10 days"
    },
    {
        "id": 3,
        "scenario": "Remote work retrieval",
        "question": "How many days can employees work from home?",
        "expected_doc": "DOC003",
        "expected_keyword": "2 days"
    },
    {
        "id": 4,
        "scenario": "Parental leave retrieval",
        "question": "How many weeks of parental leave are available?",
        "expected_doc": "DOC004",
        "expected_keyword": "12 weeks"
    },
    {
        "id": 5,
        "scenario": "Public holiday retrieval",
        "question": "How many public holidays are observed annually?",
        "expected_doc": "DOC005",
        "expected_keyword": "10 public holidays"
    },
    {
        "id": 6,
        "scenario": "IT security retrieval",
        "question": "How should employees protect their company accounts?",
        "expected_doc": "DOC006",
        "expected_keyword": "multi-factor authentication"
    },
    {
        "id": 7,
        "scenario": "Travel reimbursement retrieval",
        "question": "How can employees claim business travel expenses?",
        "expected_doc": "DOC007",
        "expected_keyword": "receipts"
    },
    {
        "id": 8,
        "scenario": "Performance review retrieval",
        "question": "How frequently are employee performance reviews conducted?",
        "expected_doc": "DOC008",
        "expected_keyword": "twice a year"
    },
    {
        "id": 9,
        "scenario": "Unrelated query",
        "question": "What is the current stock market price of gold?",
        "expected_doc": None,
        "expected_keyword": None
    },
    {
        "id": 10,
        "scenario": "Unsupported insurance query",
        "question": "Which health insurance company covers employees?",
        "expected_doc": None,
        "expected_keyword": None
    }
]

# ==========================================================
# HELPER 1: OLLAMA CONNECTION AND MODEL CHECK
# ==========================================================

def check_ollama():

    response = requests.get(
        f"{OLLAMA_URL}/api/tags",
        timeout=10
    )

    response.raise_for_status()

    models = [
        item.get("name", "")
        for item in response.json().get("models", [])
    ]

    if not any(
        name == MODEL or name.startswith(MODEL + ":")
        for name in models
    ):
        raise RuntimeError(
            f"Embedding model {MODEL} is not available."
        )

    return True


# ==========================================================
# HELPER 2: GENERATE EMBEDDINGS
# ==========================================================

def generate_embeddings(texts):

    response = requests.post(
        EMBED_URL,
        json={
            "model": MODEL,
            "input": texts
        },
        timeout=120
    )

    response.raise_for_status()

    embeddings = response.json().get("embeddings", [])

    if len(embeddings) != len(texts):
        raise ValueError(
            "Embedding count does not match input count."
        )

    return embeddings


# ==========================================================
# HELPER 3: COSINE SIMILARITY
# ==========================================================

def cosine_similarity(vector_a, vector_b):

    if len(vector_a) != len(vector_b):
        raise ValueError("Embedding dimensions do not match.")

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ==========================================================
# HELPER 4: RETRIEVE TOP-K DOCUMENTS
# ==========================================================

def retrieve_documents(question, document_embeddings):

    query_embedding = generate_embeddings([question])[0]

    scored_documents = []

    for index, document in enumerate(DOCUMENTS):

        score = cosine_similarity(
            query_embedding,
            document_embeddings[index]
        )

        scored_documents.append({
            "document": document,
            "score": score
        })

    ranked_documents = sorted(
        scored_documents,
        key=lambda item: item["score"],
        reverse=True
    )

    return ranked_documents[:TOP_K]


# ==========================================================
# STEP 3: START TEST EXECUTION
# ==========================================================

print("=" * 70)
print("RAG CONTEXT RELEVANCE AND RETRIEVAL QUALITY TESTING")
print("=" * 70)

print("Embedding model:", MODEL)
print("Total documents:", len(DOCUMENTS))
print("Total test cases:", len(TEST_CASES))
print("Top-K:", TOP_K)
print("Relevance threshold:", RELEVANCE_THRESHOLD)

# ==========================================================
# STEP 4: CHECK OLLAMA
# ==========================================================

print("\n===== STEP 1: OLLAMA CONNECTION =====")

try:

    check_ollama()

    print("Ollama connection: PASS")
    print("Embedding model availability: PASS")

except Exception as error:

    print("Ollama connection: FAIL")
    print("Error:", error)
    raise SystemExit(1)

# ==========================================================
# STEP 5: DOCUMENT EMBEDDING / INDEXING
# ==========================================================

print("\n===== STEP 2: DOCUMENT EMBEDDING =====")

document_texts = [
    document["title"] + ". " + document["content"]
    for document in DOCUMENTS
]

try:

    indexing_start = time.perf_counter()

    document_embeddings = generate_embeddings(
        document_texts
    )

    indexing_time = round(
        time.perf_counter() - indexing_start, 3
    )

    embedding_dimensions = len(document_embeddings[0])

    print("Documents embedded:", len(document_embeddings))
    print("Embedding dimensions:", embedding_dimensions)
    print("Indexing time:", indexing_time, "seconds")
    print("Document embedding: PASS")

except Exception as error:

    print("Document embedding: FAIL")
    print("Error:", error)
    raise SystemExit(1)

# ==========================================================
# STEP 6: RETRIEVAL QUALITY TESTING
# ==========================================================

print("\n===== STEP 3: RETRIEVAL QUALITY TESTING =====")

results = []

for test_case in TEST_CASES:

    print("\n" + "-" * 65)
    print(
        f"TEST CASE {test_case['id']}: "
        f"{test_case['scenario']}"
    )
    print("-" * 65)

    start_time = time.perf_counter()

    try:

        retrieved = retrieve_documents(
            test_case["question"],
            document_embeddings
        )

        retrieval_time = round(
            time.perf_counter() - start_time, 3
        )

        top_document = retrieved[0]["document"]
        top_score = retrieved[0]["score"]

        top_document_id = top_document["id"]

        expected_doc = test_case["expected_doc"]

        if expected_doc is not None:

            if top_document_id == expected_doc:
                retrieval_status = "PASS"
            else:
                retrieval_status = "FAIL"

            relevant_documents = [
                item for item in retrieved
                if item["document"]["id"] == expected_doc
            ]

            if relevant_documents:
                expected_score = relevant_documents[0]["score"]
                expected_rank = next(
                    index + 1
                    for index, item in enumerate(retrieved)
                    if item["document"]["id"] == expected_doc
                )
            else:
                expected_score = 0.0
                expected_rank = "Not retrieved"

            relevance_status = (
                "PASS"
                if expected_score >= RELEVANCE_THRESHOLD
                else "REVIEW"
            )

        else:

            expected_score = ""
            expected_rank = "N/A"

            if top_score < RELEVANCE_THRESHOLD:
                retrieval_status = "PASS"
                relevance_status = "PASS"
            else:
                retrieval_status = "REVIEW"
                relevance_status = "REVIEW"

        top_k_documents = "; ".join(
            item["document"]["id"]
            for item in retrieved
        )

        top_k_scores = "; ".join(
            f"{item['score']:.4f}"
            for item in retrieved
        )

        retrieved_context = " | ".join(
            item["document"]["content"]
            for item in retrieved
        )

        overall_status = (
            "PASS"
            if retrieval_status == "PASS"
            and relevance_status == "PASS"
            else "REVIEW"
            if retrieval_status == "REVIEW"
            or relevance_status == "REVIEW"
            else "FAIL"
        )

        result = {
            "Test ID": test_case["id"],
            "Scenario": test_case["scenario"],
            "Question": test_case["question"],
            "Expected Document": expected_doc or "None",
            "Top Retrieved Document": top_document_id,
            "Top Similarity Score": round(top_score, 4),
            "Expected Document Rank": expected_rank,
            "Expected Document Score": (
                round(expected_score, 4)
                if expected_score != "" else ""
            ),
            "Top-K Documents": top_k_documents,
            "Top-K Similarity Scores": top_k_scores,
            "Retrieved Context": retrieved_context,
            "Retrieval Time (sec)": retrieval_time,
            "Retrieval Status": retrieval_status,
            "Relevance Status": relevance_status,
            "Overall Status": overall_status,
            "Error": ""
        }

        print("Question:", test_case["question"])
        print("Top document:", top_document_id)
        print("Top similarity:", round(top_score, 4))
        print("Expected document:", expected_doc or "None")
        print("Expected document rank:", expected_rank)
        print("Retrieved documents:", top_k_documents)
        print("Retrieval time:", retrieval_time, "seconds")
        print("Retrieval status:", retrieval_status)
        print("Relevance status:", relevance_status)
        print("Overall status:", overall_status)

    except Exception as error:

        retrieval_time = round(
            time.perf_counter() - start_time, 3
        )

        result = {
            "Test ID": test_case["id"],
            "Scenario": test_case["scenario"],
            "Question": test_case["question"],
            "Expected Document": test_case["expected_doc"] or "None",
            "Top Retrieved Document": "",
            "Top Similarity Score": "",
            "Expected Document Rank": "",
            "Expected Document Score": "",
            "Top-K Documents": "",
            "Top-K Similarity Scores": "",
            "Retrieved Context": "",
            "Retrieval Time (sec)": retrieval_time,
            "Retrieval Status": "FAIL",
            "Relevance Status": "FAIL",
            "Overall Status": "FAIL",
            "Error": str(error)
        }

        print("Test execution error:", error)

    results.append(result)

# ==========================================================
# STEP 7: CALCULATE METRICS
# ==========================================================

print("\n===== STEP 4: RETRIEVAL QUALITY SUMMARY =====")

total_tests = len(results)

passed = sum(
    1 for item in results
    if item["Overall Status"] == "PASS"
)

failed = sum(
    1 for item in results
    if item["Overall Status"] == "FAIL"
)

review = sum(
    1 for item in results
    if item["Overall Status"] == "REVIEW"
)

retrieval_passed = sum(
    1 for item in results
    if item["Retrieval Status"] == "PASS"
)

relevance_passed = sum(
    1 for item in results
    if item["Relevance Status"] == "PASS"
)

supported_tests = [
    item for item in results
    if item["Expected Document"] != "None"
]

correct_top1 = sum(
    1 for item in supported_tests
    if item["Top Retrieved Document"] == item["Expected Document"]
)

top1_accuracy = (
    round(correct_top1 / len(supported_tests) * 100, 2)
    if supported_tests else 0
)

successful_times = [
    item["Retrieval Time (sec)"]
    for item in results
    if item["Error"] == ""
]

average_retrieval_time = (
    round(
        sum(successful_times) / len(successful_times), 3
    )
    if successful_times else 0
)

print("Total test cases:", total_tests)
print("Passed:", passed)
print("Failed:", failed)
print("Review:", review)
print("Retrieval checks passed:", retrieval_passed)
print("Relevance checks passed:", relevance_passed)
print("Supported-query Top-1 accuracy:", top1_accuracy, "%")
print("Average retrieval time:", average_retrieval_time, "seconds")

# ==========================================================
# STEP 8: GENERATE DETAILED CSV REPORT
# ==========================================================

print("\n===== STEP 5: GENERATING DETAILED REPORT =====")

fieldnames = [
    "Test ID",
    "Scenario",
    "Question",
    "Expected Document",
    "Top Retrieved Document",
    "Top Similarity Score",
    "Expected Document Rank",
    "Expected Document Score",
    "Top-K Documents",
    "Top-K Similarity Scores",
    "Retrieved Context",
    "Retrieval Time (sec)",
    "Retrieval Status",
    "Relevance Status",
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

# ==========================================================
# STEP 9: GENERATE SUMMARY CSV REPORT
# ==========================================================

print("\n===== STEP 6: GENERATING SUMMARY REPORT =====")

summary = {
    "Execution Date": datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
    "Embedding Model": MODEL,
    "Total Documents": len(DOCUMENTS),
    "Total Test Cases": total_tests,
    "Passed": passed,
    "Failed": failed,
    "Review": review,
    "Supported Queries": len(supported_tests),
    "Correct Top-1 Retrievals": correct_top1,
    "Top-1 Accuracy (%)": top1_accuracy,
    "Retrieval Checks Passed": retrieval_passed,
    "Relevance Checks Passed": relevance_passed,
    "Average Retrieval Time (sec)": average_retrieval_time,
    "Top-K": TOP_K,
    "Relevance Threshold": RELEVANCE_THRESHOLD
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

# ==========================================================
# FINAL RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("RAG CONTEXT RELEVANCE TESTING COMPLETED")
print("=" * 70)

print("Total test cases:", total_tests)
print("Passed:", passed)
print("Failed:", failed)
print("Review:", review)
print("Supported-query Top-1 accuracy:", top1_accuracy, "%")
print("Average retrieval time:", average_retrieval_time, "seconds")

print("\nGenerated files:")
print(REPORT_FILE)
print(SUMMARY_FILE)

print("\nIMPORTANT QA NOTES:")
print(
    "Retrieval uses cosine similarity between local embeddings."
)
print(
    "Top-1 accuracy checks whether the expected document "
    "was ranked first for supported questions."
)
print(
    "Unsupported-query checks use an experimental similarity "
    "threshold; they do not prove that no relevant information exists."
)
print(
    "This is a controlled in-memory retrieval experiment, "
    "not a production vector database benchmark."
)

print("=" * 70)
