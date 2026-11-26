
"""
LESSON 73: RAG PER-QUERY RETRIEVAL AUDIT

Objectives:
1. Inspect actual retrieved documents for each query.
2. Record similarity scores and ranking positions.
3. Compare expected and retrieved document IDs.
4. Identify false positives and missed relevant documents.
5. Classify retrieval root causes.
6. Generate detailed and summary CSV reports.
"""

import csv
import time
import requests
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_ENDPOINT = f"{OLLAMA_BASE_URL}/api/embeddings"

MODEL = "all-minilm:latest"

TOP_K_VALUES = [1, 2, 3, 5]
SIMILARITY_THRESHOLD = 0.55

REQUEST_TIMEOUT = 30

DETAIL_REPORT = "rag_query_audit_report.csv"
SUMMARY_REPORT = "rag_query_audit_summary.csv"

EXECUTION_DATE = datetime.now().astimezone().strftime(
    "%Y-%m-%d %H:%M:%S"
)


# ============================================================
# TEST DOCUMENTS
# ============================================================

DOCUMENTS = [
    {
        "id": "DOC001",
        "text": (
            "Employees receive 15 annual leave days per year. "
            "Annual leave requires manager approval."
        ),
    },
    {
        "id": "DOC002",
        "text": (
            "Employees receive 10 sick leave days per year. "
            "A medical certificate may be required."
        ),
    },
    {
        "id": "DOC003",
        "text": (
            "Employees may work remotely up to two days per week "
            "with manager approval."
        ),
    },
    {
        "id": "DOC004",
        "text": (
            "Eligible employees receive 26 weeks of maternity leave."
        ),
    },
    {
        "id": "DOC005",
        "text": (
            "The company observes 12 public holidays each calendar year."
        ),
    },
    {
        "id": "DOC006",
        "text": (
            "Employees must use multi-factor authentication. "
            "Passwords must never be shared."
        ),
    },
    {
        "id": "DOC007",
        "text": (
            "Approved business travel expenses can be reimbursed "
            "after submitting valid receipts."
        ),
    },
    {
        "id": "DOC008",
        "text": (
            "Employee performance reviews are conducted twice per year."
        ),
    },
]


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    {
        "id": "TC01",
        "query": "How many annual leave days do employees receive?",
        "expected_doc": "DOC001",
        "expected_type": "supported",
    },
    {
        "id": "TC02",
        "query": "What is the sick leave entitlement?",
        "expected_doc": "DOC002",
        "expected_type": "supported",
    },
    {
        "id": "TC03",
        "query": "Can staff work from home during the week?",
        "expected_doc": "DOC003",
        "expected_type": "supported",
    },
    {
        "id": "TC04",
        "query": "How long is maternity leave?",
        "expected_doc": "DOC004",
        "expected_type": "supported",
    },
    {
        "id": "TC05",
        "query": "What is the employee insurance premium?",
        "expected_doc": "",
        "expected_type": "unsupported",
    },
    {
        "id": "TC06",
        "query": "What is the guaranteed annual bonus?",
        "expected_doc": "",
        "expected_type": "unsupported",
    },
]


# ============================================================
# EMBEDDINGS
# ============================================================

def get_embedding(text):

    response = requests.post(
        EMBEDDING_ENDPOINT,
        json={
            "model": MODEL,
            "prompt": text,
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    embedding = response.json().get("embedding")

    if not embedding:
        raise RuntimeError(
            f"Empty embedding returned for: {text}"
        )

    return embedding


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):

    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Embedding dimensions do not match."
        )

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    norm_a = sum(
        value * value for value in vector_a
    ) ** 0.5

    norm_b = sum(
        value * value for value in vector_b
    ) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ============================================================
# BUILD DOCUMENT INDEX
# ============================================================

def build_index():

    indexed_documents = []

    print("\nGenerating document embeddings...")

    for document in DOCUMENTS:

        embedding = get_embedding(document["text"])

        indexed_documents.append({
            **document,
            "embedding": embedding,
        })

        print(
            f"Indexed {document['id']} "
            f"| Dimensions: {len(embedding)}"
        )

    return indexed_documents


# ============================================================
# RETRIEVE AND RANK DOCUMENTS
# ============================================================

def retrieve_documents(query, indexed_documents, top_k):

    start = time.perf_counter()

    query_embedding = get_embedding(query)

    scored_documents = []

    for document in indexed_documents:

        similarity = cosine_similarity(
            query_embedding,
            document["embedding"],
        )

        scored_documents.append({
            "Document ID": document["id"],
            "Similarity Score": round(similarity, 6),
        })

    scored_documents.sort(
        key=lambda item: item["Similarity Score"],
        reverse=True,
    )

    latency = (
        time.perf_counter() - start
    ) * 1000

    return scored_documents[:top_k], latency


# ============================================================
# ROOT CAUSE CLASSIFICATION
# ============================================================

def classify_root_cause(
    test_case,
    retrieved_documents,
    accepted_documents,
):

    expected_doc = test_case["expected_doc"]

    if test_case["expected_type"] == "unsupported":

        if accepted_documents:
            return "Unsupported query retrieved documents"

        if retrieved_documents:
            return "Correctly rejected unsupported query"

        return "No documents retrieved"

    accepted_ids = [
        document["Document ID"]
        for document in accepted_documents
    ]

    retrieved_ids = [
        document["Document ID"]
        for document in retrieved_documents
    ]

    if expected_doc not in retrieved_ids:

        if retrieved_documents:
            return "Expected document not in Top-K"

        return "No documents retrieved"

    if expected_doc not in accepted_ids:
        return "Expected document below similarity threshold"

    if len(accepted_ids) > 1:
        return "Correct document retrieved with extra candidates"

    return "Correct retrieval"


# ============================================================
# PER-QUERY AUDIT
# ============================================================

def run_audit(indexed_documents):

    detail_rows = []
    summary_rows = []

    total_queries = (
        len(TEST_CASES) * len(TOP_K_VALUES)
    )

    query_number = 0

    print("\nStarting per-query retrieval audit...")
    print("-" * 85)

    for top_k in TOP_K_VALUES:

        for test_case in TEST_CASES:

            query_number += 1

            retrieved, latency = retrieve_documents(
                test_case["query"],
                indexed_documents,
                top_k,
            )

            accepted = [
                document
                for document in retrieved
                if document["Similarity Score"]
                >= SIMILARITY_THRESHOLD
            ]

            accepted_ids = [
                document["Document ID"]
                for document in accepted
            ]

            retrieved_ids = [
                document["Document ID"]
                for document in retrieved
            ]

            expected_doc = test_case["expected_doc"]

            if test_case["expected_type"] == "supported":

                expected_found = expected_doc in retrieved_ids
                expected_accepted = expected_doc in accepted_ids

                false_positive_ids = [
                    document_id
                    for document_id in accepted_ids
                    if document_id != expected_doc
                ]

                if expected_accepted and not false_positive_ids:
                    status = "PASS"
                else:
                    status = "FAIL"

            else:

                expected_found = False
                expected_accepted = False

                false_positive_ids = accepted_ids

                status = (
                    "PASS" if not accepted else "FAIL"
                )

            root_cause = classify_root_cause(
                test_case,
                retrieved,
                accepted,
            )

            summary_rows.append({
                "Execution Date": EXECUTION_DATE,
                "Test Case ID": test_case["id"],
                "Query": test_case["query"],
                "Query Type": test_case["expected_type"],
                "Expected Document": expected_doc,
                "Top K": top_k,
                "Similarity Threshold": SIMILARITY_THRESHOLD,
                "Retrieved Document IDs": ";".join(retrieved_ids),
                "Accepted Document IDs": ";".join(accepted_ids),
                "Expected Document Found": expected_found,
                "Expected Document Accepted": expected_accepted,
                "False Positive Documents": ";".join(false_positive_ids),
                "Root Cause": root_cause,
                "Status": status,
                "Latency Ms": round(latency, 2),
            })

            for rank, document in enumerate(retrieved, start=1):

                document_id = document["Document ID"]
                similarity = document["Similarity Score"]

                is_expected = (
                    test_case["expected_type"] == "supported"
                    and document_id == expected_doc
                )

                detail_rows.append({
                    "Execution Date": EXECUTION_DATE,
                    "Test Case ID": test_case["id"],
                    "Query": test_case["query"],
                    "Query Type": test_case["expected_type"],
                    "Expected Document": expected_doc,
                    "Top K": top_k,
                    "Rank": rank,
                    "Retrieved Document": document_id,
                    "Similarity Score": similarity,
                    "Threshold": SIMILARITY_THRESHOLD,
                    "Accepted": similarity >= SIMILARITY_THRESHOLD,
                    "Is Expected Document": is_expected,
                    "Is False Positive": (
                        similarity >= SIMILARITY_THRESHOLD
                        and not is_expected
                    ),
                    "Latency Ms": round(latency, 2),
                })

            print(
                f"{query_number}/{total_queries}"
                f" | {test_case['id']}"
                f" | Top-K={top_k}"
                f" | {status}"
                f" | {root_cause}"
            )

    return detail_rows, summary_rows


# ============================================================
# GENERATE CSV REPORTS
# ============================================================

def generate_reports(detail_rows, summary_rows):

    if not detail_rows or not summary_rows:
        raise RuntimeError("Audit generated no report data.")

    with open(
        DETAIL_REPORT,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(detail_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(detail_rows)

    with open(
        SUMMARY_REPORT,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    total = len(summary_rows)

    passed = sum(
        row["Status"] == "PASS"
        for row in summary_rows
    )

    failed = total - passed

    pass_rate = (
        passed / total * 100
        if total else 0
    )

    false_positive_count = sum(
        bool(row["False Positive Documents"])
        for row in summary_rows
    )

    missed_expected_count = sum(
        row["Query Type"] == "supported"
        and not row["Expected Document Accepted"]
        for row in summary_rows
    )

    print("\n" + "=" * 75)
    print("LESSON 73: PER-QUERY RETRIEVAL AUDIT SUMMARY")
    print("=" * 75)

    print(f"Embedding Model: {MODEL}")
    print(f"Documents: {len(DOCUMENTS)}")
    print(f"Test Cases: {len(TEST_CASES)}")
    print(f"Top-K Values: {TOP_K_VALUES}")
    print(f"Threshold: {SIMILARITY_THRESHOLD}")
    print(f"Audit Executions: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass Rate: {pass_rate:.2f}%")
    print(f"Queries with False Positive Documents: {false_positive_count}")
    print(f"Missed Expected Document Executions: {missed_expected_count}")

    print("\nGenerated reports:")
    print(DETAIL_REPORT)
    print(SUMMARY_REPORT)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("LESSON 73: RAG PER-QUERY RETRIEVAL AUDIT")
    print("=" * 75)

    print(f"Embedding Model: {MODEL}")
    print(f"Documents: {len(DOCUMENTS)}")
    print(f"Test Cases: {len(TEST_CASES)}")

    try:

        indexed_documents = build_index()

        detail_rows, summary_rows = run_audit(
            indexed_documents
        )

        generate_reports(
            detail_rows,
            summary_rows,
        )

        print("\nLESSON 73 COMPLETED SUCCESSFULLY")

    except requests.RequestException as error:

        print("\nOLLAMA REQUEST FAILED")
        print(error)

    except (RuntimeError, ValueError) as error:

        print("\nEVALUATION ERROR")
        print(error)


if __name__ == "__main__":
    main()
