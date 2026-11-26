
"""
LESSON 72: RAG RETRIEVAL PRECISION OPTIMIZATION

Objectives:
1. Compare different Top-K retrieval configurations.
2. Evaluate multiple similarity thresholds.
3. Calculate Precision@K, Recall@K, MRR and F1.
4. Measure unsupported-query rejection.
5. Compare retrieval latency.
6. Identify the best retrieval configuration.
7. Generate detailed and summary CSV reports.

This is a reconstructed implementation based on the
original Lesson 72 experiment design and saved CSV results.
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

THRESHOLD_VALUES = [
    0.40, 0.50, 0.55, 0.60,
    0.65, 0.70, 0.75
]

REQUEST_TIMEOUT = 30

DETAIL_REPORT = "rag_retrieval_optimization_report.csv"
SUMMARY_REPORT = "rag_retrieval_optimization_summary.csv"

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
# RETRIEVAL
# ============================================================

def retrieve_documents(
    query,
    indexed_documents,
    top_k
):

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
            "Similarity Score": similarity,
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
# EVALUATION METRICS
# ============================================================

def calculate_metrics(
    test_cases,
    retrieval_results,
    threshold
):

    supported_queries = [
        case for case in test_cases
        if case["expected_type"] == "supported"
    ]

    unsupported_queries = [
        case for case in test_cases
        if case["expected_type"] == "unsupported"
    ]

    relevant_retrieved = 0
    relevant_expected = len(supported_queries)

    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    reciprocal_rank_total = 0.0

    for case in test_cases:

        retrieved = retrieval_results[case["id"]]["documents"]

        accepted = [
            document
            for document in retrieved
            if document["Similarity Score"] >= threshold
        ]

        accepted_ids = [
            document["Document ID"]
            for document in accepted
        ]

        if case["expected_type"] == "supported":

            expected_doc = case["expected_doc"]

            if expected_doc in accepted_ids:

                relevant_retrieved += 1

                rank = accepted_ids.index(expected_doc) + 1

                reciprocal_rank_total += 1 / rank

            else:

                false_negatives += 1

            false_positives += sum(
                document_id != expected_doc
                for document_id in accepted_ids
            )

        else:

            if accepted:
                false_positives += len(accepted)
            else:
                true_negatives += 1

    # Precision uses accepted document candidates.
    # A query with no accepted candidates contributes no
    # false-positive document count.

    accepted_document_count = (
        relevant_retrieved + false_positives
    )

    precision = (
        relevant_retrieved / accepted_document_count
        if accepted_document_count else 0.0
    )

    recall = (
        relevant_retrieved / relevant_expected
        if relevant_expected else 0.0
    )

    mrr = (
        reciprocal_rank_total / len(supported_queries)
        if supported_queries else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    unsupported_rejected = sum(
        not retrieval_results[case["id"]]["accepted_by_threshold"]
        for case in unsupported_queries
    )

    unsupported_rejection_rate = (
        unsupported_rejected / len(unsupported_queries) * 100
        if unsupported_queries else 0.0
    )

    return {
        "Supported Queries": len(supported_queries),
        "Unsupported Queries": len(unsupported_queries),
        "Relevant Retrieved": relevant_retrieved,
        "Relevant Expected": relevant_expected,
        "False Positives": false_positives,
        "False Negatives": false_negatives,
        "True Negatives": true_negatives,
        "Precision At K": round(precision, 4),
        "Recall At K": round(recall, 4),
        "MRR": round(mrr, 4),
        "F1 Score": round(f1, 4),
        "Unsupported Rejection Rate": round(
            unsupported_rejection_rate, 2
        ),
    }


# ============================================================
# RUN OPTIMIZATION
# ============================================================

def run_optimization(indexed_documents):

    detail_rows = []

    print("\nStarting retrieval optimization...")
    print("-" * 85)

    # Cache retrieval results for each Top-K so that
    # different thresholds use the same similarity scores.

    retrieval_cache = {}

    for top_k in TOP_K_VALUES:

        for case in TEST_CASES:

            retrieved, latency = retrieve_documents(
                case["query"],
                indexed_documents,
                top_k,
            )

            retrieval_cache[(top_k, case["id"])] = {
                "documents": retrieved,
                "latency": latency,
            }

    summary_rows = []

    total_configurations = (
        len(TOP_K_VALUES) * len(THRESHOLD_VALUES)
    )

    configuration_number = 0

    for top_k in TOP_K_VALUES:

        for threshold in THRESHOLD_VALUES:

            configuration_number += 1

            retrieval_results = {}

            for case in TEST_CASES:

                cached = retrieval_cache[
                    (top_k, case["id"])
                ]

                retrieved = cached["documents"]

                accepted = [
                    document
                    for document in retrieved
                    if document["Similarity Score"] >= threshold
                ]

                accepted_ids = [
                    document["Document ID"]
                    for document in accepted
                ]

                retrieval_results[case["id"]] = {
                    "documents": retrieved,
                    "accepted_by_threshold": bool(accepted),
                }

                for rank, document in enumerate(
                    retrieved, start=1
                ):

                    document_id = document["Document ID"]
                    score = document["Similarity Score"]

                    expected_doc = case["expected_doc"]

                    is_expected = (
                        case["expected_type"] == "supported"
                        and document_id == expected_doc
                    )

                    is_accepted = score >= threshold

                    detail_rows.append({
                        "Execution Date": EXECUTION_DATE,
                        "Top K": top_k,
                        "Similarity Threshold": threshold,
                        "Test Case ID": case["id"],
                        "Query": case["query"],
                        "Query Type": case["expected_type"],
                        "Expected Document": expected_doc,
                        "Rank": rank,
                        "Retrieved Document": document_id,
                        "Similarity Score": round(score, 6),
                        "Accepted": is_accepted,
                        "Is Expected Document": is_expected,
                        "Is False Positive": (
                            is_accepted and not is_expected
                        ),
                        "Latency Ms": round(
                            cached["latency"], 2
                        ),
                    })

            metrics = calculate_metrics(
                TEST_CASES,
                retrieval_results,
                threshold,
            )

            average_latency = sum(
                retrieval_cache[
                    (top_k, case["id"])
                ]["latency"]
                for case in TEST_CASES
            ) / len(TEST_CASES)

            summary_rows.append({
                "Execution Date": EXECUTION_DATE,
                "Top K": top_k,
                "Similarity Threshold": threshold,
                **metrics,
                "Average Retrieval Latency Ms": round(
                    average_latency, 2
                ),
            })

            print(
                f"{configuration_number}/{total_configurations}"
                f" | Top-K={top_k}"
                f" | Threshold={threshold}"
                f" | Precision={metrics['Precision At K']}"
                f" | Recall={metrics['Recall At K']}"
                f" | F1={metrics['F1 Score']}"
            )

    return detail_rows, summary_rows


# ============================================================
# SELECT BEST CONFIGURATION
# ============================================================

def select_best_configuration(summary_rows):

    if not summary_rows:
        raise RuntimeError(
            "No optimization results were generated."
        )

    return max(
        summary_rows,
        key=lambda row: (
            row["F1 Score"],
            row["Unsupported Rejection Rate"],
            row["MRR"],
            row["Precision At K"],
        ),
    )


# ============================================================
# GENERATE REPORTS
# ============================================================

def generate_reports(
    detail_rows,
    summary_rows,
):

    if not detail_rows or not summary_rows:
        raise RuntimeError(
            "Optimization generated no report data."
        )

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

    best = select_best_configuration(summary_rows)

    summary_row = {
        "Execution Date": EXECUTION_DATE,
        "Embedding Model": MODEL,
        "Documents": len(DOCUMENTS),
        "Test Cases": len(TEST_CASES),
        "Configurations Tested": len(summary_rows),
        "Best Top K": best["Top K"],
        "Best Threshold": best["Similarity Threshold"],
        "Best Precision At K": best["Precision At K"],
        "Best Recall At K": best["Recall At K"],
        "Best MRR": best["MRR"],
        "Best F1 Score": best["F1 Score"],
        "Unsupported Rejection Rate": best[
            "Unsupported Rejection Rate"
        ],
        "Average Retrieval Latency Ms": best[
            "Average Retrieval Latency Ms"
        ],
        "Selection Method": (
            "Highest F1, then unsupported rejection, "
            "MRR and precision"
        ),
        "Evaluation Note": (
            "Best configuration on a small controlled dataset; "
            "not a production guarantee."
        ),
    }

    with open(
        SUMMARY_REPORT,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_row.keys()),
        )

        writer.writeheader()
        writer.writerow(summary_row)

    print("\n" + "=" * 75)
    print("LESSON 72: RETRIEVAL OPTIMIZATION SUMMARY")
    print("=" * 75)

    print(f"Embedding Model: {MODEL}")
    print(f"Documents: {len(DOCUMENTS)}")
    print(f"Test Cases: {len(TEST_CASES)}")
    print(f"Configurations Tested: {len(summary_rows)}")

    print("\nBest Configuration:")
    print(f"Top K: {best['Top K']}")
    print(f"Threshold: {best['Similarity Threshold']}")
    print(f"Precision@K: {best['Precision At K']}")
    print(f"Recall@K: {best['Recall At K']}")
    print(f"MRR: {best['MRR']}")
    print(f"F1 Score: {best['F1 Score']}")
    print(
        "Unsupported Rejection Rate: "
        f"{best['Unsupported Rejection Rate']}%"
    )
    print(
        "Average Retrieval Latency: "
        f"{best['Average Retrieval Latency Ms']} ms"
    )

    print("\nGenerated reports:")
    print(DETAIL_REPORT)
    print(SUMMARY_REPORT)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("LESSON 72: RAG RETRIEVAL PRECISION OPTIMIZATION")
    print("=" * 75)

    print(f"Embedding Model: {MODEL}")
    print(f"Documents: {len(DOCUMENTS)}")
    print(f"Test Cases: {len(TEST_CASES)}")
    print(f"Top-K Values: {TOP_K_VALUES}")
    print(f"Threshold Values: {THRESHOLD_VALUES}")

    try:

        indexed_documents = build_index()

        detail_rows, summary_rows = run_optimization(
            indexed_documents
        )

        generate_reports(
            detail_rows,
            summary_rows,
        )

        print("\nLESSON 72 COMPLETED SUCCESSFULLY")

    except requests.RequestException as error:

        print("\nOLLAMA REQUEST FAILED")
        print(error)

    except (RuntimeError, ValueError) as error:

        print("\nEVALUATION ERROR")
        print(error)


if __name__ == "__main__":
    main()
