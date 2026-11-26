
import csv
import math
import time
from datetime import datetime

import ollama


# ============================================================
# LESSON 63: RAG SEMANTIC SEARCH EVALUATION
# ============================================================

MODEL = "nomic-embed-text:latest"
TOP_K_VALUES = [1, 3, 5]

DOCUMENTS = [
    {
        "id": "DOC001",
        "text": "Annual leave policy: Employees receive 15 days of paid annual leave per year."
    },
    {
        "id": "DOC002",
        "text": "Sick leave policy: Employees receive 10 days of paid sick leave annually."
    },
    {
        "id": "DOC003",
        "text": "Remote work policy: Employees may work remotely up to three days per week."
    },
    {
        "id": "DOC004",
        "text": "Parental leave policy: Eligible employees receive 26 weeks of maternity leave."
    },
    {
        "id": "DOC005",
        "text": "Public holiday policy: The organization observes officially declared public holidays."
    },
    {
        "id": "DOC006",
        "text": "IT security policy: Employees must use strong passwords and multi-factor authentication."
    },
    {
        "id": "DOC007",
        "text": "Travel reimbursement policy: Approved business travel expenses can be claimed with receipts."
    },
    {
        "id": "DOC008",
        "text": "Performance review policy: Employee performance is reviewed annually by managers."
    }
]


# Each supported query has one expected relevant document.
# Unsupported queries have no relevant document in this dataset.

TEST_CASES = [
    {
        "id": "TC01",
        "category": "EXACT_MATCH",
        "query": "Annual leave policy",
        "expected": "DOC001",
        "type": "SUPPORTED"
    },
    {
        "id": "TC02",
        "category": "SYNONYM",
        "query": "How many vacation days can staff take?",
        "expected": "DOC001",
        "type": "SUPPORTED"
    },
    {
        "id": "TC03",
        "category": "PARAPHRASED",
        "query": "What happens when an employee needs time off because they are ill?",
        "expected": "DOC002",
        "type": "SUPPORTED"
    },
    {
        "id": "TC04",
        "category": "DIFFERENT_STRUCTURE",
        "query": "Can I perform my job duties from home several days each week?",
        "expected": "DOC003",
        "type": "SUPPORTED"
    },
    {
        "id": "TC05",
        "category": "SEMANTIC",
        "query": "How long can a mother take leave after having a baby?",
        "expected": "DOC004",
        "type": "SUPPORTED"
    },
    {
        "id": "TC06",
        "category": "SYNONYM",
        "query": "Which official days off are recognized by the company?",
        "expected": "DOC005",
        "type": "SUPPORTED"
    },
    {
        "id": "TC07",
        "category": "PARAPHRASED",
        "query": "How should workers protect their accounts and login credentials?",
        "expected": "DOC006",
        "type": "SUPPORTED"
    },
    {
        "id": "TC08",
        "category": "DIFFERENT_STRUCTURE",
        "query": "What is the process for getting money back after an approved work trip?",
        "expected": "DOC007",
        "type": "SUPPORTED"
    },
    {
        "id": "TC09",
        "category": "SEMANTIC",
        "query": "How often does management assess staff performance?",
        "expected": "DOC008",
        "type": "SUPPORTED"
    },
    {
        "id": "TC10",
        "category": "UNSUPPORTED",
        "query": "What is the employee health insurance coverage?",
        "expected": None,
        "type": "UNSUPPORTED"
    },
    {
        "id": "TC11",
        "category": "UNSUPPORTED",
        "query": "What is the current stock market price?",
        "expected": None,
        "type": "UNSUPPORTED"
    },
    {
        "id": "TC12",
        "category": "UNSUPPORTED",
        "query": "Does the company provide free meals?",
        "expected": None,
        "type": "UNSUPPORTED"
    }
]


def get_embedding(text):
    response = ollama.embeddings(
        model=MODEL,
        prompt=text
    )
    return response["embedding"]


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def main():

    print("=" * 75)
    print("RAG SEMANTIC SEARCH EVALUATION TESTING")
    print("=" * 75)

    # --------------------------------------------------------
    # STEP 1: DOCUMENT INDEXING
    # --------------------------------------------------------

    print("\nSTEP 1: GENERATING DOCUMENT EMBEDDINGS")

    document_embeddings = {}

    indexing_start = time.perf_counter()

    for document in DOCUMENTS:

        document_embeddings[document["id"]] = get_embedding(
            document["text"]
        )

    indexing_latency = (
        time.perf_counter() - indexing_start
    ) * 1000

    print(f"Documents indexed: {len(DOCUMENTS)}")
    print(f"Indexing latency: {indexing_latency:.2f} ms")

    # --------------------------------------------------------
    # STEP 2: QUERY EMBEDDINGS
    # --------------------------------------------------------

    print("\nSTEP 2: GENERATING QUERY EMBEDDINGS")

    query_embeddings = {}
    query_latencies = {}

    for test in TEST_CASES:

        start = time.perf_counter()

        query_embeddings[test["id"]] = get_embedding(
            test["query"]
        )

        query_latencies[test["id"]] = (
            time.perf_counter() - start
        ) * 1000

    print(f"Queries processed: {len(query_embeddings)}")

    # --------------------------------------------------------
    # STEP 3: SEMANTIC SEARCH
    # --------------------------------------------------------

    print("\nSTEP 3: EXECUTING SEMANTIC SEARCH")

    all_rankings = {}

    for test in TEST_CASES:

        query_vector = query_embeddings[test["id"]]

        results = []

        for document in DOCUMENTS:

            score = cosine_similarity(
                query_vector,
                document_embeddings[document["id"]]
            )

            results.append({
                "document_id": document["id"],
                "similarity": score
            })

        results.sort(
            key=lambda item: item["similarity"],
            reverse=True
        )

        all_rankings[test["id"]] = results

    print("Semantic search completed.")

    # --------------------------------------------------------
    # STEP 4: TOP-K EVALUATION
    # --------------------------------------------------------

    print("\nSTEP 4: EVALUATING TOP-K SEARCH RESULTS")

    report_rows = []
    summary_rows = []

    for top_k in TOP_K_VALUES:

        print(f"\n{'=' * 60}")
        print(f"TOP-K VALUE: {top_k}")
        print(f"{'=' * 60}")

        total_precision = 0.0
        total_recall = 0.0
        total_reciprocal_rank = 0.0

        supported_hits = 0
        unsupported_rejections = 0
        supported_count = 0
        unsupported_count = 0

        top1_correct = 0
        total_latency = 0.0

        for test in TEST_CASES:

            ranking = all_rankings[test["id"]]

            top_results = ranking[:top_k]

            retrieved_ids = [
                item["document_id"]
                for item in top_results
            ]

            scores = [
                item["similarity"]
                for item in top_results
            ]

            ranking_latency = (
                sum(
                    1 for _ in ranking
                )
            )

            # Precision@K:
            # Number of relevant documents retrieved / K.
            if test["type"] == "SUPPORTED":

                supported_count += 1

                relevant_count = sum(
                    1 for document_id in retrieved_ids
                    if document_id == test["expected"]
                )

                precision = relevant_count / top_k

                recall = float(
                    test["expected"] in retrieved_ids
                )

                if test["expected"] in retrieved_ids:

                    supported_hits += 1

                    rank = (
                        retrieved_ids.index(test["expected"]) + 1
                    )

                    reciprocal_rank = 1 / rank

                else:

                    reciprocal_rank = 0.0

                if retrieved_ids and (
                    retrieved_ids[0] == test["expected"]
                ):
                    top1_correct += 1

                total_precision += precision
                total_recall += recall
                total_reciprocal_rank += reciprocal_rank

                status = (
                    "PASS"
                    if test["expected"] in retrieved_ids
                    else "FAIL"
                )

            else:

                unsupported_count += 1

                precision = 0.0
                recall = 0.0
                reciprocal_rank = 0.0

                # This benchmark has no expected document for
                # unsupported questions. A retrieved result is
                # treated as an unsupported-query false acceptance.
                if len(retrieved_ids) == 0:

                    unsupported_rejections += 1
                    status = "PASS"

                else:

                    status = "REVIEW"

            query_latency = query_latencies[test["id"]]

            total_latency += query_latency

            report_rows.append({
                "Execution Date": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Test ID": test["id"],
                "Category": test["category"],
                "Query": test["query"],
                "Query Type": test["type"],
                "Expected Document": test["expected"] or "NONE",
                "Top K": top_k,
                "Retrieved Documents": ", ".join(retrieved_ids),
                "Similarity Scores": ", ".join(
                    str(round(score, 4)) for score in scores
                ),
                "Precision At K": round(precision, 4),
                "Recall At K": round(recall, 4),
                "Reciprocal Rank": round(reciprocal_rank, 4),
                "Query Embedding Latency (ms)": round(
                    query_latency, 3
                ),
                "Status": status
            })

            print(
                f"{test['id']} | {test['category']} | "
                f"Retrieved: {retrieved_ids} | {status}"
            )

        # Metrics are calculated on supported queries only.
        precision_at_k = (
            total_precision / supported_count
            if supported_count else 0.0
        )

        recall_at_k = (
            total_recall / supported_count
            if supported_count else 0.0
        )

        mrr = (
            total_reciprocal_rank / supported_count
            if supported_count else 0.0
        )

        hit_rate = (
            supported_hits / supported_count
            if supported_count else 0.0
        )

        top1_accuracy = (
            top1_correct / supported_count
            if supported_count else 0.0
        )

        unsupported_rejection_rate = (
            unsupported_rejections / unsupported_count
            if unsupported_count else 0.0
        )

        average_latency = (
            total_latency / len(TEST_CASES)
        )

        summary_rows.append({
            "Execution Date": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Top K": top_k,
            "Total Test Cases": len(TEST_CASES),
            "Supported Queries": supported_count,
            "Unsupported Queries": unsupported_count,
            "Precision At K": round(precision_at_k, 4),
            "Recall At K": round(recall_at_k, 4),
            "MRR": round(mrr, 4),
            "Hit Rate": round(hit_rate, 4),
            "Top 1 Accuracy": round(top1_accuracy, 4),
            "Unsupported Rejection Rate": round(
                unsupported_rejection_rate, 4
            ),
            "Average Query Embedding Latency (ms)": round(
                average_latency, 3
            )
        })

    # --------------------------------------------------------
    # STEP 5: SUMMARY
    # --------------------------------------------------------

    print("\nSTEP 5: SEMANTIC SEARCH PERFORMANCE SUMMARY")

    for row in summary_rows:

        print(
            f"Top K: {row['Top K']} | "
            f"Precision: {row['Precision At K']} | "
            f"Recall: {row['Recall At K']} | "
            f"MRR: {row['MRR']} | "
            f"Hit Rate: {row['Hit Rate']} | "
            f"Top 1: {row['Top 1 Accuracy']}"
        )

    # --------------------------------------------------------
    # STEP 6: EXPORT REPORTS
    # --------------------------------------------------------

    print("\nSTEP 6: EXPORTING CSV REPORTS")

    with open(
        "rag_semantic_search_report.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=report_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(report_rows)

    with open(
        "rag_semantic_search_summary.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 75)
    print("RAG SEMANTIC SEARCH EVALUATION COMPLETED")
    print("=" * 75)

    print(f"Total detailed evaluations: {len(report_rows)}")
    print(f"Top-K configurations tested: {len(TOP_K_VALUES)}")

    print("Detailed report: rag_semantic_search_report.csv")
    print("Summary report: rag_semantic_search_summary.csv")


if __name__ == "__main__":
    main()
