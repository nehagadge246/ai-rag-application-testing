
import csv
import math
import time
from datetime import datetime

import ollama


# ============================================================
# LESSON 62: RAG RETRIEVAL THRESHOLD OPTIMIZATION
# ============================================================

MODEL = "nomic-embed-text:latest"
TOP_K = 3

THRESHOLDS = [0.40, 0.50, 0.60, 0.68, 0.75, 0.85]

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

# Positive cases have one expected relevant document.
# Negative cases have no relevant document in this dataset.

TEST_CASES = [
    {
        "id": "TC01",
        "query": "How many annual leave days do employees get?",
        "expected": "DOC001",
        "type": "SUPPORTED"
    },
    {
        "id": "TC02",
        "query": "What is the sick leave entitlement?",
        "expected": "DOC002",
        "type": "SUPPORTED"
    },
    {
        "id": "TC03",
        "query": "How many days can employees work remotely?",
        "expected": "DOC003",
        "type": "SUPPORTED"
    },
    {
        "id": "TC04",
        "query": "What is the maternity leave duration?",
        "expected": "DOC004",
        "type": "SUPPORTED"
    },
    {
        "id": "TC05",
        "query": "Which public holidays does the organization observe?",
        "expected": "DOC005",
        "type": "SUPPORTED"
    },
    {
        "id": "TC06",
        "query": "What are the employee password security requirements?",
        "expected": "DOC006",
        "type": "SUPPORTED"
    },
    {
        "id": "TC07",
        "query": "How can employees claim business travel expenses?",
        "expected": "DOC007",
        "type": "SUPPORTED"
    },
    {
        "id": "TC08",
        "query": "How frequently are employee performance reviews conducted?",
        "expected": "DOC008",
        "type": "SUPPORTED"
    },
    {
        "id": "TC09",
        "query": "What is the insurance coverage for employees?",
        "expected": None,
        "type": "UNSUPPORTED"
    },
    {
        "id": "TC10",
        "query": "How much annual performance bonus is guaranteed?",
        "expected": None,
        "type": "UNSUPPORTED"
    },
    {
        "id": "TC11",
        "query": "What is the current price of gold?",
        "expected": None,
        "type": "UNSUPPORTED"
    },
    {
        "id": "TC12",
        "query": "What is the employee dental insurance policy?",
        "expected": None,
        "type": "UNSUPPORTED"
    }
]


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(sum(x * x for x in vector_a))
    norm_b = math.sqrt(sum(x * x for x in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def get_embedding(text):
    response = ollama.embeddings(
        model=MODEL,
        prompt=text
    )

    return response["embedding"]


def main():
    print("=" * 70)
    print("RAG RETRIEVAL THRESHOLD OPTIMIZATION TESTING")
    print("=" * 70)

    print("\nSTEP 1: GENERATING DOCUMENT EMBEDDINGS")

    document_embeddings = {}

    indexing_start = time.perf_counter()

    for document in DOCUMENTS:
        document_embeddings[document["id"]] = get_embedding(
            document["text"]
        )

    indexing_time = (
        time.perf_counter() - indexing_start
    ) * 1000

    print(f"Documents indexed: {len(DOCUMENTS)}")
    print(f"Indexing time: {indexing_time:.2f} ms")

    report_rows = []
    summary_rows = []

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

    print(f"Queries embedded: {len(query_embeddings)}")

    print("\nSTEP 3: EVALUATING SIMILARITY THRESHOLDS")

    for threshold in THRESHOLDS:

        print(f"\n{'=' * 55}")
        print(f"SIMILARITY THRESHOLD: {threshold}")
        print(f"{'=' * 55}")

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        threshold_latencies = []

        for test in TEST_CASES:

            ranking_start = time.perf_counter()

            query_embedding = query_embeddings[test["id"]]

            candidates = []

            for document in DOCUMENTS:
                similarity = cosine_similarity(
                    query_embedding,
                    document_embeddings[document["id"]]
                )

                candidates.append({
                    "id": document["id"],
                    "score": similarity
                })

            candidates.sort(
                key=lambda item: item["score"],
                reverse=True
            )

            top_results = candidates[:TOP_K]

            accepted_results = [
                result for result in top_results
                if result["score"] >= threshold
            ]

            ranking_latency = (
                time.perf_counter() - ranking_start
            ) * 1000

            threshold_latencies.append(ranking_latency)

            retrieved_ids = [
                result["id"] for result in accepted_results
            ]

            top_score = top_results[0]["score"]

            # A supported query is detected when its expected
            # document appears in the accepted result set.
            if test["type"] == "SUPPORTED":

                correct_retrieval = (
                    test["expected"] in retrieved_ids
                )

                if correct_retrieval:
                    tp += 1
                    status = "PASS"
                else:
                    fn += 1
                    status = "FAIL"

            else:

                # An unsupported query should return no documents.
                if len(accepted_results) == 0:
                    tn += 1
                    status = "PASS"
                else:
                    fp += 1
                    status = "FAIL"

            report_rows.append({
                "Execution Date": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Test ID": test["id"],
                "Query": test["query"],
                "Query Type": test["type"],
                "Expected Document": test["expected"] or "NONE",
                "Threshold": threshold,
                "Top Similarity Score": round(top_score, 4),
                "Retrieved Documents": ", ".join(retrieved_ids),
                "Retrieved Count": len(retrieved_ids),
                "Query Embedding Latency (ms)": round(
                    query_latencies[test["id"]], 3
                ),
                "Ranking Latency (ms)": round(
                    ranking_latency, 3
                ),
                "Status": status
            })

            print(
                f"{test['id']} | "
                f"{test['type']} | "
                f"Score: {top_score:.4f} | "
                f"Retrieved: {retrieved_ids} | "
                f"{status}"
            )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )

        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        accuracy = (
            (tp + tn) / len(TEST_CASES)
        )

        avg_latency = (
            sum(threshold_latencies) /
            len(threshold_latencies)
        )

        summary_rows.append({
            "Execution Date": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Threshold": threshold,
            "Total Tests": len(TEST_CASES),
            "True Positive": tp,
            "False Positive": fp,
            "False Negative": fn,
            "True Negative": tn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "Specificity": round(specificity, 4),
            "F1 Score": round(f1_score, 4),
            "Accuracy (%)": round(accuracy * 100, 2),
            "Average Ranking Latency (ms)": round(
                avg_latency, 3
            )
        })

    print("\nSTEP 4: THRESHOLD COMPARISON")

    for row in summary_rows:
        print(
            f"Threshold: {row['Threshold']} | "
            f"Precision: {row['Precision']} | "
            f"Recall: {row['Recall']} | "
            f"F1: {row['F1 Score']} | "
            f"Accuracy: {row['Accuracy (%)']}%"
        )

    # Select the threshold with the highest F1 score.
    # Ties are resolved in favour of higher specificity.
    best_threshold = max(
        summary_rows,
        key=lambda row: (
            row["F1 Score"],
            row["Specificity"]
        )
    )

    print("\nSTEP 5: BEST THRESHOLD")

    print(
        f"Best threshold by F1 and specificity: "
        f"{best_threshold['Threshold']}"
    )

    print(f"F1 Score: {best_threshold['F1 Score']}")
    print(f"Precision: {best_threshold['Precision']}")
    print(f"Recall: {best_threshold['Recall']}")
    print(f"Specificity: {best_threshold['Specificity']}")

    print("\nSTEP 6: EXPORTING CSV REPORTS")

    with open(
        "rag_threshold_optimization_report.csv",
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
        "rag_threshold_optimization_summary.csv",
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

    print("\n" + "=" * 70)
    print("RAG THRESHOLD OPTIMIZATION COMPLETED")
    print("=" * 70)

    print(f"Total detailed evaluations: {len(report_rows)}")
    print(f"Thresholds evaluated: {len(THRESHOLDS)}")
    print(f"Best threshold: {best_threshold['Threshold']}")

    print(
        "Detailed report: "
        "rag_threshold_optimization_report.csv"
    )

    print(
        "Summary report: "
        "rag_threshold_optimization_summary.csv"
    )


if __name__ == "__main__":
    main()
