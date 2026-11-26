
import csv
import time
import math
from datetime import datetime

import ollama


# ============================================================
# RAG RERANKING EVALUATION
# ============================================================

MODEL = "nomic-embed-text:latest"
TOP_K = 3
RERANK_CANDIDATES = 5

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

TEST_CASES = [
    {"id": "TC01", "query": "How many annual leave days do employees get?", "expected": "DOC001"},
    {"id": "TC02", "query": "What is the sick leave entitlement?", "expected": "DOC002"},
    {"id": "TC03", "query": "How many days can employees work remotely?", "expected": "DOC003"},
    {"id": "TC04", "query": "What is the maternity leave duration?", "expected": "DOC004"},
    {"id": "TC05", "query": "Which public holidays does the organization observe?", "expected": "DOC005"},
    {"id": "TC06", "query": "What are the employee password security requirements?", "expected": "DOC006"},
    {"id": "TC07", "query": "How can employees claim business travel expenses?", "expected": "DOC007"},
    {"id": "TC08", "query": "How frequently are employee performance reviews conducted?", "expected": "DOC008"},
    {"id": "TC09", "query": "What is the parental leave entitlement?", "expected": "DOC004"},
    {"id": "TC10", "query": "How do employees request time off for illness?", "expected": "DOC002"}
]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def get_embedding(text):
    response = ollama.embeddings(
        model=MODEL,
        prompt=text
    )
    return response["embedding"]


def keyword_overlap(query, document):
    query_words = set(query.lower().split())
    document_words = set(document.lower().split())

    stop_words = {
        "the", "is", "a", "an", "of", "do", "does",
        "how", "what", "are", "can", "for", "and",
        "to", "in", "employees", "employee"
    }

    query_words -= stop_words
    document_words -= stop_words

    if not query_words:
        return 0.0

    return len(query_words & document_words) / len(query_words)


def reciprocal_rank(results, expected):
    for rank, result in enumerate(results, start=1):
        if result["id"] == expected:
            return 1 / rank

    return 0.0


def precision_at_k(results, expected, k):
    relevant = sum(
        1 for result in results[:k]
        if result["id"] == expected
    )
    return relevant / k


def evaluate_retrieval(results, expected):
    found = any(result["id"] == expected for result in results)

    return {
        "hit": int(found),
        "top1": int(results[0]["id"] == expected),
        "mrr": reciprocal_rank(results, expected),
        "precision": precision_at_k(results, expected, TOP_K),
        "recall": int(found)
    }


def main():
    print("=" * 65)
    print("RAG RERANKING EVALUATION TESTING")
    print("=" * 65)

    print("\nSTEP 1: GENERATING DOCUMENT EMBEDDINGS")

    document_embeddings = {}

    indexing_start = time.perf_counter()

    for document in DOCUMENTS:
        document_embeddings[document["id"]] = get_embedding(
            document["text"]
        )

    indexing_time = (time.perf_counter() - indexing_start) * 1000

    print(f"Documents indexed: {len(DOCUMENTS)}")
    print(f"Indexing time: {indexing_time:.2f} ms")

    report_rows = []

    baseline_metrics = []
    reranked_metrics = []

    baseline_latencies = []
    reranking_latencies = []

    print("\nSTEP 2: RUNNING BASELINE AND RERANKING TESTS")

    for test in TEST_CASES:
        query_start = time.perf_counter()

        query_embedding = get_embedding(test["query"])

        embedding_latency = (
            time.perf_counter() - query_start
        ) * 1000

        ranking_start = time.perf_counter()

        candidates = []

        for document in DOCUMENTS:
            similarity = cosine_similarity(
                query_embedding,
                document_embeddings[document["id"]]
            )

            candidates.append({
                "id": document["id"],
                "text": document["text"],
                "vector_score": similarity
            })

        baseline_results = sorted(
            candidates,
            key=lambda item: item["vector_score"],
            reverse=True
        )[:TOP_K]

        baseline_latency = (
            time.perf_counter() - ranking_start
        ) * 1000

        # Select a wider candidate set before reranking.
        rerank_candidates = sorted(
            candidates,
            key=lambda item: item["vector_score"],
            reverse=True
        )[:RERANK_CANDIDATES]

        reranking_start = time.perf_counter()

        for candidate in rerank_candidates:
            candidate["keyword_score"] = keyword_overlap(
                test["query"],
                candidate["text"]
            )

            # Experimental combined reranking score.
            candidate["rerank_score"] = (
                0.70 * candidate["vector_score"]
                + 0.30 * candidate["keyword_score"]
            )

        reranked_results = sorted(
            rerank_candidates,
            key=lambda item: item["rerank_score"],
            reverse=True
        )[:TOP_K]

        reranking_latency = (
            time.perf_counter() - reranking_start
        ) * 1000

        baseline_result = evaluate_retrieval(
            baseline_results,
            test["expected"]
        )

        reranked_result = evaluate_retrieval(
            reranked_results,
            test["expected"]
        )

        baseline_metrics.append(baseline_result)
        reranked_metrics.append(reranked_result)

        baseline_latencies.append(
            embedding_latency + baseline_latency
        )

        reranking_latencies.append(
            reranking_latency
        )

        baseline_ids = [
            item["id"] for item in baseline_results
        ]

        reranked_ids = [
            item["id"] for item in reranked_results
        ]

        status = (
            "PASS"
            if reranked_result["hit"] == 1
            else "FAIL"
        )

        report_rows.append({
            "Test ID": test["id"],
            "Query": test["query"],
            "Expected Document": test["expected"],
            "Baseline Results": ", ".join(baseline_ids),
            "Reranked Results": ", ".join(reranked_ids),
            "Baseline Top1": baseline_result["top1"],
            "Reranked Top1": reranked_result["top1"],
            "Baseline MRR": round(baseline_result["mrr"], 4),
            "Reranked MRR": round(reranked_result["mrr"], 4),
            "Baseline Precision@3": round(baseline_result["precision"], 4),
            "Reranked Precision@3": round(reranked_result["precision"], 4),
            "Baseline Recall@3": baseline_result["recall"],
            "Reranked Recall@3": reranked_result["recall"],
            "Embedding Latency (ms)": round(embedding_latency, 3),
            "Baseline Ranking Latency (ms)": round(baseline_latency, 3),
            "Reranking Latency (ms)": round(reranking_latency, 3),
            "Status": status
        })

        print(
            f"{test['id']} | "
            f"Expected: {test['expected']} | "
            f"Baseline: {baseline_ids} | "
            f"Reranked: {reranked_ids} | "
            f"{status}"
        )

    print("\nSTEP 3: CALCULATING SUMMARY METRICS")

    def average(metrics, key):
        return sum(item[key] for item in metrics) / len(metrics)

    summary_rows = []

    for name, metrics in [
        ("Baseline Vector Search", baseline_metrics),
        ("Hybrid Reranking", reranked_metrics)
    ]:
        summary_rows.append({
            "Execution Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Method": name,
            "Total Tests": len(metrics),
            "Top1 Accuracy (%)": round(
                average(metrics, "top1") * 100, 2
            ),
            "Hit Rate (%)": round(
                average(metrics, "hit") * 100, 2
            ),
            "Precision@3": round(
                average(metrics, "precision"), 4
            ),
            "Recall@3": round(
                average(metrics, "recall"), 4
            ),
            "MRR": round(
                average(metrics, "mrr"), 4
            ),
            "Average Retrieval Latency (ms)": round(
                sum(baseline_latencies) / len(baseline_latencies), 3
            ) if name == "Baseline Vector Search" else "",
            "Average Reranking Latency (ms)": round(
                sum(reranking_latencies) / len(reranking_latencies), 3
            ) if name == "Hybrid Reranking" else ""
        })

    print("\nBASELINE VS RERANKING")

    for row in summary_rows:
        print("\n" + row["Method"])
        for key, value in row.items():
            if key not in ["Execution Date", "Method"]:
                print(f"{key}: {value}")

    print("\nSTEP 4: EXPORTING CSV REPORTS")

    with open(
        "rag_reranking_report.csv",
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
        "rag_reranking_summary.csv",
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

    print("\n" + "=" * 65)
    print("RAG RERANKING EVALUATION COMPLETED")
    print("=" * 65)

    print(f"Total test cases: {len(TEST_CASES)}")
    print("Detailed report: rag_reranking_report.csv")
    print("Summary report: rag_reranking_summary.csv")


if __name__ == "__main__":
    main()
