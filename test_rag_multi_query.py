
import csv
import time
from datetime import datetime

import ollama
import numpy as np

# ============================================================
# LESSON 64: RAG QUERY EXPANSION AND MULTI-QUERY RETRIEVAL
# ============================================================

MODEL = "nomic-embed-text:latest"
TOP_K = 3
THRESHOLD = 0.68

DOCUMENTS = {
    "DOC001": "Employees receive 15 annual leave days per year.",
    "DOC002": "Employees can take up to 10 sick leave days annually.",
    "DOC003": "Employees may work remotely up to three days per week.",
    "DOC004": "Eligible employees receive 26 weeks of parental leave.",
    "DOC005": "The company observes 12 public holidays each year.",
    "DOC006": "Employees must use strong passwords and enable multi-factor authentication.",
    "DOC007": "Business travel expenses must be submitted with receipts for reimbursement.",
    "DOC008": "Employee performance reviews are conducted twice a year."
}

# Each query has a known expected document.
# Expansion variations are intentionally predefined so that
# this experiment isolates retrieval behavior from LLM variability.

TEST_CASES = [
    {
        "id": "TC01",
        "query": "annual leave",
        "expected": "DOC001",
        "variations": [
            "How many annual vacation days do employees receive?",
            "What is the yearly paid leave entitlement?",
            "How many days of vacation are provided each year?"
        ]
    },
    {
        "id": "TC02",
        "query": "sick days",
        "expected": "DOC002",
        "variations": [
            "How many sick leave days can employees take?",
            "What is the annual medical leave allowance?",
            "How much leave is available when employees are ill?"
        ]
    },
    {
        "id": "TC03",
        "query": "work from home",
        "expected": "DOC003",
        "variations": [
            "What is the remote working policy?",
            "How many days can staff work remotely?",
            "Can employees work from their home?"
        ]
    },
    {
        "id": "TC04",
        "query": "maternity leave",
        "expected": "DOC004",
        "variations": [
            "What parental leave is available to eligible employees?",
            "How long is the leave available after having a child?",
            "What is the company's family leave policy?"
        ]
    },
    {
        "id": "TC05",
        "query": "holidays",
        "expected": "DOC005",
        "variations": [
            "How many public holidays are observed?",
            "What are the company's official days off?",
            "How many holidays does the organization provide annually?"
        ]
    },
    {
        "id": "TC06",
        "query": "password rules",
        "expected": "DOC006",
        "variations": [
            "What are the employee account security requirements?",
            "Do employees need multi-factor authentication?",
            "What security practices are required for passwords?"
        ]
    },
    {
        "id": "TC07",
        "query": "travel costs",
        "expected": "DOC007",
        "variations": [
            "How can employees claim business travel expenses?",
            "What documents are needed for travel reimbursement?",
            "How are official trip expenses reimbursed?"
        ]
    },
    {
        "id": "TC08",
        "query": "performance review",
        "expected": "DOC008",
        "variations": [
            "How often are employee appraisals conducted?",
            "When does management assess employee performance?",
            "What is the schedule for staff evaluations?"
        ]
    },
    {
        "id": "TC09",
        "query": "employee bonus",
        "expected": None,
        "variations": [
            "What annual bonus do employees receive?",
            "How much incentive payment is provided?",
            "What is the employee bonus policy?"
        ]
    },
    {
        "id": "TC10",
        "query": "health insurance",
        "expected": None,
        "variations": [
            "What medical insurance coverage is offered?",
            "Does the company provide health insurance?",
            "What healthcare benefits are available?"
        ]
    },
    {
        "id": "TC11",
        "query": "gold prices",
        "expected": None,
        "variations": [
            "What is the current price of gold?",
            "How much does one gram of gold cost?",
            "What are today's precious metal rates?"
        ]
    },
    {
        "id": "TC12",
        "query": "office parking",
        "expected": None,
        "variations": [
            "Where can employees park their vehicles?",
            "Is parking available at the office?",
            "What are the workplace parking arrangements?"
        ]
    }
]


def get_embedding(text):
    response = ollama.embeddings(
        model=MODEL,
        prompt=text
    )
    return np.array(response["embedding"], dtype=float)


def cosine_similarity(vector_a, vector_b):
    denominator = (
        np.linalg.norm(vector_a) *
        np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b) / denominator
    )


def retrieve(query, document_embeddings):
    query_embedding = get_embedding(query)

    scores = []

    for doc_id, doc_embedding in document_embeddings.items():
        score = cosine_similarity(
            query_embedding,
            doc_embedding
        )

        scores.append((doc_id, score))

    scores.sort(key=lambda item: item[1], reverse=True)

    return scores[:TOP_K]


def evaluate_retrieval(results, expected):
    retrieved_ids = [doc_id for doc_id, score in results]

    if expected is None:
        return {
            "status": "REVIEW" if results else "REJECTED",
            "hit": 0,
            "precision": 0.0,
            "reciprocal_rank": 0.0,
            "top1_correct": 0
        }

    hit = int(expected in retrieved_ids)

    precision = hit / TOP_K

    reciprocal_rank = 0.0

    if expected in retrieved_ids:
        rank = retrieved_ids.index(expected) + 1
        reciprocal_rank = 1 / rank

    return {
        "status": "PASS" if hit else "FAIL",
        "hit": hit,
        "precision": precision,
        "reciprocal_rank": reciprocal_rank,
        "top1_correct": int(
            bool(retrieved_ids) and retrieved_ids[0] == expected
        )
    }


def main():
    print("\n===== RAG MULTI-QUERY RETRIEVAL TESTING =====")

    print("\nLoading document embeddings...")

    document_embeddings = {}

    indexing_start = time.perf_counter()

    for doc_id, content in DOCUMENTS.items():
        document_embeddings[doc_id] = get_embedding(content)

    indexing_time = (
        time.perf_counter() - indexing_start
    ) * 1000

    print(f"Indexed documents: {len(document_embeddings)}")
    print(f"Indexing time: {indexing_time:.2f} ms")

    report_rows = []

    for test in TEST_CASES:
        print(f"\n--- {test['id']}: {test['query']} ---")

        # Strategy 1: Original query only
        original_start = time.perf_counter()

        original_results = retrieve(
            test["query"],
            document_embeddings
        )

        original_latency = (
            time.perf_counter() - original_start
        ) * 1000

        original_evaluation = evaluate_retrieval(
            original_results,
            test["expected"]
        )

        # Strategy 2: Original query plus expanded queries
        expanded_queries = [
            test["query"]
        ] + test["variations"]

        multi_start = time.perf_counter()

        combined_scores = {}

        for expanded_query in expanded_queries:
            results = retrieve(
                expanded_query,
                document_embeddings
            )

            # Reciprocal Rank Fusion across query variations.
            for rank, (doc_id, score) in enumerate(results, start=1):
                combined_scores[doc_id] = (
                    combined_scores.get(doc_id, 0.0)
                    + 1 / (60 + rank)
                )

        multi_results = sorted(
            combined_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )[:TOP_K]

        multi_latency = (
            time.perf_counter() - multi_start
        ) * 1000

        multi_evaluation = evaluate_retrieval(
            multi_results,
            test["expected"]
        )

        original_ids = [
            doc_id for doc_id, score in original_results
        ]

        multi_ids = [
            doc_id for doc_id, score in multi_results
        ]

        print(
            f"Original: {original_evaluation['status']} "
            f"| Retrieved: {original_ids}"
        )

        print(
            f"Multi-query: {multi_evaluation['status']} "
            f"| Retrieved: {multi_ids}"
        )

        for strategy, retrieved_ids, evaluation, latency in [
            (
                "Original Query",
                original_ids,
                original_evaluation,
                original_latency
            ),
            (
                "Multi-Query RRF",
                multi_ids,
                multi_evaluation,
                multi_latency
            )
        ]:
            report_rows.append({
                "Test ID": test["id"],
                "Original Query": test["query"],
                "Strategy": strategy,
                "Number Of Query Variations": (
                    1 if strategy == "Original Query"
                    else len(expanded_queries)
                ),
                "Expected Document": (
                    test["expected"] or "UNSUPPORTED"
                ),
                "Retrieved Documents": ", ".join(retrieved_ids),
                "Status": evaluation["status"],
                "Hit": evaluation["hit"],
                "Precision At K": round(
                    evaluation["precision"], 4
                ),
                "Reciprocal Rank": round(
                    evaluation["reciprocal_rank"], 4
                ),
                "Top 1 Correct": evaluation["top1_correct"],
                "Retrieval Latency Ms": round(latency, 2)
            })

    report_file = "rag_multi_query_report.csv"
    summary_file = "rag_multi_query_summary.csv"

    with open(
        report_file,
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

    summary_rows = []

    for strategy in ["Original Query", "Multi-Query RRF"]:
        strategy_rows = [
            row for row in report_rows
            if row["Strategy"] == strategy
        ]

        supported_rows = [
            row for row in strategy_rows
            if row["Expected Document"] != "UNSUPPORTED"
        ]

        unsupported_rows = [
            row for row in strategy_rows
            if row["Expected Document"] == "UNSUPPORTED"
        ]

        passed = sum(
            row["Status"] == "PASS"
            for row in supported_rows
        )

        failed = len(supported_rows) - passed

        unsupported_rejected = sum(
            row["Status"] == "REJECTED"
            for row in unsupported_rows
        )

        summary_rows.append({
            "Strategy": strategy,
            "Total Test Cases": len(strategy_rows),
            "Supported Queries": len(supported_rows),
            "Unsupported Queries": len(unsupported_rows),
            "Passed": passed,
            "Failed": failed,
            "Pass Rate": round(
                passed / len(supported_rows), 4
            ),
            "Precision At K": round(
                sum(row["Precision At K"] for row in supported_rows)
                / len(supported_rows), 4
            ),
            "MRR": round(
                sum(row["Reciprocal Rank"] for row in supported_rows)
                / len(supported_rows), 4
            ),
            "Top 1 Accuracy": round(
                sum(row["Top 1 Correct"] for row in supported_rows)
                / len(supported_rows), 4
            ),
            "Unsupported Rejection Rate": round(
                unsupported_rejected / len(unsupported_rows), 4
            ),
            "Average Retrieval Latency Ms": round(
                sum(row["Retrieval Latency Ms"] for row in strategy_rows)
                / len(strategy_rows), 2
            )
        })

    with open(
        summary_file,
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

    print("\n===== MULTI-QUERY EVALUATION SUMMARY =====")

    for row in summary_rows:
        print(f"\nStrategy: {row['Strategy']}")
        print(f"Pass Rate: {row['Pass Rate'] * 100:.2f}%")
        print(f"Precision@{TOP_K}: {row['Precision At K']}")
        print(f"MRR: {row['MRR']}")
        print(f"Top-1 Accuracy: {row['Top 1 Accuracy']}")
        print(
            "Unsupported Rejection Rate: "
            f"{row['Unsupported Rejection Rate'] * 100:.2f}%"
        )
        print(
            "Average Retrieval Latency: "
            f"{row['Average Retrieval Latency Ms']} ms"
        )

    print("\n===== TESTING COMPLETED =====")
    print(f"Detailed report: {report_file}")
    print(f"Summary report: {summary_file}")
    print(f"Execution date: {datetime.now()}")


if __name__ == "__main__":
    main()
