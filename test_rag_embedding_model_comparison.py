
import csv
import time
from datetime import datetime

import ollama


# ============================================================
# LESSON 66: RAG EMBEDDING MODEL COMPARISON TESTING
# ============================================================

MODELS = [
    "nomic-embed-text:latest",
    "all-minilm:latest",
]

TOP_K = 3
SUPPORTED_THRESHOLD = 0.68

DOCUMENTS = {
    "DOC001": "Annual leave policy: Employees receive 15 days of annual leave per year.",
    "DOC002": "Sick leave policy: Employees can take 10 days of paid sick leave per year.",
    "DOC003": "Remote work policy: Employees may work remotely up to 3 days per week with manager approval.",
    "DOC004": "Parental leave policy: Eligible employees receive 26 weeks of maternity leave.",
    "DOC005": "Public holiday policy: Employees receive paid leave on officially declared public holidays.",
    "DOC006": "IT security policy: Employees must use strong passwords and enable multi-factor authentication.",
    "DOC007": "Travel reimbursement policy: Employees can claim approved business travel expenses with receipts.",
    "DOC008": "Performance review policy: Employee performance is reviewed twice every year by managers.",
}

TEST_CASES = [
    {
        "id": "TC01",
        "query": "How many annual leave days do employees get?",
        "expected": "DOC001",
        "supported": True,
    },
    {
        "id": "TC02",
        "query": "What is the paid sick leave allowance?",
        "expected": "DOC002",
        "supported": True,
    },
    {
        "id": "TC03",
        "query": "Can staff work from home?",
        "expected": "DOC003",
        "supported": True,
    },
    {
        "id": "TC04",
        "query": "How much maternity leave is provided?",
        "expected": "DOC004",
        "supported": True,
    },
    {
        "id": "TC05",
        "query": "Which public holidays are paid?",
        "expected": "DOC005",
        "supported": True,
    },
    {
        "id": "TC06",
        "query": "What are the password and MFA requirements?",
        "expected": "DOC006",
        "supported": True,
    },
    {
        "id": "TC07",
        "query": "How can I claim business trip expenses?",
        "expected": "DOC007",
        "supported": True,
    },
    {
        "id": "TC08",
        "query": "How often are employee performance reviews conducted?",
        "expected": "DOC008",
        "supported": True,
    },
    {
        "id": "TC09",
        "query": "How many vacation days can staff take?",
        "expected": "DOC001",
        "supported": True,
    },
    {
        "id": "TC10",
        "query": "What is the employee insurance coverage?",
        "expected": "",
        "supported": False,
    },
    {
        "id": "TC11",
        "query": "What is the annual employee bonus amount?",
        "expected": "",
        "supported": False,
    },
    {
        "id": "TC12",
        "query": "What is the current price of gold?",
        "expected": "",
        "supported": False,
    },
]


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = sum(a * a for a in vector_a) ** 0.5
    norm_b = sum(b * b for b in vector_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def get_embedding(model, text):
    start = time.perf_counter()

    response = ollama.embeddings(
        model=model,
        prompt=text,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return response["embedding"], elapsed_ms


def build_index(model):
    print(f"\nBuilding index for {model}...")

    index = {}
    total_embedding_time = 0.0

    for doc_id, content in DOCUMENTS.items():
        vector, latency = get_embedding(model, content)

        index[doc_id] = {
            "content": content,
            "vector": vector,
        }

        total_embedding_time += latency

    print(f"Indexed documents: {len(index)}")
    print(f"Embedding dimensions: {len(next(iter(index.values()))['vector'])}")
    print(f"Index embedding time: {total_embedding_time:.2f} ms")

    return index, total_embedding_time


def retrieve_documents(query_vector, index):
    results = []

    for doc_id, document in index.items():
        score = cosine_similarity(
            query_vector,
            document["vector"],
        )

        results.append({
            "document_id": doc_id,
            "similarity": score,
        })

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return results[:TOP_K]


def calculate_metrics(rows):
    supported_rows = [
        row for row in rows if row["Supported"] == "Yes"
    ]

    unsupported_rows = [
        row for row in rows if row["Supported"] == "No"
    ]

    passed = sum(
        row["Status"] == "PASS" for row in supported_rows
    )

    reciprocal_ranks = []

    for row in supported_rows:
        rank = int(row["Expected Rank"])

        reciprocal_ranks.append(
            1 / rank if rank > 0 else 0
        )

    precision_values = [
        float(row["Precision@K"])
        for row in supported_rows
    ]

    recall_values = [
        float(row["Recall@K"])
        for row in supported_rows
    ]

    top1_correct = sum(
        row["Top1 Correct"] == "Yes"
        for row in supported_rows
    )

    rejected = sum(
        row["Status"] == "REJECTED"
        for row in unsupported_rows
    )

    review = sum(
        row["Status"] == "REVIEW"
        for row in unsupported_rows
    )

    total = len(rows)

    return {
        "Total Test Cases": total,
        "Supported Cases": len(supported_rows),
        "Unsupported Cases": len(unsupported_rows),
        "Passed": passed,
        "Failed": len(supported_rows) - passed,
        "Pass Rate": round(
            passed / len(supported_rows) * 100, 2
        ) if supported_rows else 0,
        "Precision At K": round(
            sum(precision_values) / len(precision_values), 4
        ) if precision_values else 0,
        "Recall At K": round(
            sum(recall_values) / len(recall_values), 4
        ) if recall_values else 0,
        "MRR": round(
            sum(reciprocal_ranks) / len(reciprocal_ranks), 4
        ) if reciprocal_ranks else 0,
        "Top 1 Accuracy": round(
            top1_correct / len(supported_rows) * 100, 2
        ) if supported_rows else 0,
        "Unsupported Rejection Rate": round(
            rejected / len(unsupported_rows) * 100, 2
        ) if unsupported_rows else 0,
        "Unsupported Review Rate": round(
            review / len(unsupported_rows) * 100, 2
        ) if unsupported_rows else 0,
        "Average Query Embedding Latency Ms": round(
            sum(float(row["Query Embedding Latency Ms"]) for row in rows)
            / total, 2
        ) if total else 0,
    }


def evaluate_model(model):
    print("\n" + "=" * 65)
    print(f"EVALUATING MODEL: {model}")
    print("=" * 65)

    index, index_time = build_index(model)

    rows = []

    for test in TEST_CASES:
        query_vector, query_latency = get_embedding(
            model,
            test["query"],
        )

        results = retrieve_documents(query_vector, index)

        retrieved_ids = [
            item["document_id"] for item in results
        ]

        top_scores = [
            item["similarity"] for item in results
        ]

        best_score = top_scores[0] if top_scores else 0.0

        expected = test["expected"]
        supported = test["supported"]

        expected_rank = (
            retrieved_ids.index(expected) + 1
            if expected in retrieved_ids else 0
        )

        if supported:
            status = "PASS" if expected_rank > 0 else "FAIL"

            precision = (
                1 / TOP_K if expected_rank > 0 else 0
            )

            recall = 1.0 if expected_rank > 0 else 0.0

            top1_correct = (
                "Yes" if retrieved_ids and retrieved_ids[0] == expected
                else "No"
            )

        else:
            # Similarity threshold is a simple experimental safeguard.
            # REVIEW means the model retrieved something but did not
            # provide enough evidence to safely reject the query.
            if best_score < SUPPORTED_THRESHOLD:
                status = "REJECTED"
            else:
                status = "REVIEW"

            precision = 0.0
            recall = 0.0
            top1_correct = "N/A"

        row = {
            "Model": model,
            "Test ID": test["id"],
            "Query": test["query"],
            "Expected Document": expected or "N/A",
            "Retrieved Documents": ", ".join(retrieved_ids),
            "Top Similarity": round(best_score, 4),
            "Expected Rank": expected_rank,
            "Top1 Correct": top1_correct,
            "Precision@K": round(precision, 4),
            "Recall@K": round(recall, 4),
            "Query Embedding Latency Ms": round(query_latency, 2),
            "Supported": "Yes" if supported else "No",
            "Status": status,
        }

        rows.append(row)

        print(
            f"{test['id']} | {status:8} | "
            f"Expected: {expected or 'N/A':6} | "
            f"Retrieved: {', '.join(retrieved_ids)}"
        )

    summary = calculate_metrics(rows)
    summary["Model"] = model
    summary["Index Embedding Time Ms"] = round(index_time, 2)
    summary["Top K"] = TOP_K
    summary["Similarity Threshold"] = SUPPORTED_THRESHOLD

    return rows, summary


def write_csv(filename, rows):
    if not rows:
        return

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 65)
    print("RAG EMBEDDING MODEL COMPARISON")
    print("=" * 65)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Documents: {len(DOCUMENTS)}")
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Top K: {TOP_K}")
    print(f"Threshold: {SUPPORTED_THRESHOLD}")

    all_reports = []
    all_summaries = []

    for model in MODELS:
        try:
            report, summary = evaluate_model(model)

            all_reports.extend(report)
            all_summaries.append(summary)

        except Exception as error:
            print(f"\nERROR evaluating {model}: {error}")
            print("Check that the model is available in Ollama.")

    if not all_reports:
        print("\nNo results generated. Check Ollama and model availability.")
        return

    write_csv(
        "rag_embedding_model_comparison_report.csv",
        all_reports,
    )

    write_csv(
        "rag_embedding_model_comparison_summary.csv",
        all_summaries,
    )

    print("\n" + "=" * 65)
    print("EMBEDDING MODEL COMPARISON SUMMARY")
    print("=" * 65)

    for summary in all_summaries:
        print(f"\nModel: {summary['Model']}")
        print(f"Pass Rate: {summary['Pass Rate']}%")
        print(f"Precision@K: {summary['Precision At K']}")
        print(f"Recall@K: {summary['Recall At K']}")
        print(f"MRR: {summary['MRR']}")
        print(f"Top-1 Accuracy: {summary['Top 1 Accuracy']}%")
        print(
            "Unsupported Rejection Rate: "
            f"{summary['Unsupported Rejection Rate']}%"
        )
        print(
            "Average Query Embedding Latency: "
            f"{summary['Average Query Embedding Latency Ms']} ms"
        )
        print(
            f"Index Embedding Time: "
            f"{summary['Index Embedding Time Ms']} ms"
        )

    print("\nReports generated:")
    print("1. rag_embedding_model_comparison_report.csv")
    print("2. rag_embedding_model_comparison_summary.csv")

    print("\nLesson 66 execution completed.")


if __name__ == "__main__":
    main()
