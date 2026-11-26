
import csv
import re
import time
from datetime import datetime

import ollama
import numpy as np

# ============================================================
# LESSON 65: RAG DOCUMENT CHUNKING OPTIMIZATION
# ============================================================

MODEL = "nomic-embed-text:latest"
TOP_K = 3
THRESHOLD = 0.68

CHUNK_CONFIGURATIONS = [
    {"chunk_size": 100, "overlap": 0},
    {"chunk_size": 100, "overlap": 20},
    {"chunk_size": 200, "overlap": 0},
    {"chunk_size": 200, "overlap": 40},
    {"chunk_size": 300, "overlap": 0},
    {"chunk_size": 300, "overlap": 60},
    {"chunk_size": 500, "overlap": 0},
    {"chunk_size": 500, "overlap": 100},
]

DOCUMENTS = {
    "DOC001": (
        "Annual Leave Policy. Employees receive 15 annual leave "
        "days per year. Leave requests must be submitted to the "
        "manager for approval. Unused annual leave is handled "
        "according to company policy."
    ),
    "DOC002": (
        "Sick Leave Policy. Employees can take up to 10 sick "
        "leave days annually. Employees should inform their "
        "manager when they are unable to attend work due to illness."
    ),
    "DOC003": (
        "Remote Work Policy. Employees may work remotely up to "
        "three days per week. Remote workers must remain available "
        "during working hours and attend scheduled team meetings."
    ),
    "DOC004": (
        "Parental Leave Policy. Eligible employees receive "
        "26 weeks of parental leave. Employees should notify "
        "HR before the planned leave start date."
    ),
    "DOC005": (
        "Public Holiday Policy. The company observes 12 public "
        "holidays each year. The holiday calendar is published "
        "by HR at the beginning of each year."
    ),
    "DOC006": (
        "IT Security Policy. Employees must use strong passwords "
        "and enable multi-factor authentication. Passwords must "
        "not be shared with other employees."
    ),
    "DOC007": (
        "Travel Reimbursement Policy. Business travel expenses "
        "must be submitted with receipts for reimbursement. "
        "Claims should be submitted within 30 days of travel."
    ),
    "DOC008": (
        "Performance Review Policy. Employee performance reviews "
        "are conducted twice a year. Managers discuss goals, "
        "achievements and development opportunities."
    ),
}

TEST_CASES = [
    {"id": "TC01", "query": "How many annual leave days do employees receive?", "expected": "DOC001"},
    {"id": "TC02", "query": "How many sick leave days are allowed?", "expected": "DOC002"},
    {"id": "TC03", "query": "How many days can employees work remotely?", "expected": "DOC003"},
    {"id": "TC04", "query": "How long is parental leave?", "expected": "DOC004"},
    {"id": "TC05", "query": "How many public holidays are observed?", "expected": "DOC005"},
    {"id": "TC06", "query": "What are the password security requirements?", "expected": "DOC006"},
    {"id": "TC07", "query": "How are business travel expenses reimbursed?", "expected": "DOC007"},
    {"id": "TC08", "query": "How often are performance reviews conducted?", "expected": "DOC008"},
    {"id": "TC09", "query": "What is the employee bonus policy?", "expected": None},
    {"id": "TC10", "query": "What health insurance coverage is provided?", "expected": None},
    {"id": "TC11", "query": "What is the current gold price?", "expected": None},
    {"id": "TC12", "query": "Where can employees park their vehicles?", "expected": None},
]


def split_into_chunks(text, chunk_size, overlap):
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def get_embedding(text):
    response = ollama.embeddings(
        model=MODEL,
        prompt=text
    )
    return np.array(response["embedding"], dtype=float)


def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def retrieve(query_embedding, chunk_embeddings, top_k):
    results = []

    for chunk in chunk_embeddings:
        score = cosine_similarity(
            query_embedding,
            chunk["embedding"]
        )

        results.append({
            "document_id": chunk["document_id"],
            "chunk_number": chunk["chunk_number"],
            "score": score,
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]


def evaluate(results, expected_document):
    retrieved_documents = [
        result["document_id"] for result in results
    ]

    unique_documents = list(dict.fromkeys(retrieved_documents))

    if expected_document is None:
        return {
            "status": "REVIEW" if results else "REJECTED",
            "hit": 0,
            "precision": 0.0,
            "reciprocal_rank": 0.0,
            "top1": 0,
        }

    hit = int(expected_document in unique_documents)

    precision = (
        hit / len(unique_documents)
        if unique_documents else 0.0
    )

    reciprocal_rank = 0.0

    if expected_document in unique_documents:
        reciprocal_rank = (
            1 / (unique_documents.index(expected_document) + 1)
        )

    return {
        "status": "PASS" if hit else "FAIL",
        "hit": hit,
        "precision": precision,
        "reciprocal_rank": reciprocal_rank,
        "top1": int(
            bool(unique_documents)
            and unique_documents[0] == expected_document
        ),
    }


def main():
    print("\n===== RAG CHUNKING OPTIMIZATION TESTING =====")
    print(f"Embedding model: {MODEL}")
    print(f"Top K: {TOP_K}")

    report_rows = []
    summary_rows = []

    for config in CHUNK_CONFIGURATIONS:
        chunk_size = config["chunk_size"]
        overlap = config["overlap"]

        print(
            f"\nTesting chunk size={chunk_size}, "
            f"overlap={overlap}"
        )

        chunk_data = []

        chunking_start = time.perf_counter()

        for document_id, document_text in DOCUMENTS.items():
            chunks = split_into_chunks(
                document_text,
                chunk_size,
                overlap
            )

            for index, chunk_text in enumerate(chunks, start=1):
                chunk_data.append({
                    "document_id": document_id,
                    "chunk_number": index,
                    "text": chunk_text,
                })

        chunking_latency = (
            time.perf_counter() - chunking_start
        ) * 1000

        embedding_start = time.perf_counter()

        chunk_embeddings = []

        for chunk in chunk_data:
            chunk_embeddings.append({
                **chunk,
                "embedding": get_embedding(chunk["text"]),
            })

        embedding_latency = (
            time.perf_counter() - embedding_start
        ) * 1000

        supported_rows = []
        unsupported_rows = []
        retrieval_latencies = []

        for test in TEST_CASES:
            query_start = time.perf_counter()

            query_embedding = get_embedding(test["query"])

            results = retrieve(
                query_embedding,
                chunk_embeddings,
                TOP_K
            )

            retrieval_latency = (
                time.perf_counter() - query_start
            ) * 1000

            retrieval_latencies.append(retrieval_latency)

            evaluation = evaluate(
                results,
                test["expected"]
            )

            retrieved_documents = list(dict.fromkeys(
                result["document_id"] for result in results
            ))

            row = {
                "Test ID": test["id"],
                "Query": test["query"],
                "Chunk Size": chunk_size,
                "Overlap": overlap,
                "Expected Document": (
                    test["expected"] or "UNSUPPORTED"
                ),
                "Retrieved Documents": ", ".join(
                    retrieved_documents
                ),
                "Top Chunk Similarity": round(
                    results[0]["score"], 4
                ) if results else 0.0,
                "Status": evaluation["status"],
                "Hit": evaluation["hit"],
                "Precision At K": round(
                    evaluation["precision"], 4
                ),
                "Reciprocal Rank": round(
                    evaluation["reciprocal_rank"], 4
                ),
                "Top 1 Correct": evaluation["top1"],
                "Retrieval Latency Ms": round(
                    retrieval_latency, 2
                ),
            }

            report_rows.append(row)

            if test["expected"] is None:
                unsupported_rows.append(row)
            else:
                supported_rows.append(row)

        passed = sum(
            row["Status"] == "PASS"
            for row in supported_rows
        )

        rejected = sum(
            row["Status"] == "REJECTED"
            for row in unsupported_rows
        )

        summary_rows.append({
            "Chunk Size": chunk_size,
            "Overlap": overlap,
            "Total Chunks": len(chunk_data),
            "Total Test Cases": len(TEST_CASES),
            "Supported Queries": len(supported_rows),
            "Unsupported Queries": len(unsupported_rows),
            "Passed": passed,
            "Failed": len(supported_rows) - passed,
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
                rejected / len(unsupported_rows), 4
            ),
            "Chunking Latency Ms": round(chunking_latency, 2),
            "Embedding Latency Ms": round(embedding_latency, 2),
            "Average Retrieval Latency Ms": round(
                sum(retrieval_latencies) / len(retrieval_latencies),
                2
            ),
        })

        print(f"Total chunks: {len(chunk_data)}")
        print(f"Supported queries passed: {passed}/8")

    report_file = "rag_chunking_optimization_report.csv"
    summary_file = "rag_chunking_optimization_summary.csv"

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

    print("\n===== CHUNKING OPTIMIZATION SUMMARY =====")

    for row in summary_rows:
        print(
            f"Size={row['Chunk Size']}, "
            f"Overlap={row['Overlap']}, "
            f"Chunks={row['Total Chunks']}, "
            f"Pass Rate={row['Pass Rate'] * 100:.2f}%, "
            f"MRR={row['MRR']}, "
            f"Avg Retrieval={row['Average Retrieval Latency Ms']} ms"
        )

    print("\n===== TESTING COMPLETED =====")
    print(f"Detailed report: {report_file}")
    print(f"Summary report: {summary_file}")
    print(f"Execution date: {datetime.now()}")


if __name__ == "__main__":
    main()
