
import csv
import time
import math
from datetime import datetime

import ollama


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "nomic-embed-text:latest"

TOP_K_VALUES = [1, 2, 3, 5]

REPORT_FILE = "rag_search_relevance_report.csv"
SUMMARY_FILE = "rag_search_relevance_summary.csv"


# ============================================================
# HR POLICY DOCUMENT DATASET
# ============================================================

DOCUMENTS = {
    "DOC001": """
    Annual Leave Policy:
    Employees receive 15 annual leave days per year.
    Annual leave requests must be submitted through the
    HR system and approved by the reporting manager.
    """,

    "DOC002": """
    Sick Leave Policy:
    Employees can take sick leave when they are unwell.
    Employees must inform their manager and provide
    medical documentation when required.
    """,

    "DOC003": """
    Remote Work Policy:
    Employees can work remotely with manager approval.
    Remote work arrangements must follow company
    security and attendance requirements.
    """,

    "DOC004": """
    Parental Leave Policy:
    Eligible employees can apply for parental leave.
    Employees should contact HR for eligibility,
    duration and application requirements.
    """,

    "DOC005": """
    Public Holiday Policy:
    The company observes designated public holidays.
    The annual holiday calendar is published by HR.
    Employees should check the calendar for holiday dates.
    """,

    "DOC006": """
    IT Security Policy:
    Employees must use strong passwords and enable
    multi-factor authentication. Company information
    must be protected. Employees must report security
    incidents to the IT department.
    """,

    "DOC007": """
    Travel Reimbursement Policy:
    Employees can claim reimbursement for approved
    business travel expenses. Claims must include
    receipts and manager approval.
    """,

    "DOC008": """
    Performance Review Policy:
    Employee performance reviews are conducted
    periodically by managers. Reviews cover goals,
    performance feedback and development plans.
    """
}


# ============================================================
# SEARCH TEST DATASET
# ============================================================

TEST_CASES = [
    {
        "test_id": "TC01",
        "query": "How many annual leave days do employees receive?",
        "expected_doc": "DOC001"
    },
    {
        "test_id": "TC02",
        "query": "What is the process for taking sick leave?",
        "expected_doc": "DOC002"
    },
    {
        "test_id": "TC03",
        "query": "Can employees work remotely?",
        "expected_doc": "DOC003"
    },
    {
        "test_id": "TC04",
        "query": "What are the parental leave requirements?",
        "expected_doc": "DOC004"
    },
    {
        "test_id": "TC05",
        "query": "Which public holidays does the company observe?",
        "expected_doc": "DOC005"
    },
    {
        "test_id": "TC06",
        "query": "What are the company password and security rules?",
        "expected_doc": "DOC006"
    },
    {
        "test_id": "TC07",
        "query": "How can employees claim business travel expenses?",
        "expected_doc": "DOC007"
    },
    {
        "test_id": "TC08",
        "query": "How are employee performance reviews conducted?",
        "expected_doc": "DOC008"
    }
]


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ============================================================
# EMBEDDING GENERATION
# ============================================================

def generate_embeddings(texts):

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts
    )

    return response["embeddings"]


# ============================================================
# DOCUMENT INDEXING
# ============================================================

def build_document_index():

    print("\n===== DOCUMENT INDEXING =====")

    document_ids = list(DOCUMENTS.keys())
    document_texts = list(DOCUMENTS.values())

    start_time = time.perf_counter()

    embeddings = generate_embeddings(document_texts)

    indexing_time = (
        time.perf_counter() - start_time
    ) * 1000

    document_index = {}

    for doc_id, embedding in zip(document_ids, embeddings):
        document_index[doc_id] = embedding

    print("Total documents:", len(document_index))
    print("Embedding model:", EMBEDDING_MODEL)
    print("Indexing time: {:.2f} ms".format(indexing_time))

    return document_index, indexing_time


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(query, document_index):

    start_time = time.perf_counter()

    query_embedding = generate_embeddings([query])[0]

    query_embedding_time = (
        time.perf_counter() - start_time
    ) * 1000

    ranking_start = time.perf_counter()

    scores = []

    for doc_id, doc_embedding in document_index.items():

        score = cosine_similarity(
            query_embedding,
            doc_embedding
        )

        scores.append({
            "document_id": doc_id,
            "similarity_score": score
        })

    scores.sort(
        key=lambda item: item["similarity_score"],
        reverse=True
    )

    ranking_time = (
        time.perf_counter() - ranking_start
    ) * 1000

    return scores, query_embedding_time, ranking_time


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(ranked_results, expected_doc, top_k):

    retrieved = ranked_results[:top_k]

    retrieved_ids = [
        item["document_id"] for item in retrieved
    ]

    relevant_count = sum(
        1 for doc_id in retrieved_ids
        if doc_id == expected_doc
    )

    precision = relevant_count / top_k

    recall = 1.0 if expected_doc in retrieved_ids else 0.0

    reciprocal_rank = 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):

        if doc_id == expected_doc:
            reciprocal_rank = 1 / rank
            break

    hit = 1 if expected_doc in retrieved_ids else 0

    top1_hit = (
        1 if ranked_results[0]["document_id"] == expected_doc
        else 0
    )

    return {
        "retrieved_ids": retrieved_ids,
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank,
        "hit": hit,
        "top1_hit": top1_hit
    }


# ============================================================
# MAIN TEST EXECUTION
# ============================================================

def run_tests():

    print("\n==============================================")
    print(" RAG SEARCH RELEVANCE AND OPTIMIZATION TESTING")
    print("==============================================")

    execution_date = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    document_index, indexing_time = build_document_index()

    report_rows = []
    summary_rows = []

    for top_k in TOP_K_VALUES:

        print("\n==============================================")
        print("TESTING TOP-K =", top_k)
        print("==============================================")

        total_precision = 0.0
        total_recall = 0.0
        total_mrr = 0.0
        total_hit = 0
        total_top1 = 0

        total_embedding_time = 0.0
        total_ranking_time = 0.0

        passed = 0
        failed = 0

        for test_case in TEST_CASES:

            test_id = test_case["test_id"]
            query = test_case["query"]
            expected_doc = test_case["expected_doc"]

            print("\nTest:", test_id)
            print("Query:", query)

            ranked_results, embedding_time, ranking_time = (
                retrieve_documents(query, document_index)
            )

            metrics = calculate_metrics(
                ranked_results,
                expected_doc,
                top_k
            )

            status = (
                "PASS" if metrics["hit"] == 1
                else "FAIL"
            )

            if status == "PASS":
                passed += 1
            else:
                failed += 1

            total_precision += metrics["precision"]
            total_recall += metrics["recall"]
            total_mrr += metrics["mrr"]
            total_hit += metrics["hit"]
            total_top1 += metrics["top1_hit"]

            total_embedding_time += embedding_time
            total_ranking_time += ranking_time

            retrieved_text = ", ".join(
                metrics["retrieved_ids"]
            )

            print("Expected:", expected_doc)
            print("Retrieved:", retrieved_text)
            print("Precision@K:", round(metrics["precision"], 4))
            print("Recall@K:", round(metrics["recall"], 4))
            print("MRR:", round(metrics["mrr"], 4))
            print("Status:", status)

            report_rows.append({
                "Execution Date": execution_date,
                "Test ID": test_id,
                "Query": query,
                "Top K": top_k,
                "Expected Document": expected_doc,
                "Retrieved Documents": retrieved_text,
                "Precision@K": round(metrics["precision"], 4),
                "Recall@K": round(metrics["recall"], 4),
                "MRR": round(metrics["mrr"], 4),
                "Hit": metrics["hit"],
                "Top1 Hit": metrics["top1_hit"],
                "Embedding Latency (ms)": round(embedding_time, 3),
                "Ranking Latency (ms)": round(ranking_time, 3),
                "Status": status
            })

        total_tests = len(TEST_CASES)

        pass_rate = (passed / total_tests) * 100

        avg_precision = total_precision / total_tests
        avg_recall = total_recall / total_tests
        avg_mrr = total_mrr / total_tests

        hit_rate = (total_hit / total_tests) * 100
        top1_accuracy = (total_top1 / total_tests) * 100

        avg_embedding_time = (
            total_embedding_time / total_tests
        )

        avg_ranking_time = (
            total_ranking_time / total_tests
        )

        print("\n===== TOP-K SUMMARY =====")
        print("Top K:", top_k)
        print("Passed:", passed)
        print("Failed:", failed)
        print("Pass Rate:", round(pass_rate, 2), "%")
        print("Precision@K:", round(avg_precision, 4))
        print("Recall@K:", round(avg_recall, 4))
        print("MRR:", round(avg_mrr, 4))
        print("Hit Rate:", round(hit_rate, 2), "%")
        print("Top-1 Accuracy:", round(top1_accuracy, 2), "%")
        print(
            "Average Query Embedding Latency:",
            round(avg_embedding_time, 3), "ms"
        )
        print(
            "Average Ranking Latency:",
            round(avg_ranking_time, 3), "ms"
        )

        summary_rows.append({
            "Execution Date": execution_date,
            "Top K": top_k,
            "Total Tests": total_tests,
            "Passed": passed,
            "Failed": failed,
            "Pass Rate (%)": round(pass_rate, 2),
            "Precision@K": round(avg_precision, 4),
            "Recall@K": round(avg_recall, 4),
            "MRR": round(avg_mrr, 4),
            "Hit Rate (%)": round(hit_rate, 2),
            "Top1 Accuracy (%)": round(top1_accuracy, 2),
            "Average Embedding Latency (ms)": round(
                avg_embedding_time, 3
            ),
            "Average Ranking Latency (ms)": round(
                avg_ranking_time, 3
            ),
            "Indexing Latency (ms)": round(indexing_time, 3)
        })

    # ========================================================
    # CSV REPORT GENERATION
    # ========================================================

    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=report_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(report_rows)

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=summary_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n==============================================")
    print("FINAL TOP-K COMPARISON")
    print("==============================================")

    print(
        "{:<10} {:<12} {:<12} {:<12} {:<12}".format(
            "Top K",
            "Pass Rate",
            "Precision",
            "Recall",
            "MRR"
        )
    )

    for row in summary_rows:

        print(
            "{:<10} {:<12} {:<12} {:<12} {:<12}".format(
                row["Top K"],
                str(row["Pass Rate (%)"]) + "%",
                row["Precision@K"],
                row["Recall@K"],
                row["MRR"]
            )
        )

    print("\n===== REPORTS GENERATED =====")
    print(REPORT_FILE)
    print(SUMMARY_FILE)

    print("\nTotal detailed report rows:", len(report_rows))
    print("Total summary rows:", len(summary_rows))

    print("\nRAG SEARCH RELEVANCE TESTING COMPLETED")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        run_tests()

    except Exception as error:

        print("\nERROR DURING TEST EXECUTION")
        print(str(error))
        raise
