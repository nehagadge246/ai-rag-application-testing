
import csv
import time
import requests
from datetime import datetime

# ============================================================
# LESSON 54: RAG CONTEXT PRECISION AND RECALL TESTING
# ============================================================

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"

TOP_K = 3
SIMILARITY_THRESHOLD = 0.55

REPORT_FILE = "rag_context_precision_recall_report.csv"
SUMMARY_FILE = "rag_context_precision_recall_summary.csv"

# ------------------------------------------------------------
# STEP 1: KNOWLEDGE BASE
# ------------------------------------------------------------

documents = [
    {
        "id": "DOC001",
        "topic": "annual leave",
        "content": "Employees receive 15 annual leave days every year."
    },
    {
        "id": "DOC002",
        "topic": "sick leave",
        "content": "Employees receive 10 paid sick leave days every year."
    },
    {
        "id": "DOC003",
        "topic": "remote work",
        "content": "Employees can work remotely two days per week."
    },
    {
        "id": "DOC004",
        "topic": "parental leave",
        "content": "Eligible employees receive 12 weeks of parental leave."
    },
    {
        "id": "DOC005",
        "topic": "public holidays",
        "content": "The company observes 10 public holidays each year."
    },
    {
        "id": "DOC006",
        "topic": "IT security",
        "content": "Employees must use multi-factor authentication to access company systems."
    },
    {
        "id": "DOC007",
        "topic": "travel reimbursement",
        "content": "Employees can claim approved business travel expenses with valid receipts."
    },
    {
        "id": "DOC008",
        "topic": "performance review",
        "content": "Employee performance reviews are conducted twice a year."
    }
]

# ------------------------------------------------------------
# STEP 2: TEST DATASET
# Relevant document IDs are the ground-truth expected results.
# ------------------------------------------------------------

test_cases = [
    {
        "scenario": "Annual Leave",
        "question": "How many annual leave days do employees receive?",
        "relevant_ids": ["DOC001"]
    },
    {
        "scenario": "Sick Leave",
        "question": "How many paid sick days are available?",
        "relevant_ids": ["DOC002"]
    },
    {
        "scenario": "Remote Work",
        "question": "How many days can employees work from home?",
        "relevant_ids": ["DOC003"]
    },
    {
        "scenario": "Parental Leave",
        "question": "How long is parental leave?",
        "relevant_ids": ["DOC004"]
    },
    {
        "scenario": "Public Holidays",
        "question": "How many company holidays are observed?",
        "relevant_ids": ["DOC005"]
    },
    {
        "scenario": "IT Security",
        "question": "What security method is required to access company systems?",
        "relevant_ids": ["DOC006"]
    },
    {
        "scenario": "Travel Reimbursement",
        "question": "How can employees claim business travel expenses?",
        "relevant_ids": ["DOC007"]
    },
    {
        "scenario": "Performance Review",
        "question": "How frequently are employee performance reviews conducted?",
        "relevant_ids": ["DOC008"]
    },
    {
        "scenario": "Unsupported Insurance",
        "question": "Which health insurance company does the organization use?",
        "relevant_ids": []
    },
    {
        "scenario": "Unrelated Stock Price",
        "question": "What is the current price of a technology company stock?",
        "relevant_ids": []
    }
]

# ------------------------------------------------------------
# STEP 3: GENERATE EMBEDDINGS
# ------------------------------------------------------------

def get_embedding(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=60
    )

    response.raise_for_status()

    data = response.json()
    embeddings = data.get("embeddings", [])

    if not embeddings:
        raise ValueError("Embedding response did not contain embeddings.")

    return embeddings[0]


# ------------------------------------------------------------
# STEP 4: COSINE SIMILARITY
# ------------------------------------------------------------

def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ------------------------------------------------------------
# STEP 5: PRECISION@K
# Relevant retrieved documents / Total retrieved documents
# ------------------------------------------------------------

def calculate_precision_at_k(retrieved_ids, relevant_ids):
    if not retrieved_ids:
        return 0.0

    relevant_count = sum(
        1 for doc_id in retrieved_ids
        if doc_id in relevant_ids
    )

    return relevant_count / len(retrieved_ids)


# ------------------------------------------------------------
# STEP 6: RECALL@K
# Relevant retrieved documents / Total expected relevant docs
# ------------------------------------------------------------

def calculate_recall_at_k(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return None

    relevant_count = sum(
        1 for doc_id in retrieved_ids
        if doc_id in relevant_ids
    )

    return relevant_count / len(relevant_ids)


# ------------------------------------------------------------
# STEP 7: MEAN RECIPROCAL RANK
# Reciprocal rank of first relevant retrieved document.
# ------------------------------------------------------------

def calculate_reciprocal_rank(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return None

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1 / rank

    return 0.0


# ------------------------------------------------------------
# STEP 8: BUILD DOCUMENT EMBEDDINGS
# ------------------------------------------------------------

print("\n===== RAG CONTEXT PRECISION AND RECALL TESTING =====")

print("\n===== STEP 1: CHECK OLLAMA CONNECTION =====")

try:
    health_response = requests.get(
        f"{OLLAMA_URL}/api/tags",
        timeout=10
    )
    health_response.raise_for_status()

    print("Ollama connection: PASS")

except requests.RequestException as error:
    print("Ollama connection: FAIL")
    print(f"Error: {error}")
    raise SystemExit(1)


print("\n===== STEP 2: CREATE DOCUMENT EMBEDDINGS =====")

document_embeddings = []
embedding_start = time.perf_counter()

try:
    for document in documents:
        vector = get_embedding(document["content"])

        document_embeddings.append({
            "id": document["id"],
            "topic": document["topic"],
            "content": document["content"],
            "embedding": vector
        })

        print(f"Embedded: {document['id']} - {document['topic']}")

except Exception as error:
    print(f"Document embedding failed: {error}")
    raise SystemExit(1)

embedding_time = time.perf_counter() - embedding_start

print(f"\nDocuments indexed: {len(document_embeddings)}")
print(f"Embedding time: {embedding_time:.3f} seconds")
print(f"Embedding dimensions: {len(document_embeddings[0]['embedding'])}")


# ------------------------------------------------------------
# STEP 9: RETRIEVAL AND METRIC EVALUATION
# ------------------------------------------------------------

print("\n===== STEP 3: RETRIEVAL QUALITY EVALUATION =====")

results = []

for index, test in enumerate(test_cases, start=1):

    print(f"\n===== TEST CASE {index}: {test['scenario']} =====")
    print(f"Question: {test['question']}")

    start_time = time.perf_counter()

    try:
        query_embedding = get_embedding(test["question"])

        scored_documents = []

        for document in document_embeddings:
            score = cosine_similarity(
                query_embedding,
                document["embedding"]
            )

            scored_documents.append({
                "id": document["id"],
                "topic": document["topic"],
                "score": score
            })

        # Sort by descending similarity.
        scored_documents.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        # Retrieve top K documents.
        top_documents = scored_documents[:TOP_K]

        # Apply similarity threshold.
        retrieved_documents = [
            document for document in top_documents
            if document["score"] >= SIMILARITY_THRESHOLD
        ]

        retrieved_ids = [
            document["id"] for document in retrieved_documents
        ]

        relevant_ids = test["relevant_ids"]

        precision = calculate_precision_at_k(
            retrieved_ids,
            relevant_ids
        )

        recall = calculate_recall_at_k(
            retrieved_ids,
            relevant_ids
        )

        reciprocal_rank = calculate_reciprocal_rank(
            retrieved_ids,
            relevant_ids
        )

        latency = time.perf_counter() - start_time

        # Supported query: expected relevant document must be retrieved.
        # Unsupported query: no document should pass the threshold.
        if relevant_ids:
            status = (
                "PASS"
                if any(doc_id in relevant_ids for doc_id in retrieved_ids)
                else "FAIL"
            )
        else:
            status = (
                "PASS"
                if len(retrieved_ids) == 0
                else "REVIEW"
            )

        retrieved_topics = [
            document["topic"] for document in retrieved_documents
        ]

        retrieved_scores = [
            round(document["score"], 4)
            for document in retrieved_documents
        ]

        result = {
            "Scenario": test["scenario"],
            "Question": test["question"],
            "Expected Relevant IDs": ", ".join(relevant_ids) if relevant_ids else "None",
            "Retrieved IDs": ", ".join(retrieved_ids) if retrieved_ids else "None",
            "Retrieved Topics": ", ".join(retrieved_topics) if retrieved_topics else "None",
            "Retrieved Scores": ", ".join(map(str, retrieved_scores)) if retrieved_scores else "None",
            "Precision@K": round(precision, 4),
            "Recall@K": round(recall, 4) if recall is not None else "N/A",
            "Reciprocal Rank": round(reciprocal_rank, 4) if reciprocal_rank is not None else "N/A",
            "Retrieval Latency (seconds)": round(latency, 3),
            "Status": status
        }

        results.append(result)

        print(f"Retrieved IDs: {retrieved_ids}")
        print(f"Precision@{TOP_K}: {precision:.4f}")

        if recall is not None:
            print(f"Recall@{TOP_K}: {recall:.4f}")
        else:
            print(f"Recall@{TOP_K}: N/A (unsupported question)")

        if reciprocal_rank is not None:
            print(f"Reciprocal Rank: {reciprocal_rank:.4f}")
        else:
            print("Reciprocal Rank: N/A (unsupported question)")

        print(f"Latency: {latency:.3f} seconds")
        print(f"Status: {status}")

    except Exception as error:
        latency = time.perf_counter() - start_time

        print(f"Test execution error: {error}")

        results.append({
            "Scenario": test["scenario"],
            "Question": test["question"],
            "Expected Relevant IDs": ", ".join(test["relevant_ids"]) or "None",
            "Retrieved IDs": "Error",
            "Retrieved Topics": "Error",
            "Retrieved Scores": "Error",
            "Precision@K": "N/A",
            "Recall@K": "N/A",
            "Reciprocal Rank": "N/A",
            "Retrieval Latency (seconds)": round(latency, 3),
            "Status": "FAIL"
        })


# ------------------------------------------------------------
# STEP 10: GENERATE CSV REPORT
# ------------------------------------------------------------

print("\n===== STEP 4: GENERATE DETAILED REPORT =====")

report_fields = [
    "Scenario",
    "Question",
    "Expected Relevant IDs",
    "Retrieved IDs",
    "Retrieved Topics",
    "Retrieved Scores",
    "Precision@K",
    "Recall@K",
    "Reciprocal Rank",
    "Retrieval Latency (seconds)",
    "Status"
]

with open(REPORT_FILE, "w", newline="", encoding="utf-8-sig") as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=report_fields
    )

    writer.writeheader()
    writer.writerows(results)

print(f"Detailed report generated: {REPORT_FILE}")


# ------------------------------------------------------------
# STEP 11: SUMMARY METRICS
# ------------------------------------------------------------

print("\n===== STEP 5: GENERATE SUMMARY =====")

total_tests = len(results)

passed = sum(1 for result in results if result["Status"] == "PASS")
failed = sum(1 for result in results if result["Status"] == "FAIL")
review = sum(1 for result in results if result["Status"] == "REVIEW")

supported_results = [
    result for result, test in zip(results, test_cases)
    if test["relevant_ids"] and result["Status"] != "FAIL"
]

def numeric_values(field):
    return [
        float(result[field])
        for result in supported_results
        if isinstance(result[field], (int, float))
    ]


precision_values = numeric_values("Precision@K")
recall_values = numeric_values("Recall@K")
rr_values = numeric_values("Reciprocal Rank")
latency_values = [
    float(result["Retrieval Latency (seconds)"])
    for result in results
    if isinstance(result["Retrieval Latency (seconds)"], (int, float))
]

average_precision = (
    sum(precision_values) / len(precision_values)
    if precision_values else 0
)

average_recall = (
    sum(recall_values) / len(recall_values)
    if recall_values else 0
)

mrr = (
    sum(rr_values) / len(rr_values)
    if rr_values else 0
)

average_latency = (
    sum(latency_values) / len(latency_values)
    if latency_values else 0
)

summary = {
    "Metric": [
        "Execution Date",
        "Total Test Cases",
        "Passed",
        "Failed",
        "Review",
        "Pass Rate (%)",
        "Average Precision@K (supported queries)",
        "Average Recall@K (supported queries)",
        "Mean Reciprocal Rank (supported queries)",
        "Average Retrieval Latency (seconds)",
        "Top K",
        "Similarity Threshold",
        "Documents Indexed",
        "Embedding Dimensions"
    ],
    "Value": [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_tests,
        passed,
        failed,
        review,
        round((passed / total_tests) * 100, 2) if total_tests else 0,
        round(average_precision, 4),
        round(average_recall, 4),
        round(mrr, 4),
        round(average_latency, 3),
        TOP_K,
        SIMILARITY_THRESHOLD,
        len(document_embeddings),
        len(document_embeddings[0]["embedding"])
    ]
}

with open(SUMMARY_FILE, "w", newline="", encoding="utf-8-sig") as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow(["Metric", "Value"])

    for metric, value in zip(summary["Metric"], summary["Value"]):
        writer.writerow([metric, value])

print(f"Summary report generated: {SUMMARY_FILE}")


# ------------------------------------------------------------
# STEP 12: FINAL RESULTS
# ------------------------------------------------------------

print("\n========== FINAL TEST SUMMARY ==========")

print(f"Total Test Cases: {total_tests}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Review: {review}")

print(f"Pass Rate: {(passed / total_tests) * 100:.2f}%")
print(f"Average Precision@{TOP_K}: {average_precision:.4f}")
print(f"Average Recall@{TOP_K}: {average_recall:.4f}")
print(f"Mean Reciprocal Rank: {mrr:.4f}")
print(f"Average Retrieval Latency: {average_latency:.3f} seconds")

print("\nGenerated Files:")
print(f"1. {REPORT_FILE}")
print(f"2. {SUMMARY_FILE}")

print("\n===== RAG CONTEXT PRECISION AND RECALL TESTING COMPLETED =====")
