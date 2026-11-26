
import csv
import math
import time
import requests
from datetime import datetime

# ============================================================
# LESSON 55: RAG RETRIEVAL RANKING EVALUATION
# Metrics: DCG@K, NDCG@K, Hit Rate@K
# ============================================================

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"

TOP_K = 3
SIMILARITY_THRESHOLD = 0.55

REPORT_FILE = "rag_ranking_evaluation_report.csv"
SUMMARY_FILE = "rag_ranking_evaluation_summary.csv"

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
# STEP 2: TEST DATASET WITH GRADED RELEVANCE
#
# 3 = Highly relevant
# 2 = Relevant
# 1 = Partially relevant
# 0 = Not relevant
#
# These relevance labels are manually defined ground truth.
# ------------------------------------------------------------

test_cases = [
    {
        "scenario": "Annual Leave",
        "question": "How many annual leave days do employees receive?",
        "relevance": {
            "DOC001": 3,
            "DOC002": 1,
            "DOC004": 0,
            "DOC005": 0,
            "DOC003": 0,
            "DOC006": 0,
            "DOC007": 0,
            "DOC008": 0
        }
    },
    {
        "scenario": "Sick Leave",
        "question": "How many paid sick days are available?",
        "relevance": {
            "DOC002": 3,
            "DOC001": 1,
            "DOC004": 0,
            "DOC005": 0,
            "DOC003": 0,
            "DOC006": 0,
            "DOC007": 0,
            "DOC008": 0
        }
    },
    {
        "scenario": "Remote Work",
        "question": "How many days can employees work from home?",
        "relevance": {
            "DOC003": 3,
            "DOC006": 0,
            "DOC001": 0,
            "DOC002": 0,
            "DOC004": 0,
            "DOC005": 0,
            "DOC007": 0,
            "DOC008": 0
        }
    },
    {
        "scenario": "Parental Leave",
        "question": "How long is parental leave?",
        "relevance": {
            "DOC004": 3,
            "DOC001": 1,
            "DOC002": 0,
            "DOC005": 0,
            "DOC003": 0,
            "DOC006": 0,
            "DOC007": 0,
            "DOC008": 0
        }
    },
    {
        "scenario": "Public Holidays",
        "question": "How many company holidays are observed?",
        "relevance": {
            "DOC005": 3,
            "DOC001": 0,
            "DOC002": 0,
            "DOC003": 0,
            "DOC004": 0,
            "DOC006": 0,
            "DOC007": 0,
            "DOC008": 0
        }
    },
    {
        "scenario": "IT Security",
        "question": "What security method is required to access company systems?",
        "relevance": {
            "DOC006": 3,
            "DOC001": 0,
            "DOC002": 0,
            "DOC003": 0,
            "DOC004": 0,
            "DOC005": 0,
            "DOC007": 0,
            "DOC008": 0
        }
    },
    {
        "scenario": "Travel Reimbursement",
        "question": "How can employees claim business travel expenses?",
        "relevance": {
            "DOC007": 3,
            "DOC006": 1,
            "DOC003": 0,
            "DOC001": 0,
            "DOC002": 0,
            "DOC004": 0,
            "DOC005": 0,
            "DOC008": 0
        }
    },
    {
        "scenario": "Performance Review",
        "question": "How frequently are employee performance reviews conducted?",
        "relevance": {
            "DOC008": 3,
            "DOC001": 0,
            "DOC002": 0,
            "DOC003": 0,
            "DOC004": 0,
            "DOC005": 0,
            "DOC006": 0,
            "DOC007": 0
        }
    },
    {
        "scenario": "Unsupported Insurance",
        "question": "Which health insurance company does the organization use?",
        "relevance": {
            "DOC001": 0,
            "DOC002": 0,
            "DOC003": 0,
            "DOC004": 0,
            "DOC005": 0,
            "DOC006": 0,
            "DOC007": 0,
            "DOC008": 0
        }
    },
    {
        "scenario": "Unrelated Stock Price",
        "question": "What is the current price of a technology company stock?",
        "relevance": {
            "DOC001": 0,
            "DOC002": 0,
            "DOC003": 0,
            "DOC004": 0,
            "DOC005": 0,
            "DOC006": 0,
            "DOC007": 0,
            "DOC008": 0
        }
    }
]

# ------------------------------------------------------------
# STEP 3: EMBEDDING GENERATION
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
        raise ValueError("No embedding returned by Ollama.")

    return embeddings[0]


# ------------------------------------------------------------
# STEP 4: COSINE SIMILARITY
# ------------------------------------------------------------

def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ------------------------------------------------------------
# STEP 5: DISCOUNTED CUMULATIVE GAIN (DCG@K)
#
# DCG = sum((2^relevance - 1) / log2(rank + 1))
# ------------------------------------------------------------

def calculate_dcg(relevance_scores):
    dcg = 0.0

    for rank, relevance in enumerate(relevance_scores, start=1):
        gain = (2 ** relevance) - 1
        discount = math.log2(rank + 1)

        dcg += gain / discount

    return dcg


# ------------------------------------------------------------
# STEP 6: NORMALIZED DCG (NDCG@K)
#
# NDCG = Actual DCG / Ideal DCG
# ------------------------------------------------------------

def calculate_ndcg(retrieved_scores, all_relevance_scores):
    actual_dcg = calculate_dcg(retrieved_scores)

    ideal_scores = sorted(
        all_relevance_scores,
        reverse=True
    )[:TOP_K]

    ideal_dcg = calculate_dcg(ideal_scores)

    if ideal_dcg == 0:
        return None

    return actual_dcg / ideal_dcg


# ------------------------------------------------------------
# STEP 7: HIT RATE@K
#
# Hit = At least one relevant document in top K.
# ------------------------------------------------------------

def calculate_hit_rate(relevance_scores):
    return 1 if any(score > 0 for score in relevance_scores) else 0


# ------------------------------------------------------------
# STEP 8: CHECK OLLAMA
# ------------------------------------------------------------

print("\n===== RAG RETRIEVAL RANKING EVALUATION =====")

print("\n===== STEP 1: CHECK OLLAMA CONNECTION =====")

try:
    health_response = requests.get(
        f"{OLLAMA_URL}/api/tags",
        timeout=10
    )

    health_response.raise_for_status()

    print("Ollama connection: PASS")

except requests.RequestException as error:
    print(f"Ollama connection: FAIL - {error}")
    raise SystemExit(1)


# ------------------------------------------------------------
# STEP 9: INDEX DOCUMENTS
# ------------------------------------------------------------

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
    print(f"Embedding error: {error}")
    raise SystemExit(1)

embedding_time = time.perf_counter() - embedding_start

print(f"\nDocuments indexed: {len(document_embeddings)}")
print(f"Embedding dimensions: {len(document_embeddings[0]['embedding'])}")
print(f"Embedding time: {embedding_time:.3f} seconds")


# ------------------------------------------------------------
# STEP 10: RETRIEVAL RANKING EVALUATION
# ------------------------------------------------------------

print("\n===== STEP 3: RANKING EVALUATION =====")

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
                "similarity": score,
                "relevance": test["relevance"][document["id"]]
            })

        # Sort documents by embedding similarity.
        scored_documents.sort(
            key=lambda item: item["similarity"],
            reverse=True
        )

        # Apply top-K and similarity threshold.
        top_documents = scored_documents[:TOP_K]

        retrieved_documents = [
            document
            for document in top_documents
            if document["similarity"] >= SIMILARITY_THRESHOLD
        ]

        retrieved_ids = [
            document["id"] for document in retrieved_documents
        ]

        retrieved_topics = [
            document["topic"] for document in retrieved_documents
        ]

        retrieved_relevance = [
            document["relevance"] for document in retrieved_documents
        ]

        # Relevance scores of retrieved documents, in ranking order.
        dcg = calculate_dcg(retrieved_relevance)

        # Ideal ranking uses all document relevance labels.
        all_relevance = list(test["relevance"].values())

        ndcg = calculate_ndcg(
            retrieved_relevance,
            all_relevance
        )

        hit_rate = calculate_hit_rate(retrieved_relevance)

        # Check whether the question has any relevant document.
        has_ground_truth = any(
            score > 0 for score in all_relevance
        )

        latency = time.perf_counter() - start_time

        # Test status.
        if has_ground_truth:
            status = "PASS" if hit_rate == 1 else "FAIL"
        else:
            status = "PASS" if not retrieved_documents else "REVIEW"

        result = {
            "Scenario": test["scenario"],
            "Question": test["question"],
            "Retrieved IDs": ", ".join(retrieved_ids) if retrieved_ids else "None",
            "Retrieved Topics": ", ".join(retrieved_topics) if retrieved_topics else "None",
            "Relevance Scores": ", ".join(map(str, retrieved_relevance)) if retrieved_relevance else "None",
            "DCG@K": round(dcg, 4),
            "NDCG@K": round(ndcg, 4) if ndcg is not None else "N/A",
            "Hit Rate@K": hit_rate if has_ground_truth else "N/A",
            "Retrieval Latency (seconds)": round(latency, 3),
            "Status": status
        }

        results.append(result)

        print(f"Retrieved IDs: {retrieved_ids}")
        print(f"Relevance scores: {retrieved_relevance}")
        print(f"DCG@{TOP_K}: {dcg:.4f}")

        if ndcg is not None:
            print(f"NDCG@{TOP_K}: {ndcg:.4f}")
        else:
            print(f"NDCG@{TOP_K}: N/A (no relevant documents)")

        if has_ground_truth:
            print(f"Hit Rate@{TOP_K}: {hit_rate}")
        else:
            print(f"Hit Rate@{TOP_K}: N/A (unsupported question)")

        print(f"Latency: {latency:.3f} seconds")
        print(f"Status: {status}")

    except Exception as error:
        latency = time.perf_counter() - start_time

        print(f"Test execution error: {error}")

        results.append({
            "Scenario": test["scenario"],
            "Question": test["question"],
            "Retrieved IDs": "Error",
            "Retrieved Topics": "Error",
            "Relevance Scores": "Error",
            "DCG@K": "N/A",
            "NDCG@K": "N/A",
            "Hit Rate@K": "N/A",
            "Retrieval Latency (seconds)": round(latency, 3),
            "Status": "FAIL"
        })


# ------------------------------------------------------------
# STEP 11: EXPORT DETAILED REPORT
# ------------------------------------------------------------

print("\n===== STEP 4: GENERATE DETAILED REPORT =====")

report_fields = [
    "Scenario",
    "Question",
    "Retrieved IDs",
    "Retrieved Topics",
    "Relevance Scores",
    "DCG@K",
    "NDCG@K",
    "Hit Rate@K",
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
# STEP 12: SUMMARY CALCULATION
# ------------------------------------------------------------

print("\n===== STEP 5: GENERATE SUMMARY =====")

total_tests = len(results)

passed = sum(
    1 for result in results
    if result["Status"] == "PASS"
)

failed = sum(
    1 for result in results
    if result["Status"] == "FAIL"
)

review = sum(
    1 for result in results
    if result["Status"] == "REVIEW"
)

supported_results = [
    result for result, test in zip(results, test_cases)
    if any(score > 0 for score in test["relevance"].values())
    and result["Status"] != "FAIL"
]

ndcg_values = [
    float(result["NDCG@K"])
    for result in supported_results
    if isinstance(result["NDCG@K"], (int, float))
]

hit_values = [
    float(result["Hit Rate@K"])
    for result in supported_results
    if isinstance(result["Hit Rate@K"], (int, float))
]

dcg_values = [
    float(result["DCG@K"])
    for result in supported_results
    if isinstance(result["DCG@K"], (int, float))
]

latency_values = [
    float(result["Retrieval Latency (seconds)"])
    for result in results
    if isinstance(result["Retrieval Latency (seconds)"], (int, float))
]

average_ndcg = (
    sum(ndcg_values) / len(ndcg_values)
    if ndcg_values else 0
)

average_hit_rate = (
    sum(hit_values) / len(hit_values)
    if hit_values else 0
)

average_dcg = (
    sum(dcg_values) / len(dcg_values)
    if dcg_values else 0
)

average_latency = (
    sum(latency_values) / len(latency_values)
    if latency_values else 0
)

summary_rows = [
    ["Execution Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ["Total Test Cases", total_tests],
    ["Passed", passed],
    ["Failed", failed],
    ["Review", review],
    ["Pass Rate (%)", round((passed / total_tests) * 100, 2) if total_tests else 0],
    ["Average DCG@K (supported queries)", round(average_dcg, 4)],
    ["Average NDCG@K (supported queries)", round(average_ndcg, 4)],
    ["Average Hit Rate@K (supported queries)", round(average_hit_rate, 4)],
    ["Average Retrieval Latency (seconds)", round(average_latency, 3)],
    ["Top K", TOP_K],
    ["Similarity Threshold", SIMILARITY_THRESHOLD],
    ["Documents Indexed", len(document_embeddings)],
    ["Embedding Dimensions", len(document_embeddings[0]["embedding"])]
]

with open(SUMMARY_FILE, "w", newline="", encoding="utf-8-sig") as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow(["Metric", "Value"])
    writer.writerows(summary_rows)

print(f"Summary report generated: {SUMMARY_FILE}")


# ------------------------------------------------------------
# STEP 13: FINAL RESULTS
# ------------------------------------------------------------

print("\n========== FINAL TEST SUMMARY ==========")

print(f"Total Test Cases: {total_tests}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Review: {review}")

print(f"Pass Rate: {(passed / total_tests) * 100:.2f}%")
print(f"Average DCG@{TOP_K}: {average_dcg:.4f}")
print(f"Average NDCG@{TOP_K}: {average_ndcg:.4f}")
print(f"Average Hit Rate@{TOP_K}: {average_hit_rate:.4f}")
print(f"Average Retrieval Latency: {average_latency:.3f} seconds")

print("\nGenerated Files:")
print(f"1. {REPORT_FILE}")
print(f"2. {SUMMARY_FILE}")

print("\n===== RAG RETRIEVAL RANKING EVALUATION COMPLETED =====")
