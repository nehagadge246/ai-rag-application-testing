
"""
LESSON 74: RAG RETRIEVAL RERANKING

Objectives:
1. Retrieve candidate documents using embedding similarity.
2. Rerank candidates using a local Qwen model through Ollama.
3. Compare initial and reranked document rankings.
4. Measure Precision@K, Recall@K and Mean Reciprocal Rank.
5. Identify missed relevant documents and false positives.
6. Measure retrieval and reranking latency.
7. Validate unsupported-query rejection.
8. Generate detailed CSV reports with decision reasons.

Models:
- Embedding: all-minilm:latest
- Reranker: qwen2.5-coder:3b
"""

import csv
import json
import time
import requests
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = "http://localhost:11434"

EMBEDDING_MODEL = "all-minilm:latest"
RERANKING_MODEL = "qwen2.5-coder:3b"

CANDIDATE_TOP_K = 5
FINAL_TOP_K = 3

SIMILARITY_THRESHOLD = 0.55
RERANKING_SCORE_THRESHOLD = 55

REQUEST_TIMEOUT = 120

DETAIL_REPORT = "rag_reranking_comparison_report.csv"
SUMMARY_REPORT = "rag_reranking_summary.csv"


# ============================================================
# DOCUMENT DATASET
# ============================================================

DOCUMENTS = [
    {
        "id": "DOC001",
        "content": (
            "Employees receive 15 annual leave days per year. "
            "Annual leave requires manager approval."
        )
    },
    {
        "id": "DOC002",
        "content": (
            "Employees receive 10 sick leave days per year. "
            "A medical certificate may be required."
        )
    },
    {
        "id": "DOC003",
        "content": (
            "Employees can work remotely up to two days per week "
            "with manager approval."
        )
    },
    {
        "id": "DOC004",
        "content": (
            "Eligible employees receive 26 weeks of maternity leave."
        )
    },
    {
        "id": "DOC005",
        "content": (
            "Employees receive 12 public holidays each calendar year."
        )
    },
    {
        "id": "DOC006",
        "content": (
            "Multi-factor authentication is required. "
            "Employees must never share their passwords."
        )
    },
    {
        "id": "DOC007",
        "content": (
            "Business travel expenses can be reimbursed "
            "when employees provide receipts."
        )
    },
    {
        "id": "DOC008",
        "content": (
            "Employee performance reviews are conducted twice a year."
        )
    }
]


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    {
        "id": "TC01",
        "query": "How many annual leave days do employees receive?",
        "expected_document": "DOC001",
        "supported": True
    },
    {
        "id": "TC02",
        "query": "What is the sick leave entitlement?",
        "expected_document": "DOC002",
        "supported": True
    },
    {
        "id": "TC03",
        "query": "How many days can employees work from home?",
        "expected_document": "DOC003",
        "supported": True
    },
    {
        "id": "TC04",
        "query": "How long is maternity leave?",
        "expected_document": "DOC004",
        "supported": True
    },
    {
        "id": "TC05",
        "query": "What is the employee insurance premium?",
        "expected_document": None,
        "supported": False
    },
    {
        "id": "TC06",
        "query": "Are employees guaranteed an annual bonus?",
        "expected_document": None,
        "supported": False
    }
]


# ============================================================
# EMBEDDING GENERATION
# ============================================================

def get_embedding(text):

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("embeddings"):
        raise ValueError("Embedding response is empty.")

    return data["embeddings"][0]


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):

    if len(vector_a) != len(vector_b):
        raise ValueError("Embedding dimensions do not match.")

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = sum(
        a * a for a in vector_a
    ) ** 0.5

    magnitude_b = sum(
        b * b for b in vector_b
    ) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ============================================================
# BUILD DOCUMENT INDEX
# ============================================================

def build_index():

    print("\nBuilding document embedding index...")

    index = []

    for document in DOCUMENTS:

        embedding = get_embedding(document["content"])

        index.append({
            "id": document["id"],
            "content": document["content"],
            "embedding": embedding
        })

        print(f"Embedded: {document['id']}")

    print(f"\nTotal indexed documents: {len(index)}")

    return index


# ============================================================
# INITIAL EMBEDDING RETRIEVAL
# ============================================================

def retrieve_candidates(query, index):

    query_embedding = get_embedding(query)

    results = []

    for document in index:

        similarity = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        results.append({
            "id": document["id"],
            "content": document["content"],
            "similarity": similarity
        })

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return results[:CANDIDATE_TOP_K]


# ============================================================
# LLM-BASED RERANKING
# ============================================================

def rerank_documents(query, candidates):

    if not candidates:
        return [], 0.0, "No candidates"

    candidate_text = []

    for document in candidates:

        candidate_text.append({
            "id": document["id"],
            "content": document["content"]
        })

    prompt = f"""
You are a strict document relevance evaluator for a RAG application.

Evaluate whether each candidate document contains evidence
that directly answers the user's query.

USER QUERY:
{query}

CANDIDATE DOCUMENTS:
{json.dumps(candidate_text, ensure_ascii=False)}

Instructions:
1. Evaluate whether the document directly answers the query.
2. Do not give high scores merely because the topic is related.
3. A document about annual leave does not answer an insurance question.
4. A document about public holidays does not answer an annual bonus question.
5. Give each document an integer score from 0 to 100.
6. 90-100: Directly answers the query.
7. 60-89: Contains useful evidence but may be incomplete.
8. 30-59: Only loosely related.
9. 0-29: Irrelevant or unrelated.
10. Do not invent facts.
11. Return every candidate document exactly once.
12. Do not create document IDs.
13. Sort by score from highest to lowest.

Return ONLY valid JSON:

{{
  "ranked_documents": [
    {{
      "id": "DOC001",
      "score": 90,
      "reason": "Contains evidence directly answering the query."
    }}
  ]
}}
"""

    start_time = time.time()

    try:

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": RERANKING_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0
                }
            },
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        response_data = response.json()

        content = response_data["message"]["content"]

        parsed = json.loads(content)

        returned_documents = parsed.get(
            "ranked_documents", []
        )

        candidate_ids = {
            document["id"] for document in candidates
        }

        validated = []
        seen_ids = set()

        for item in returned_documents:

            if not isinstance(item, dict):
                continue

            document_id = item.get("id")

            if document_id not in candidate_ids:
                continue

            if document_id in seen_ids:
                continue

            try:
                score = float(item.get("score", 0))
            except (TypeError, ValueError):
                score = 0

            score = max(0, min(100, score))

            original = next(
                document for document in candidates
                if document["id"] == document_id
            )

            validated.append({
                "id": document_id,
                "content": original["content"],
                "similarity": original["similarity"],
                "reranking_score": score,
                "reason": str(item.get("reason", ""))
            })

            seen_ids.add(document_id)

        if not validated:
            raise ValueError(
                "Reranker returned no valid candidate documents."
            )

        # Preserve candidates omitted by the LLM.
        for document in candidates:

            if document["id"] not in seen_ids:

                validated.append({
                    "id": document["id"],
                    "content": document["content"],
                    "similarity": document["similarity"],
                    "reranking_score": 0,
                    "reason": "Not returned by reranker."
                })

        validated.sort(
            key=lambda item: item["reranking_score"],
            reverse=True
        )

        latency = (time.time() - start_time) * 1000

        return validated, latency, "Success"

    except Exception as error:

        latency = (time.time() - start_time) * 1000

        print(f"Reranking error: {error}")

        # Preserve original embedding ranking on failure.
        fallback = []

        for document in candidates:

            fallback.append({
                "id": document["id"],
                "content": document["content"],
                "similarity": document["similarity"],
                "reranking_score": 0,
                "reason": "Fallback to embedding ranking."
            })

        return fallback, latency, f"Fallback: {error}"


# ============================================================
# METRIC CALCULATIONS
# ============================================================

def calculate_reciprocal_rank(ranking, expected_document):

    if not expected_document:
        return None

    for position, document in enumerate(ranking, start=1):

        if document["id"] == expected_document:
            return 1 / position

    return 0.0


def calculate_precision_at_k(ranking, expected_document, k):

    if not expected_document:
        return None

    top_documents = ranking[:k]

    relevant_count = sum(
        1 for document in top_documents
        if document["id"] == expected_document
    )

    return relevant_count / k


def calculate_recall_at_k(ranking, expected_document, k):

    if not expected_document:
        return None

    top_documents = ranking[:k]

    relevant_count = sum(
        1 for document in top_documents
        if document["id"] == expected_document
    )

    return float(relevant_count)


def find_expected_rank(ranking, expected_document):

    if not expected_document:
        return None

    for position, document in enumerate(ranking, start=1):

        if document["id"] == expected_document:
            return position

    return None


# ============================================================
# RUN RERANKING AUDIT
# ============================================================

def run_audit(index):

    detail_rows = []
    case_results = []

    print("\n" + "=" * 70)
    print("RAG RETRIEVAL RERANKING AUDIT")
    print("=" * 70)

    for test_case in TEST_CASES:

        case_id = test_case["id"]
        query = test_case["query"]
        expected = test_case["expected_document"]
        supported = test_case["supported"]

        print(f"\n{case_id}: {query}")

        # Initial embedding retrieval
        retrieval_start = time.time()

        candidates = retrieve_candidates(query, index)

        retrieval_latency = (
            time.time() - retrieval_start
        ) * 1000

        # LLM reranking
        reranked, reranking_latency, reranking_status = (
            rerank_documents(query, candidates)
        )

        initial_ranking = candidates

        initial_rank = find_expected_rank(
            initial_ranking, expected
        )

        reranked_rank = find_expected_rank(
            reranked, expected
        )

        initial_mrr = calculate_reciprocal_rank(
            initial_ranking, expected
        )

        reranked_mrr = calculate_reciprocal_rank(
            reranked, expected
        )

        initial_precision = calculate_precision_at_k(
            initial_ranking, expected, FINAL_TOP_K
        )

        reranked_precision = calculate_precision_at_k(
            reranked, expected, FINAL_TOP_K
        )

        initial_recall = calculate_recall_at_k(
            initial_ranking, expected, FINAL_TOP_K
        )

        reranked_recall = calculate_recall_at_k(
            reranked, expected, FINAL_TOP_K
        )

        top_score = (
            reranked[0]["reranking_score"]
            if reranked else 0
        )

        max_similarity = max(
            (
                document["similarity"]
                for document in candidates
            ),
            default=0
        )

        # Two-stage acceptance validation.
        similarity_check = (
            max_similarity >= SIMILARITY_THRESHOLD
        )

        reranking_check = (
            top_score >= RERANKING_SCORE_THRESHOLD
        )

        accepted = (
            similarity_check and reranking_check
        )

        if not similarity_check:

            acceptance_reason = (
                "Rejected: no candidate met the embedding "
                "similarity threshold."
            )

        elif not reranking_check:

            acceptance_reason = (
                "Rejected: reranking score is below "
                "the required threshold."
            )

        else:

            acceptance_reason = (
                "Accepted: embedding similarity and "
                "reranking checks passed."
            )

        # Check whether expected document was retrieved.
        candidate_ids = {
            document["id"] for document in candidates
        }

        expected_in_candidates = (
            expected in candidate_ids
            if expected else None
        )

        result = {
            "test_case_id": case_id,
            "query": query,
            "expected_document": expected or "NONE",
            "supported": supported,
            "initial_rank": initial_rank,
            "reranked_rank": reranked_rank,
            "initial_mrr": initial_mrr,
            "reranked_mrr": reranked_mrr,
            "initial_precision": initial_precision,
            "reranked_precision": reranked_precision,
            "initial_recall": initial_recall,
            "reranked_recall": reranked_recall,
            "expected_in_candidates": expected_in_candidates,
            "top_reranking_score": top_score,
            "max_similarity": max_similarity,
            "similarity_check": similarity_check,
            "reranking_check": reranking_check,
            "accepted": accepted,
            "acceptance_reason": acceptance_reason,
            "retrieval_latency_ms": round(
                retrieval_latency, 2
            ),
            "reranking_latency_ms": round(
                reranking_latency, 2
            ),
            "reranking_status": reranking_status
        }

        case_results.append(result)

        print(f"Initial expected rank: {initial_rank}")
        print(f"Reranked expected rank: {reranked_rank}")
        print(f"Maximum similarity: {max_similarity:.4f}")
        print(f"Top reranking score: {top_score}")
        print(f"Similarity check: {similarity_check}")
        print(f"Reranking check: {reranking_check}")
        print(f"Accepted: {accepted}")
        print(f"Decision: {acceptance_reason}")
        print(f"Retrieval latency: {retrieval_latency:.2f} ms")
        print(f"Reranking latency: {reranking_latency:.2f} ms")

        # Record every candidate for detailed analysis.
        for position, document in enumerate(
            reranked, start=1
        ):

            initial_position = next(
                (
                    i for i, item in enumerate(
                        initial_ranking, start=1
                    )
                    if item["id"] == document["id"]
                ),
                None
            )

            is_expected = (
                document["id"] == expected
                if expected else False
            )

            detail_rows.append({
                "Execution Date": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "Test Case ID": case_id,
                "Query": query,
                "Expected Document": expected or "NONE",
                "Document ID": document["id"],
                "Initial Rank": initial_position,
                "Reranked Rank": position,
                "Similarity Score": round(
                    document["similarity"], 4
                ),
                "Reranking Score": document[
                    "reranking_score"
                ],
                "Maximum Query Similarity": round(
                    max_similarity, 4
                ),
                "Similarity Check": similarity_check,
                "Reranking Check": reranking_check,
                "Is Expected Document": is_expected,
                "Is Supported Query": supported,
                "Accepted": accepted,
                "Acceptance Reason": acceptance_reason,
                "Reranking Reason": document["reason"],
                "Document Content": document["content"]
            })

    return detail_rows, case_results


# ============================================================
# GENERATE CSV REPORTS
# ============================================================

def generate_reports(detail_rows, case_results):

    if detail_rows:

        with open(
            DETAIL_REPORT,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=detail_rows[0].keys()
            )

            writer.writeheader()
            writer.writerows(detail_rows)

    supported_results = [
        result for result in case_results
        if result["supported"]
    ]

    unsupported_results = [
        result for result in case_results
        if not result["supported"]
    ]

    def average_metric(results, key):

        values = [
            result[key]
            for result in results
            if result[key] is not None
        ]

        if not values:
            return 0.0

        return sum(values) / len(values)

    true_acceptances = sum(
        1 for result in supported_results
        if result["accepted"]
    )

    false_rejections = sum(
        1 for result in supported_results
        if not result["accepted"]
    )

    true_rejections = sum(
        1 for result in unsupported_results
        if not result["accepted"]
    )

    false_acceptances = sum(
        1 for result in unsupported_results
        if result["accepted"]
    )

    summary = {
        "Execution Date": datetime.now().isoformat(
            timespec="seconds"
        ),
        "Total Test Cases": len(case_results),
        "Supported Queries": len(supported_results),
        "Unsupported Queries": len(unsupported_results),

        "Initial MRR": round(
            average_metric(supported_results, "initial_mrr"),
            4
        ),

        "Reranked MRR": round(
            average_metric(supported_results, "reranked_mrr"),
            4
        ),

        "Initial Precision@3": round(
            average_metric(
                supported_results, "initial_precision"
            ),
            4
        ),

        "Reranked Precision@3": round(
            average_metric(
                supported_results, "reranked_precision"
            ),
            4
        ),

        "Initial Recall@3": round(
            average_metric(supported_results, "initial_recall"),
            4
        ),

        "Reranked Recall@3": round(
            average_metric(
                supported_results, "reranked_recall"
            ),
            4
        ),

        "Expected Documents in Candidate Set": sum(
            1 for result in supported_results
            if result["expected_in_candidates"]
        ),

        "Reranked Expected Documents in Top 3": sum(
            1 for result in supported_results
            if result["reranked_rank"] is not None
            and result["reranked_rank"] <= FINAL_TOP_K
        ),

        "True Acceptances": true_acceptances,
        "False Rejections": false_rejections,
        "True Rejections": true_rejections,
        "False Acceptances": false_acceptances,

        "Unsupported Queries Rejected": true_rejections,

        "Acceptance Accuracy": round(
            (
                true_acceptances + true_rejections
            ) / len(case_results),
            4
        ) if case_results else 0,

        "Average Retrieval Latency Ms": round(
            average_metric(
                case_results, "retrieval_latency_ms"
            ),
            2
        ),

        "Average Reranking Latency Ms": round(
            average_metric(
                case_results, "reranking_latency_ms"
            ),
            2
        )
    }

    with open(
        SUMMARY_REPORT,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary.keys()
        )

        writer.writeheader()
        writer.writerow(summary)

    return summary


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("=" * 70)
    print("LESSON 74: RAG RETRIEVAL RERANKING")
    print("=" * 70)

    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Reranking Model: {RERANKING_MODEL}")
    print(f"Candidate Top K: {CANDIDATE_TOP_K}")
    print(f"Final Top K: {FINAL_TOP_K}")

    print(
        f"Similarity Threshold: {SIMILARITY_THRESHOLD}"
    )

    print(
        f"Reranking Acceptance Threshold: "
        f"{RERANKING_SCORE_THRESHOLD}"
    )

    try:

        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=10
        )

        response.raise_for_status()

        print("\nOllama connection: SUCCESS")

        index = build_index()

        detail_rows, case_results = run_audit(index)

        summary = generate_reports(
            detail_rows,
            case_results
        )

        print("\n" + "=" * 70)
        print("LESSON 74 SUMMARY")
        print("=" * 70)

        for key, value in summary.items():
            print(f"{key}: {value}")

        print("\nGenerated Reports:")
        print(f"1. {DETAIL_REPORT}")
        print(f"2. {SUMMARY_REPORT}")

        print("\nLesson 74 execution completed.")

    except Exception as error:

        print("\nLESSON 74 EXECUTION FAILED")
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
