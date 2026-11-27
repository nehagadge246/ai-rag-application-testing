
"""
LESSON 75: RAG RERANKING OPTIMIZATION AND EVALUATION

Objectives:
1. Compare baseline embedding similarity ranking.
2. Implement evidence-based LLM reranking.
3. Validate LLM evidence against original documents.
4. Implement hybrid ranking using embeddings and verified LLM scores.
5. Compare Top-1 Accuracy, MRR, Precision@3, Recall@3 and latency.
6. Generate detailed CSV comparison reports.

Methods:
A. Baseline Embedding Ranking
B. Evidence-Based LLM Reranking
C. Hybrid Ranking

This experiment uses the same dataset as Lesson 74.
"""

import csv
import json
import re
import time
import requests
from datetime import datetime
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = "http://localhost:11434"

EMBEDDING_MODEL = "all-minilm:latest"
RERANKING_MODEL = "qwen2.5-coder:3b"

REQUEST_TIMEOUT = 180

CANDIDATE_TOP_K = 5
FINAL_TOP_K = 3

SIMILARITY_THRESHOLD = 0.55

# Hybrid ranking weights
EMBEDDING_WEIGHT = 0.60
LLM_WEIGHT = 0.40

OUTPUT_DIR = Path(".")

DETAIL_REPORT = OUTPUT_DIR / "rag_reranking_optimization_comparison_report.csv"
SUMMARY_REPORT = OUTPUT_DIR / "rag_reranking_optimization_summary.csv"


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

    embeddings = data.get("embeddings")

    if not embeddings or not embeddings[0]:
        raise ValueError("Embedding response is empty.")

    return embeddings[0]


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):

    if len(vector_a) != len(vector_b):
        raise ValueError("Embedding dimensions do not match.")

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ============================================================
# DOCUMENT INDEXING
# ============================================================

def build_document_index():

    print("\nBuilding document embedding index...")

    index = {}

    for document in DOCUMENTS:

        embedding = get_embedding(document["content"])

        index[document["id"]] = {
            "content": document["content"],
            "embedding": embedding
        }

        print(f"Indexed {document['id']}")

    print("Document indexing completed.")

    return index


# ============================================================
# BASELINE RETRIEVAL
# ============================================================

def retrieve_candidates(query, document_index):

    start_time = time.time()

    query_embedding = get_embedding(query)

    candidates = []

    for document in DOCUMENTS:

        document_id = document["id"]

        document_embedding = document_index[document_id]["embedding"]

        similarity = cosine_similarity(
            query_embedding,
            document_embedding
        )

        candidates.append({
            "id": document_id,
            "content": document["content"],
            "similarity": similarity
        })

    candidates.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    latency_ms = (time.time() - start_time) * 1000

    return candidates[:CANDIDATE_TOP_K], latency_ms


# ============================================================
# EVIDENCE VALIDATION
# ============================================================

def normalize_text(text):

    return re.sub(
        r"\s+",
        " ",
        text.casefold()
    ).strip()


def validate_evidence(evidence, document_content):

    if not evidence or not isinstance(evidence, str):
        return False

    normalized_evidence = normalize_text(evidence)
    normalized_document = normalize_text(document_content)

    if not normalized_evidence:
        return False

    return normalized_evidence in normalized_document


# ============================================================
# STRICT EVIDENCE-BASED LLM RERANKING
# ============================================================

def rerank_with_evidence(query, candidates):

    if not candidates:
        return [], 0.0, "No candidates"

    candidate_data = [
        {
            "id": document["id"],
            "content": document["content"]
        }
        for document in candidates
    ]

    prompt = f"""
You are a strict evidence-based relevance evaluator
for a Retrieval-Augmented Generation application.

Your task is to evaluate whether each document directly
contains evidence that answers the user's query.

USER QUERY:
{query}

CANDIDATE DOCUMENTS:
{json.dumps(candidate_data, ensure_ascii=False)}

STRICT RULES:

1. Evaluate every candidate document.
2. Give high scores only when the document directly
   answers the user's question.
3. Do not confuse related topics with actual answers.
4. Do not invent facts.
5. Every nonzero score MUST include an exact supporting
   quote copied from the document.
6. The evidence quote must be copied verbatim.
7. If the document does not answer the query,
   use score 0 and evidence as an empty string.
8. A document about public holidays does not answer
   an annual leave question.
9. A document about performance reviews does not
   establish an annual bonus entitlement.
10. Return each candidate ID exactly once.

SCORING:

90-100: Directly answers the query with clear evidence.
60-89: Contains relevant evidence but the answer is incomplete.
30-59: Related topic but does not directly answer.
0-29: Irrelevant or unsupported.

Return ONLY valid JSON in this format:

{{
  "ranked_documents": [
    {{
      "id": "DOC001",
      "score": 95,
      "evidence": "Exact sentence copied from the document",
      "reason": "Directly answers the user's question"
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
            "ranked_documents",
            []
        )

        candidate_map = {
            document["id"]: document
            for document in candidates
        }

        validated = []
        seen_ids = set()

        for item in returned_documents:

            if not isinstance(item, dict):
                continue

            document_id = item.get("id")

            if document_id not in candidate_map:
                continue

            if document_id in seen_ids:
                continue

            original = candidate_map[document_id]

            try:
                raw_score = float(item.get("score", 0))
            except (TypeError, ValueError):
                raw_score = 0

            raw_score = max(0, min(100, raw_score))

            evidence = str(item.get("evidence", ""))

            evidence_valid = validate_evidence(
                evidence,
                original["content"]
            )

            # Invalid evidence means the score cannot be trusted.
            verified_score = (
                raw_score if evidence_valid else 0
            )

            validated.append({
                "id": document_id,
                "content": original["content"],
                "similarity": original["similarity"],
                "llm_score": raw_score,
                "verified_score": verified_score,
                "evidence": evidence,
                "evidence_valid": evidence_valid,
                "reason": str(item.get("reason", ""))
            })

            seen_ids.add(document_id)

        # Missing IDs receive zero verified relevance.
        for document in candidates:

            if document["id"] not in seen_ids:

                validated.append({
                    "id": document["id"],
                    "content": document["content"],
                    "similarity": document["similarity"],
                    "llm_score": 0,
                    "verified_score": 0,
                    "evidence": "",
                    "evidence_valid": False,
                    "reason": "Document was not returned by the LLM."
                })

        latency_ms = (time.time() - start_time) * 1000

        validated.sort(
            key=lambda item: (
                item["verified_score"],
                item["similarity"]
            ),
            reverse=True
        )

        return validated, latency_ms, "Success"

    except Exception as error:

        latency_ms = (time.time() - start_time) * 1000

        print(f"Reranking error: {error}")

        fallback = []

        for document in candidates:

            fallback.append({
                "id": document["id"],
                "content": document["content"],
                "similarity": document["similarity"],
                "llm_score": 0,
                "verified_score": 0,
                "evidence": "",
                "evidence_valid": False,
                "reason": f"Fallback: {error}"
            })

        return fallback, latency_ms, f"Fallback: {error}"


# ============================================================
# RANKING STRATEGIES
# ============================================================

def build_rankings(candidates, reranked):

    baseline = sorted(
        candidates,
        key=lambda item: item["similarity"],
        reverse=True
    )

    evidence_ranking = sorted(
        reranked,
        key=lambda item: (
            item["verified_score"],
            item["similarity"]
        ),
        reverse=True
    )

    hybrid = []

    for document in reranked:

        embedding_score = max(
            0,
            min(100, document["similarity"] * 100)
        )

        hybrid_score = (
            EMBEDDING_WEIGHT * embedding_score
            + LLM_WEIGHT * document["verified_score"]
        )

        hybrid.append({
            **document,
            "embedding_score": embedding_score,
            "hybrid_score": hybrid_score
        })

    hybrid.sort(
        key=lambda item: (
            item["hybrid_score"],
            item["similarity"]
        ),
        reverse=True
    )

    return {
        "Baseline Embedding": baseline,
        "Evidence-Based LLM": evidence_ranking,
        "Hybrid Ranking": hybrid
    }


# ============================================================
# RANKING METRICS
# ============================================================

def calculate_metrics(ranking, test_case):

    expected_document = test_case["expected_document"]

    ranked_ids = [
        document["id"]
        for document in ranking
    ]

    top_ids = ranked_ids[:FINAL_TOP_K]

    if expected_document is None:

        return {
            "top1_accuracy": None,
            "mrr": None,
            "precision_at_3": None,
            "recall_at_3": None,
            "expected_rank": None,
            "top_document": ranked_ids[0] if ranked_ids else "",
            "top1_correct": None
        }

    expected_rank = (
        ranked_ids.index(expected_document) + 1
        if expected_document in ranked_ids
        else None
    )

    reciprocal_rank = (
        1 / expected_rank
        if expected_rank is not None
        else 0
    )

    relevant_retrieved = (
        1 if expected_document in top_ids else 0
    )

    precision_at_3 = relevant_retrieved / FINAL_TOP_K

    recall_at_3 = relevant_retrieved

    top1_correct = (
        bool(ranked_ids)
        and ranked_ids[0] == expected_document
    )

    return {
        "top1_accuracy": int(top1_correct),
        "mrr": reciprocal_rank,
        "precision_at_3": precision_at_3,
        "recall_at_3": recall_at_3,
        "expected_rank": expected_rank,
        "top_document": ranked_ids[0] if ranked_ids else "",
        "top1_correct": int(top1_correct)
    }


# ============================================================
# CSV REPORT GENERATION
# ============================================================

def write_csv(path, rows):

    if not rows:
        print(f"No rows available for {path}")
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Report created: {path}")


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation():

    print("=" * 65)
    print("LESSON 75: RAG RERANKING OPTIMIZATION")
    print("=" * 65)

    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Reranking model: {RERANKING_MODEL}")
    print(f"Candidate Top K: {CANDIDATE_TOP_K}")
    print(f"Final Top K: {FINAL_TOP_K}")
    print(
        f"Hybrid weights: Embedding={EMBEDDING_WEIGHT}, "
        f"LLM={LLM_WEIGHT}"
    )

    document_index = build_document_index()

    detailed_rows = []

    summary_data = {}

    for method in [
        "Baseline Embedding",
        "Evidence-Based LLM",
        "Hybrid Ranking"
    ]:

        summary_data[method] = {
            "top1": [],
            "mrr": [],
            "precision": [],
            "recall": [],
            "latency": []
        }

    for test_case in TEST_CASES:

        test_id = test_case["id"]
        query = test_case["query"]

        print("\n" + "-" * 65)
        print(f"{test_id}: {query}")
        print("-" * 65)

        candidates, retrieval_latency = retrieve_candidates(
            query,
            document_index
        )

        reranked, reranking_latency, reranking_status = (
            rerank_with_evidence(
                query,
                candidates
            )
        )

        rankings = build_rankings(
            candidates,
            reranked
        )

        expected_document = test_case["expected_document"]

        print(f"Expected document: {expected_document}")
        print(f"Retrieval latency: {retrieval_latency:.2f} ms")
        print(f"Reranking latency: {reranking_latency:.2f} ms")
        print(f"Reranking status: {reranking_status}")

        for method, ranking in rankings.items():

            metrics = calculate_metrics(
                ranking,
                test_case
            )

            if method == "Baseline Embedding":
                method_latency = retrieval_latency

            elif method == "Evidence-Based LLM":
                method_latency = (
                    retrieval_latency + reranking_latency
                )

            else:
                method_latency = (
                    retrieval_latency + reranking_latency
                )

            summary_data[method]["latency"].append(
                method_latency
            )

            if metrics["top1_accuracy"] is not None:

                summary_data[method]["top1"].append(
                    metrics["top1_accuracy"]
                )

                summary_data[method]["mrr"].append(
                    metrics["mrr"]
                )

                summary_data[method]["precision"].append(
                    metrics["precision_at_3"]
                )

                summary_data[method]["recall"].append(
                    metrics["recall_at_3"]
                )

            print(f"\nMethod: {method}")
            print(f"Top document: {metrics['top_document']}")
            print(f"Expected rank: {metrics['expected_rank']}")
            print(f"MRR: {metrics['mrr']}")
            print(f"Precision@3: {metrics['precision_at_3']}")
            print(f"Recall@3: {metrics['recall_at_3']}")

            for rank, document in enumerate(ranking, start=1):

                detail = {
                    "Execution Date": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "Test Case ID": test_id,
                    "Query": query,
                    "Supported": test_case["supported"],
                    "Expected Document": expected_document or "",
                    "Method": method,
                    "Rank": rank,
                    "Retrieved Document": document["id"],
                    "Similarity Score": round(
                        document["similarity"], 5
                    ),
                    "LLM Score": document.get(
                        "llm_score", ""
                    ),
                    "Verified LLM Score": document.get(
                        "verified_score", ""
                    ),
                    "Evidence Valid": document.get(
                        "evidence_valid", ""
                    ),
                    "Evidence": document.get(
                        "evidence", ""
                    ),
                    "Hybrid Score": round(
                        document.get("hybrid_score", 0), 3
                    ) if method == "Hybrid Ranking" else "",
                    "Expected Rank": metrics["expected_rank"] or "",
                    "Top1 Correct": metrics["top1_correct"],
                    "MRR": metrics["mrr"],
                    "Precision@3": metrics["precision_at_3"],
                    "Recall@3": metrics["recall_at_3"],
                    "Retrieval Latency Ms": round(
                        retrieval_latency, 2
                    ),
                    "Reranking Latency Ms": round(
                        reranking_latency, 2
                    ),
                    "Method Latency Ms": round(
                        method_latency, 2
                    ),
                    "Reranking Status": reranking_status
                }

                detailed_rows.append(detail)

    # ========================================================
    # SUMMARY REPORT
    # ========================================================

    summary_rows = []

    for method, values in summary_data.items():

        def average(items):
            return (
                sum(items) / len(items)
                if items else 0
            )

        summary_rows.append({
            "Method": method,
            "Supported Test Cases": len(values["top1"]),
            "Top1 Accuracy": round(
                average(values["top1"]), 4
            ),
            "MRR": round(
                average(values["mrr"]), 4
            ),
            "Precision@3": round(
                average(values["precision"]), 4
            ),
            "Recall@3": round(
                average(values["recall"]), 4
            ),
            "Average Latency Ms": round(
                average(values["latency"]), 2
            ),
            "Embedding Weight": (
                EMBEDDING_WEIGHT
                if method == "Hybrid Ranking"
                else ""
            ),
            "LLM Weight": (
                LLM_WEIGHT
                if method == "Hybrid Ranking"
                else ""
            )
        })

    write_csv(DETAIL_REPORT, detailed_rows)

    write_csv(SUMMARY_REPORT, summary_rows)

    # ========================================================
    # FINAL CONSOLE SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("LESSON 75 SUMMARY")
    print("=" * 65)

    for row in summary_rows:

        print(f"\n{row['Method']}")
        print(f"Top1 Accuracy: {row['Top1 Accuracy']}")
        print(f"MRR: {row['MRR']}")
        print(f"Precision@3: {row['Precision@3']}")
        print(f"Recall@3: {row['Recall@3']}")
        print(f"Average Latency: {row['Average Latency Ms']} ms")

    print("\nGenerated reports:")
    print(DETAIL_REPORT)
    print(SUMMARY_REPORT)

    print("\nLesson 75 evaluation completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        run_evaluation()

    except requests.exceptions.ConnectionError:

        print(
            "\nERROR: Cannot connect to Ollama. "
            "Make sure Ollama is running."
        )

    except Exception as error:

        print(f"\nERROR: {error}")
