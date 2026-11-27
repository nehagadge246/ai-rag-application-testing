
"""
LESSON 75: RAG ANSWERABILITY OPTIMIZATION

Improvements:
1. Evaluate each document independently.
2. Skip LLM evaluation when similarity is below threshold.
3. Validate exact evidence against the original document.
4. Distinguish failed evaluations from valid zero scores.
5. Calculate classification metrics.
6. Generate detailed and summary CSV reports.

Acceptance requires:
- Similarity >= configured threshold
- Verified LLM score >= configured threshold
- Exact supporting evidence exists
"""

import csv
import json
import re
import time
import requests

from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = "http://localhost:11434"

EMBEDDING_MODEL = "all-minilm:latest"
RERANKING_MODEL = "qwen2.5-coder:3b"

REQUEST_TIMEOUT = 180

CANDIDATE_TOP_K = 5

SIMILARITY_THRESHOLD = 0.55
LLM_SCORE_THRESHOLD = 55

DETAIL_REPORT = Path(
    "rag_answerability_optimization_report.csv"
)

SUMMARY_REPORT = Path(
    "rag_answerability_optimization_summary.csv"
)


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
        raise ValueError("Empty embedding response.")

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

    document_index = {}

    for document in DOCUMENTS:

        document_index[document["id"]] = {
            "content": document["content"],
            "embedding": get_embedding(document["content"])
        }

        print(f"Indexed {document['id']}")

    return document_index


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_candidates(query, document_index):

    start_time = time.time()

    query_embedding = get_embedding(query)

    candidates = []

    for document in DOCUMENTS:

        document_id = document["id"]

        similarity = cosine_similarity(
            query_embedding,
            document_index[document_id]["embedding"]
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

    return (
        normalize_text(evidence)
        in normalize_text(document_content)
    )


# ============================================================
# SINGLE-DOCUMENT LLM EVALUATION
# ============================================================

def evaluate_single_document(query, document):

    prompt = f"""
You are a strict evidence-based RAG answerability evaluator.

USER QUERY:
{query}

DOCUMENT:
{document["content"]}

Your task:
Determine whether THIS DOCUMENT directly answers the user query.

Important rules:
1. Evaluate only the document provided.
2. Understand the meaning of the query, not just matching keywords.
3. A paraphrased question can be answered by a document.
4. If the document directly provides the requested information,
   give a score between 90 and 100.
5. If the document is related but does not provide the answer,
   give a score between 0 and 59.
6. Never invent information.
7. For a score of 60 or higher, copy one exact supporting
   sentence from the document into the evidence field.
8. The evidence must be copied exactly, including its wording.
9. If the document does not answer the query, use score 0
   and an empty evidence string.

Examples:
Query: What is the sick leave entitlement?
Document: Employees receive 10 sick leave days per year.
Correct score: 95
Correct evidence: Employees receive 10 sick leave days per year.

Query: How many days can employees work from home?
Document: Employees can work remotely up to two days per week
with manager approval.
Correct score: 95
Correct evidence: Employees can work remotely up to two days per week
with manager approval.

Query: What is the employee insurance premium?
Document: Employees receive 15 annual leave days per year.
Correct score: 0
Correct evidence: empty string

Return ONLY valid JSON in this format:

{{
    "score": 95,
    "evidence": "Exact sentence copied from the document",
    "reason": "The document directly answers the query"
}}
"""

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

    if not isinstance(parsed, dict):
        raise ValueError("LLM response is not a JSON object.")

    if "score" not in parsed or "evidence" not in parsed:
        raise ValueError("LLM response is missing required fields.")

    try:
        score = float(parsed["score"])
    except (TypeError, ValueError):
        raise ValueError("LLM returned an invalid score.")

    if not 0 <= score <= 100:
        raise ValueError("LLM score is outside the 0-100 range.")

    evidence = parsed["evidence"]

    if not isinstance(evidence, str):
        raise ValueError("LLM evidence must be a string.")

    reason = str(parsed.get("reason", ""))

    valid_evidence = validate_evidence(
        evidence,
        document["content"]
    )

    verified_score = (
        score
        if valid_evidence
        else 0
    )

    # A high score without verifiable evidence is not accepted.
    return {
        "llm_score": score,
        "verified_score": verified_score,
        "evidence": evidence,
        "evidence_valid": valid_evidence,
        "reason": reason,
        "evaluation_status": "Success"
    }


# ============================================================
# MULTI-DOCUMENT RELEVANCE EVALUATION
# ============================================================

def evaluate_relevance(query, candidates):

    if not candidates:
        return {}, 0.0, "No candidates"

    start_time = time.time()

    results = {}

    successful = 0
    failed = 0
    skipped = 0

    for document in candidates:

        document_id = document["id"]

        # Do not spend LLM time on documents that cannot
        # pass the similarity acceptance condition.
        if document["similarity"] < SIMILARITY_THRESHOLD:

            results[document_id] = {
                "llm_score": 0,
                "verified_score": 0,
                "evidence": "",
                "evidence_valid": False,
                "reason": "Skipped: similarity below threshold",
                "evaluation_status": "Skipped"
            }

            skipped += 1
            continue

        print(f"Evaluating {document_id} individually...")

        try:

            result = evaluate_single_document(
                query,
                document
            )

            results[document_id] = result
            successful += 1

            print(
                f"{document_id}: "
                f"LLM score={result['llm_score']}, "
                f"verified score={result['verified_score']}, "
                f"evidence valid={result['evidence_valid']}"
            )

        except Exception as error:

            results[document_id] = {
                "llm_score": 0,
                "verified_score": 0,
                "evidence": "",
                "evidence_valid": False,
                "reason": f"Evaluation error: {error}",
                "evaluation_status": "Failed"
            }

            failed += 1

            print(f"{document_id} evaluation failed: {error}")

    latency_ms = (time.time() - start_time) * 1000

    status = (
        f"Success: {successful}, "
        f"Failed: {failed}, "
        f"Skipped: {skipped}"
    )

    return results, latency_ms, status


# ============================================================
# ANSWERABILITY DECISION
# ============================================================

def make_answerability_decision(
    candidates,
    relevance_results
):

    eligible_documents = []

    for document in candidates:

        document_id = document["id"]

        relevance = relevance_results[document_id]

        similarity = document["similarity"]

        verified_score = relevance["verified_score"]

        evidence_valid = relevance["evidence_valid"]

        accepted = (
            similarity >= SIMILARITY_THRESHOLD
            and verified_score >= LLM_SCORE_THRESHOLD
            and evidence_valid
        )

        eligible_documents.append({
            **document,
            **relevance,
            "accepted": accepted
        })

    eligible_documents.sort(
        key=lambda item: (
            item["accepted"],
            item["verified_score"],
            item["similarity"]
        ),
        reverse=True
    )

    accepted_documents = [
        item for item in eligible_documents
        if item["accepted"]
    ]

    answerable = len(accepted_documents) > 0

    return answerable, eligible_documents, accepted_documents


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def calculate_classification_metrics(rows):

    tp = sum(
        1 for row in rows
        if row["expected_supported"] and row["predicted_supported"]
    )

    tn = sum(
        1 for row in rows
        if not row["expected_supported"]
        and not row["predicted_supported"]
    )

    fp = sum(
        1 for row in rows
        if not row["expected_supported"]
        and row["predicted_supported"]
    )

    fn = sum(
        1 for row in rows
        if row["expected_supported"]
        and not row["predicted_supported"]
    )

    total = len(rows)

    accuracy = (
        (tp + tn) / total
        if total else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )

    rejection_accuracy = (
        tn / (tn + fp)
        if (tn + fp) else 0
    )

    return {
        "True Positives": tp,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "Rejection Accuracy": round(rejection_accuracy, 4)
    }


# ============================================================
# CSV WRITER
# ============================================================

def write_csv(path, rows):

    if not rows:
        print(f"No rows to write: {path}")
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
    print("LESSON 75: RAG ANSWERABILITY OPTIMIZATION")
    print("=" * 65)

    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Reranking model: {RERANKING_MODEL}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"LLM score threshold: {LLM_SCORE_THRESHOLD}")
    print("Evaluation mode: Individual document evaluation")

    document_index = build_document_index()

    detailed_rows = []
    classification_rows = []

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

        relevance_results, llm_latency, llm_status = (
            evaluate_relevance(
                query,
                candidates
            )
        )

        predicted_supported, evaluated_documents, accepted = (
            make_answerability_decision(
                candidates,
                relevance_results
            )
        )

        expected_supported = test_case["supported"]

        decision = (
            "ACCEPT" if predicted_supported else "REJECT"
        )

        correct = (
            predicted_supported == expected_supported
        )

        top_document = (
            accepted[0]["id"] if accepted else ""
        )

        print(
            f"Expected decision: "
            f"{'ACCEPT' if expected_supported else 'REJECT'}"
        )

        print(f"Predicted decision: {decision}")
        print(f"Decision correct: {correct}")
        print(f"Accepted document: {top_document or 'None'}")
        print(f"Retrieval latency: {retrieval_latency:.2f} ms")
        print(f"LLM latency: {llm_latency:.2f} ms")
        print(f"LLM status: {llm_status}")

        classification_rows.append({
            "test_case_id": test_id,
            "expected_supported": expected_supported,
            "predicted_supported": predicted_supported
        })

        for rank, document in enumerate(
            evaluated_documents,
            start=1
        ):

            detail = {
                "Execution Date": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "Test Case ID": test_id,
                "Query": query,
                "Expected Supported": expected_supported,
                "Predicted Supported": predicted_supported,
                "Decision": decision,
                "Decision Correct": correct,
                "Rank": rank,
                "Document ID": document["id"],
                "Similarity Score": round(
                    document["similarity"], 5
                ),
                "LLM Score": document["llm_score"],
                "Verified LLM Score": document["verified_score"],
                "Evidence Valid": document["evidence_valid"],
                "Evidence": document["evidence"],
                "Reason": document["reason"],
                "LLM Evaluation Status": document["evaluation_status"],
                "Similarity Passed": (
                    document["similarity"]
                    >= SIMILARITY_THRESHOLD
                ),
                "LLM Score Passed": (
                    document["verified_score"]
                    >= LLM_SCORE_THRESHOLD
                ),
                "Document Accepted": document["accepted"],
                "Expected Document": (
                    test_case["expected_document"] or ""
                ),
                "Retrieval Latency Ms": round(
                    retrieval_latency, 2
                ),
                "LLM Latency Ms": round(
                    llm_latency, 2
                ),
                "LLM Status": llm_status
            }

            detailed_rows.append(detail)

    # ========================================================
    # CLASSIFICATION SUMMARY
    # ========================================================

    metrics = calculate_classification_metrics(
        classification_rows
    )

    summary_row = {
        "Experiment": "Individual Similarity + Verified Evidence Gate",
        "Test Cases": len(classification_rows),
        "Similarity Threshold": SIMILARITY_THRESHOLD,
        "LLM Score Threshold": LLM_SCORE_THRESHOLD,
        **metrics
    }

    write_csv(
        DETAIL_REPORT,
        detailed_rows
    )

    write_csv(
        SUMMARY_REPORT,
        [summary_row]
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("ANSWERABILITY EVALUATION SUMMARY")
    print("=" * 65)

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print("\nGenerated reports:")
    print(DETAIL_REPORT)
    print(SUMMARY_REPORT)

    print("\nAnswerability evaluation completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        run_evaluation()

    except requests.exceptions.ConnectionError:

        print(
            "\nERROR: Cannot connect to Ollama. "
            "Check that Ollama is running."
        )

    except Exception as error:

        print(f"\nERROR: {error}")
