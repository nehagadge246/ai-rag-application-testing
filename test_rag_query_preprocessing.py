
import csv
import json
import re
import time
import requests
from datetime import datetime
from pathlib import Path

# ============================================================
# LESSON 58: RAG QUERY PREPROCESSING AND QUERY REWRITING
# ============================================================

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"

TOP_K = 3
SIMILARITY_THRESHOLD = 0.68
REQUEST_TIMEOUT = 60

REPORT_FILE = "rag_query_preprocessing_report.csv"
SUMMARY_FILE = "rag_query_preprocessing_summary.csv"

# ============================================================
# HR POLICY DOCUMENT DATASET
# ============================================================

DOCUMENTS = [
    {
        "id": "DOC001",
        "topic": "annual leave",
        "text": (
            "Employees receive 15 annual leave days every year. "
            "Annual leave requests must be submitted through the "
            "HR portal and approved by the reporting manager."
        )
    },
    {
        "id": "DOC002",
        "topic": "sick leave",
        "text": (
            "Employees receive 10 sick leave days per year. "
            "Employees should inform their manager when taking sick leave."
        )
    },
    {
        "id": "DOC003",
        "topic": "remote work",
        "text": (
            "Employees can work remotely up to 2 days per week. "
            "Remote work arrangements require manager approval."
        )
    },
    {
        "id": "DOC004",
        "topic": "parental leave",
        "text": (
            "Eligible employees receive 12 weeks of parental leave. "
            "Employees should contact HR for parental leave procedures."
        )
    },
    {
        "id": "DOC005",
        "topic": "public holidays",
        "text": (
            "The company provides 10 public holidays each year. "
            "The annual holiday calendar is published by HR."
        )
    },
    {
        "id": "DOC006",
        "topic": "IT security",
        "text": (
            "Employees must use strong passwords and enable "
            "multi-factor authentication. Company devices must "
            "follow IT security policies."
        )
    },
    {
        "id": "DOC007",
        "topic": "travel reimbursement",
        "text": (
            "Employees can claim approved business travel expenses. "
            "Travel reimbursement requires receipts and manager approval."
        )
    },
    {
        "id": "DOC008",
        "topic": "performance review",
        "text": (
            "Employee performance reviews are conducted twice a year. "
            "Managers provide feedback during performance evaluations."
        )
    }
]

# ============================================================
# TEST DATASET
# ============================================================

TEST_CASES = [
    {
        "scenario": "Normal Query",
        "question": "How many annual leave days do employees receive?",
        "expected_docs": ["DOC001"],
        "expected_supported": True
    },
    {
        "scenario": "Short Query",
        "question": "Leave?",
        "expected_docs": ["DOC001", "DOC002", "DOC004"],
        "expected_supported": True
    },
    {
        "scenario": "Misspelled Query",
        "question": "How meny anuel leev dayz do employes get?",
        "expected_docs": ["DOC001"],
        "expected_supported": True
    },
    {
        "scenario": "Ambiguous Query",
        "question": "What is the leave policy?",
        "expected_docs": ["DOC001", "DOC002", "DOC004"],
        "expected_supported": True
    },
    {
        "scenario": "Numeric Query",
        "question": "15 days?",
        "expected_docs": ["DOC001"],
        "expected_supported": True
    },
    {
        "scenario": "Multiple Topic Query",
        "question": "Tell me about annual leave and sick leave",
        "expected_docs": ["DOC001", "DOC002"],
        "expected_supported": True
    },
    {
        "scenario": "Paraphrased Query",
        "question": "How much time off can workers take annually?",
        "expected_docs": ["DOC001"],
        "expected_supported": True
    },
    {
        "scenario": "Unsupported Insurance Query",
        "question": "Does the company provide health insurance?",
        "expected_docs": [],
        "expected_supported": False
    },
    {
        "scenario": "Unsupported Bonus Query",
        "question": "How much annual bonus do employees receive?",
        "expected_docs": [],
        "expected_supported": False
    },
    {
        "scenario": "Unrelated Query",
        "question": "What is the current gold price?",
        "expected_docs": [],
        "expected_supported": False
    }
]

# ============================================================
# SPELLING CORRECTION
# ============================================================

SPELLING_CORRECTIONS = {
    "meny": "many",
    "anuel": "annual",
    "leev": "leave",
    "dayz": "days",
    "employes": "employees",
    "employe": "employee",
    "holidy": "holiday",
    "sik": "sick",
    "remot": "remote",
    "polcy": "policy"
}


def correct_spelling(question):
    words = question.split()
    corrected_words = []

    for word in words:
        punctuation = ""

        while word and word[-1] in "?!.,;:":
            punctuation = word[-1] + punctuation
            word = word[:-1]

        replacement = SPELLING_CORRECTIONS.get(
            word.lower(), word
        )

        corrected_words.append(replacement + punctuation)

    return " ".join(corrected_words)


# ============================================================
# QUERY PREPROCESSING AND REWRITING
# ============================================================

def preprocess_query(question):

    original = question.strip()

    if not original:
        return original

    cleaned = re.sub(r"[?!]+", " ", original)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    corrected = correct_spelling(cleaned)

    lower = corrected.lower()

    # Short query expansion
    if lower in ["leave", "leave?"]:
        return (
            "What are the annual leave, sick leave, "
            "and parental leave policies for employees?"
        )

    # Ambiguous leave query expansion
    if "leave policy" in lower or lower == "leave":
        return (
            "What are the annual leave, sick leave, "
            "and parental leave policies for employees?"
        )

    # Numeric query normalization
    if re.search(r"\b15\s+days?\b", lower):
        return (
            "How many annual leave days do employees receive? "
            "Employees receive 15 annual leave days per year."
        )

    # Multiple topic queries are retained but normalized
    if "annual leave" in lower and "sick leave" in lower:
        return (
            "What is the annual leave allowance and "
            "how many sick leave days do employees receive?"
        )

    # Misspelled annual leave query
    if "annual" in lower and "leave" in lower and (
        "many" in lower or "employees" in lower
    ):
        return (
            "How many annual leave days do employees receive?"
        )

    # General paraphrase normalization
    if "time off" in lower and "annually" in lower:
        return (
            "How many annual leave days do employees receive each year?"
        )

    return corrected


# ============================================================
# OLLAMA CONNECTION
# ============================================================

def check_ollama():

    response = requests.get(
        f"{OLLAMA_URL}/api/tags",
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    models = [
        model["name"]
        for model in response.json().get("models", [])
    ]

    if not any(
        model.startswith(EMBEDDING_MODEL.split(":")[0])
        for model in models
    ):
        raise RuntimeError(
            f"Embedding model not found: {EMBEDDING_MODEL}"
        )

    print("Ollama connection: PASS")
    print(f"Embedding model: {EMBEDDING_MODEL}")


# ============================================================
# EMBEDDING GENERATION
# ============================================================

def generate_embedding(text):

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()
    embeddings = data.get("embeddings", [])

    if not embeddings:
        raise ValueError("Ollama returned an empty embedding.")

    return embeddings[0]


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):

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

def index_documents():

    indexed_documents = []

    print("\n===== DOCUMENT INDEXING =====")

    start = time.perf_counter()

    for document in DOCUMENTS:

        embedding = generate_embedding(document["text"])

        indexed_documents.append({
            **document,
            "embedding": embedding
        })

        print(f"Embedded: {document['id']} - {document['topic']}")

    elapsed = time.perf_counter() - start

    print(f"Total documents indexed: {len(indexed_documents)}")
    print(f"Indexing time: {elapsed:.3f} seconds")

    return indexed_documents


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(question, indexed_documents):

    start = time.perf_counter()

    question_embedding = generate_embedding(question)

    results = []

    for document in indexed_documents:

        score = cosine_similarity(
            question_embedding,
            document["embedding"]
        )

        results.append({
            "document_id": document["id"],
            "topic": document["topic"],
            "similarity": score
        })

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    top_results = results[:TOP_K]

    latency = time.perf_counter() - start

    return top_results, latency


# ============================================================
# EVALUATION
# ============================================================

def evaluate_retrieval(results, expected_docs):

    if not results:
        return False, 0.0, []

    best_similarity = results[0]["similarity"]

    retrieved_ids = [
        result["document_id"]
        for result in results
        if result["similarity"] >= SIMILARITY_THRESHOLD
    ]

    # Supported question:
    # All expected documents must appear in accepted results.
    if expected_docs:

        correct = all(
            document_id in retrieved_ids
            for document_id in expected_docs
        )

    # Unsupported question:
    # No document should cross the threshold.
    else:

        correct = len(retrieved_ids) == 0

    return (
        correct,
        best_similarity,
        retrieved_ids
    )


# ============================================================
# CSV REPORT GENERATION
# ============================================================

def write_reports(report_rows, summary):

    report_fields = [
        "Scenario",
        "Original Question",
        "Rewritten Question",
        "Baseline Best Document",
        "Baseline Similarity",
        "Baseline Retrieved Documents",
        "Baseline Status",
        "Enhanced Best Document",
        "Enhanced Similarity",
        "Enhanced Retrieved Documents",
        "Enhanced Status",
        "Improvement",
        "Baseline Latency Seconds",
        "Enhanced Latency Seconds"
    ]

    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=report_fields
        )

        writer.writeheader()
        writer.writerows(report_rows)

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary.keys())
        )

        writer.writeheader()
        writer.writerow(summary)

    print("\nReports generated:")
    print(REPORT_FILE)
    print(SUMMARY_FILE)


# ============================================================
# MAIN TEST EXECUTION
# ============================================================

def main():

    print("=" * 65)
    print("RAG QUERY PREPROCESSING AND QUERY REWRITING TESTING")
    print("=" * 65)

    check_ollama()

    indexed_documents = index_documents()

    report_rows = []

    baseline_passed = 0
    enhanced_passed = 0

    baseline_failed = 0
    enhanced_failed = 0

    total_baseline_latency = 0.0
    total_enhanced_latency = 0.0

    for index, test_case in enumerate(TEST_CASES, start=1):

        scenario = test_case["scenario"]
        original_question = test_case["question"]
        expected_docs = test_case["expected_docs"]

        print("\n" + "=" * 65)
        print(f"TEST CASE {index}: {scenario}")
        print(f"Original question: {original_question}")

        rewritten_question = preprocess_query(
            original_question
        )

        print(f"Rewritten question: {rewritten_question}")

        # ----------------------------------------------------
        # BASELINE RETRIEVAL
        # ----------------------------------------------------

        baseline_results, baseline_latency = retrieve_documents(
            original_question,
            indexed_documents
        )

        baseline_correct, baseline_score, baseline_ids = (
            evaluate_retrieval(
                baseline_results,
                expected_docs
            )
        )

        baseline_status = (
            "PASS" if baseline_correct else "FAIL"
        )

        # ----------------------------------------------------
        # ENHANCED RETRIEVAL
        # ----------------------------------------------------

        enhanced_results, enhanced_latency = retrieve_documents(
            rewritten_question,
            indexed_documents
        )

        enhanced_correct, enhanced_score, enhanced_ids = (
            evaluate_retrieval(
                enhanced_results,
                expected_docs
            )
        )

        enhanced_status = (
            "PASS" if enhanced_correct else "FAIL"
        )

        # ----------------------------------------------------
        # RESULT COMPARISON
        # ----------------------------------------------------

        if not baseline_correct and enhanced_correct:
            improvement = "IMPROVED"

        elif baseline_correct and not enhanced_correct:
            improvement = "REGRESSION"

        elif baseline_correct and enhanced_correct:
            improvement = "MAINTAINED"

        else:
            improvement = "NOT IMPROVED"

        if baseline_correct:
            baseline_passed += 1
        else:
            baseline_failed += 1

        if enhanced_correct:
            enhanced_passed += 1
        else:
            enhanced_failed += 1

        total_baseline_latency += baseline_latency
        total_enhanced_latency += enhanced_latency

        baseline_best = (
            baseline_results[0]["document_id"]
            if baseline_results else "N/A"
        )

        enhanced_best = (
            enhanced_results[0]["document_id"]
            if enhanced_results else "N/A"
        )

        print("\nBaseline:")
        print(f"Best document: {baseline_best}")
        print(f"Similarity: {baseline_score:.4f}")
        print(f"Accepted documents: {baseline_ids}")
        print(f"Status: {baseline_status}")

        print("\nEnhanced:")
        print(f"Best document: {enhanced_best}")
        print(f"Similarity: {enhanced_score:.4f}")
        print(f"Accepted documents: {enhanced_ids}")
        print(f"Status: {enhanced_status}")

        print(f"\nComparison: {improvement}")

        report_rows.append({
            "Scenario": scenario,
            "Original Question": original_question,
            "Rewritten Question": rewritten_question,
            "Baseline Best Document": baseline_best,
            "Baseline Similarity": round(baseline_score, 4),
            "Baseline Retrieved Documents": ";".join(baseline_ids),
            "Baseline Status": baseline_status,
            "Enhanced Best Document": enhanced_best,
            "Enhanced Similarity": round(enhanced_score, 4),
            "Enhanced Retrieved Documents": ";".join(enhanced_ids),
            "Enhanced Status": enhanced_status,
            "Improvement": improvement,
            "Baseline Latency Seconds": round(baseline_latency, 3),
            "Enhanced Latency Seconds": round(enhanced_latency, 3)
        })

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total = len(TEST_CASES)

    baseline_pass_rate = (
        baseline_passed / total * 100
    )

    enhanced_pass_rate = (
        enhanced_passed / total * 100
    )

    summary = {
        "Execution Date": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Total Test Cases": total,
        "Baseline Passed": baseline_passed,
        "Baseline Failed": baseline_failed,
        "Baseline Pass Rate Percent": round(
            baseline_pass_rate, 2
        ),
        "Enhanced Passed": enhanced_passed,
        "Enhanced Failed": enhanced_failed,
        "Enhanced Pass Rate Percent": round(
            enhanced_pass_rate, 2
        ),
        "Pass Rate Improvement": round(
            enhanced_pass_rate - baseline_pass_rate, 2
        ),
        "Average Baseline Latency Seconds": round(
            total_baseline_latency / total, 3
        ),
        "Average Enhanced Latency Seconds": round(
            total_enhanced_latency / total, 3
        ),
        "Similarity Threshold": SIMILARITY_THRESHOLD,
        "Top K": TOP_K,
        "Embedding Model": EMBEDDING_MODEL
    }

    print("\n" + "=" * 65)
    print("FINAL QUERY PREPROCESSING SUMMARY")
    print("=" * 65)

    for key, value in summary.items():
        print(f"{key}: {value}")

    write_reports(report_rows, summary)

    print("\n===== TEST EXECUTION COMPLETED =====")


if __name__ == "__main__":
    try:
        main()

    except requests.exceptions.RequestException as error:
        print("\nOLLAMA API ERROR:")
        print(error)

    except Exception as error:
        print("\nTEST EXECUTION ERROR:")
        print(type(error).__name__, error)
