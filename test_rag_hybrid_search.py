
import csv
import math
import re
import time
import requests

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"

TOP_K = 3
VECTOR_THRESHOLD = 0.68
KEYWORD_COVERAGE_THRESHOLD = 0.75
RRF_CONSTANT = 60

REPORT_FILE = "rag_hybrid_search_report.csv"
SUMMARY_FILE = "rag_hybrid_search_summary.csv"

# ============================================================
# DOCUMENT DATASET
# ============================================================

DOCUMENTS = [
    {
        "id": "DOC001",
        "topic": "annual leave",
        "text": (
            "Annual leave policy: Full-time employees receive 15 annual "
            "leave days per year. Employees can request leave through "
            "the HR portal, subject to manager approval."
        )
    },
    {
        "id": "DOC002",
        "topic": "sick leave",
        "text": (
            "Sick leave policy: Employees receive 10 sick leave days "
            "per year. Employees should inform their manager when "
            "they are unable to work due to illness."
        )
    },
    {
        "id": "DOC003",
        "topic": "remote work",
        "text": (
            "Remote work policy: Employees can work remotely up to "
            "2 days per week with manager approval. Employees must "
            "follow company security policies while working remotely."
        )
    },
    {
        "id": "DOC004",
        "topic": "parental leave",
        "text": (
            "Parental leave policy: Eligible employees can receive "
            "12 weeks of parental leave. Employees should contact "
            "HR before starting parental leave."
        )
    },
    {
        "id": "DOC005",
        "topic": "public holidays",
        "text": (
            "Public holiday policy: The company provides 12 public "
            "holidays each year. The annual holiday calendar is "
            "shared with employees by HR."
        )
    },
    {
        "id": "DOC006",
        "topic": "IT security",
        "text": (
            "IT security policy: Employees must use strong passwords "
            "and multi-factor authentication. Passwords must be "
            "changed according to company security requirements."
        )
    },
    {
        "id": "DOC007",
        "topic": "travel reimbursement",
        "text": (
            "Travel reimbursement policy: Employees can claim approved "
            "business travel expenses by submitting receipts and "
            "the required expense report to Finance."
        )
    },
    {
        "id": "DOC008",
        "topic": "performance review",
        "text": (
            "Performance review policy: Employee performance reviews "
            "are conducted twice a year. Managers discuss employee "
            "goals, feedback, and development plans."
        )
    }
]

# ============================================================
# TEST DATASET
# ============================================================

TEST_CASES = [
    {
        "name": "Annual Leave",
        "question": "How many annual leave days do employees receive?",
        "expected": ["DOC001"]
    },
    {
        "name": "Sick Leave",
        "question": "How many sick leave days are available?",
        "expected": ["DOC002"]
    },
    {
        "name": "Remote Work",
        "question": "What is the remote work policy?",
        "expected": ["DOC003"]
    },
    {
        "name": "Parental Leave",
        "question": "How many weeks of parental leave are provided?",
        "expected": ["DOC004"]
    },
    {
        "name": "Public Holidays",
        "question": "How many public holidays are provided annually?",
        "expected": ["DOC005"]
    },
    {
        "name": "IT Security",
        "question": "What are the password and authentication requirements?",
        "expected": ["DOC006"]
    },
    {
        "name": "Travel Reimbursement",
        "question": "How can employees claim business travel expenses?",
        "expected": ["DOC007"]
    },
    {
        "name": "Performance Review",
        "question": "How frequently are employee performance reviews conducted?",
        "expected": ["DOC008"]
    },
    {
        "name": "Unsupported Insurance",
        "question": "Does the company provide health insurance?",
        "expected": []
    },
    {
        "name": "Unsupported Bonus",
        "question": "How much annual bonus do employees receive?",
        "expected": []
    },
    {
        "name": "Unrelated Query",
        "question": "What is the current gold price?",
        "expected": []
    },
    {
        "name": "Exact Number Query",
        "question": "Which policy mentions 15 days?",
        "expected": ["DOC001"]
    }
]

# ============================================================
# TEXT PROCESSING
# ============================================================

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can",
    "do", "does", "for", "from", "how", "i", "in", "is",
    "it", "of", "on", "or", "the", "their", "this", "to",
    "up", "what", "when", "which", "who", "with", "would",
    "many", "much", "per", "year", "annually", "policy",
    "mention", "mentions", "company", "companies",
    "employees", "employee", "current", "receive",
    "receives", "received", "available", "provided",
    "frequently", "often", "much", "does"
}

# Explicit normalization avoids damaging words such as
# bonus and business through indiscriminate suffix removal.
NORMALIZATION = {
    "days": "day",
    "weeks": "week",
    "holidays": "holiday",
    "passwords": "password",
    "requirements": "requirement",
    "provides": "provide",
    "provided": "provide",
    "providing": "provide",
    "receives": "receive",
    "received": "receive",
    "receiving": "receive",
    "conducted": "conduct",
    "conducts": "conduct",
    "conducting": "conduct",
    "reviews": "review",
    "expenses": "expense",
    "claims": "claim",
    "employees": "employee",
    "managers": "manager"
}


def normalize_token(word):
    if word.isdigit():
        return word

    if word in NORMALIZATION:
        return NORMALIZATION[word]

    # Only remove a final s for ordinary plural forms.
    # Preserve words ending in ss, us, or is.
    if (
        word.endswith("s")
        and len(word) > 4
        and not word.endswith(("ss", "us", "is"))
    ):
        return word[:-1]

    return word


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    normalized = []

    for word in words:
        word = normalize_token(word)

        if word not in STOPWORDS:
            normalized.append(word)

    return normalized


# ============================================================
# CORPUS VOCABULARY VALIDATION
# ============================================================

def build_corpus_vocabulary(indexed_documents):
    vocabulary = set()

    for document in indexed_documents:
        vocabulary.update(tokenize(document["text"]))

    return vocabulary


def find_out_of_vocabulary_terms(question, vocabulary):
    query_tokens = set(tokenize(question))

    return sorted(
        token for token in query_tokens
        if not token.isdigit() and token not in vocabulary
    )


# ============================================================
# OLLAMA EMBEDDINGS
# ============================================================

def get_embedding(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    if "embedding" not in data:
        raise ValueError(
            f"Embedding missing from Ollama response: {data}"
        )

    return data["embedding"]


def cosine_similarity(vector_a, vector_b):
    if len(vector_a) != len(vector_b):
        raise ValueError("Embedding dimensions do not match.")

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ============================================================
# VECTOR SEARCH
# ============================================================

def vector_search(query_embedding, indexed_documents):
    start = time.perf_counter()

    scores = []

    for document in indexed_documents:
        score = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        scores.append({
            "id": document["id"],
            "score": score
        })

    ranked = sorted(
        scores,
        key=lambda item: item["score"],
        reverse=True
    )

    latency_ms = (time.perf_counter() - start) * 1000

    return ranked, latency_ms


# ============================================================
# KEYWORD SEARCH
# ============================================================

def keyword_search(question, indexed_documents, vocabulary):
    start = time.perf_counter()

    query_tokens = tokenize(question)

    query_numbers = {
        token for token in query_tokens if token.isdigit()
    }

    out_of_vocabulary = find_out_of_vocabulary_terms(
        question,
        vocabulary
    )

    results = []

    for document in indexed_documents:
        document_tokens = tokenize(document["text"])
        document_token_set = set(document_tokens)

        document_numbers = {
            token for token in document_tokens if token.isdigit()
        }

        numeric_match = query_numbers.issubset(document_numbers)

        matched_tokens = [
            token for token in query_tokens
            if token in document_token_set
        ]

        coverage = (
            len(matched_tokens) / len(query_tokens)
            if query_tokens else 0.0
        )

        if query_numbers and not numeric_match:
            coverage = 0.0

        # Do not accept a document when meaningful query
        # concepts cannot be found anywhere in the corpus.
        if out_of_vocabulary:
            coverage = 0.0

        results.append({
            "id": document["id"],
            "score": coverage,
            "coverage": coverage,
            "numeric_match": numeric_match,
            "out_of_vocabulary": ";".join(out_of_vocabulary)
        })

    ranked = sorted(
        results,
        key=lambda item: item["score"],
        reverse=True
    )

    latency_ms = (time.perf_counter() - start) * 1000

    return ranked, latency_ms


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(vector_ranked, keyword_ranked):
    vector_positions = {
        item["id"]: rank
        for rank, item in enumerate(vector_ranked, start=1)
    }

    keyword_positions = {
        item["id"]: rank
        for rank, item in enumerate(keyword_ranked, start=1)
    }

    fused_results = []

    for document in DOCUMENTS:
        document_id = document["id"]

        vector_rank = vector_positions[document_id]
        keyword_rank = keyword_positions[document_id]

        rrf_score = (
            1 / (RRF_CONSTANT + vector_rank)
            + 1 / (RRF_CONSTANT + keyword_rank)
        )

        fused_results.append({
            "id": document_id,
            "rrf_score": rrf_score,
            "vector_rank": vector_rank,
            "keyword_rank": keyword_rank
        })

    return sorted(
        fused_results,
        key=lambda item: item["rrf_score"],
        reverse=True
    )


# ============================================================
# RETRIEVAL ACCEPTANCE RULES
# ============================================================

def get_accepted_documents(
    method,
    ranked,
    vector_scores,
    keyword_scores,
    out_of_vocabulary
):
    accepted = []

    for item in ranked:
        document_id = item["id"]

        vector_score = vector_scores[document_id]
        keyword_coverage = keyword_scores[document_id]["coverage"]
        numeric_match = keyword_scores[document_id]["numeric_match"]

        if method == "Vector Search":

            is_accepted = vector_score >= VECTOR_THRESHOLD

        elif method == "Keyword Search":

            is_accepted = (
                keyword_coverage >= KEYWORD_COVERAGE_THRESHOLD
                and not out_of_vocabulary
                and numeric_match
            )

        else:

            corpus_supported = not out_of_vocabulary

            is_accepted = (
                corpus_supported
                and numeric_match
                and (
                    vector_score >= VECTOR_THRESHOLD
                    or keyword_coverage >= KEYWORD_COVERAGE_THRESHOLD
                )
            )

        if is_accepted:
            accepted.append(document_id)

    return accepted[:TOP_K]


# ============================================================
# EVALUATION METRICS
# ============================================================

def calculate_metrics(retrieved, expected):
    if not expected:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "top1": 0
        }

    relevant_retrieved = [
        document_id for document_id in retrieved
        if document_id in expected
    ]

    precision = len(relevant_retrieved) / TOP_K
    recall = len(relevant_retrieved) / len(expected)

    mrr = 0.0

    for rank, document_id in enumerate(retrieved, start=1):
        if document_id in expected:
            mrr = 1 / rank
            break

    top1 = int(
        bool(retrieved) and retrieved[0] in expected
    )

    return {
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "top1": top1
    }


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 70)
    print("RAG HYBRID SEARCH TESTING")
    print("Vector Search + Keyword Search + RRF Hybrid Search")
    print("=" * 70)

    print(f"Top K: {TOP_K}")
    print(f"Vector threshold: {VECTOR_THRESHOLD}")
    print(f"Keyword coverage threshold: {KEYWORD_COVERAGE_THRESHOLD}")
    print(f"RRF constant: {RRF_CONSTANT}")

    # --------------------------------------------------------
    # Check Ollama connection
    # --------------------------------------------------------

    try:
        connection = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=10
        )
        connection.raise_for_status()

        print("\nOllama connection: PASS")
        print(f"Embedding model: {EMBEDDING_MODEL}")

    except requests.RequestException as error:
        print("\nOllama connection: FAIL")
        print(error)
        return

    # --------------------------------------------------------
    # Document indexing
    # --------------------------------------------------------

    print("\n===== DOCUMENT INDEXING =====")

    indexed_documents = []
    indexing_start = time.perf_counter()

    try:
        for document in DOCUMENTS:
            embedding = get_embedding(document["text"])

            indexed_documents.append({
                **document,
                "embedding": embedding
            })

            print(
                f"Embedded: {document['id']} - {document['topic']}"
            )

    except Exception as error:
        print(f"Document indexing failed: {error}")
        return

    indexing_time = time.perf_counter() - indexing_start

    vocabulary = build_corpus_vocabulary(indexed_documents)

    print(f"Documents indexed: {len(indexed_documents)}")
    print(f"Corpus vocabulary terms: {len(vocabulary)}")
    print(f"Indexing time: {indexing_time:.3f} seconds")

    report_rows = []

    methods = [
        "Vector Search",
        "Keyword Search",
        "Hybrid Search"
    ]

    # --------------------------------------------------------
    # Run test cases
    # --------------------------------------------------------

    for case_number, test_case in enumerate(TEST_CASES, start=1):

        question = test_case["question"]
        expected = test_case["expected"]

        print("\n" + "=" * 70)
        print(f"TEST CASE {case_number}: {test_case['name']}")
        print(f"Question: {question}")
        print(f"Expected documents: {expected}")

        try:
            query_embedding = get_embedding(question)
        except Exception as error:
            print(f"Query embedding failed: {error}")
            return

        vector_ranked, vector_latency = vector_search(
            query_embedding,
            indexed_documents
        )

        keyword_ranked, keyword_latency = keyword_search(
            question,
            indexed_documents,
            vocabulary
        )

        fusion_start = time.perf_counter()

        hybrid_ranked = reciprocal_rank_fusion(
            vector_ranked,
            keyword_ranked
        )

        fusion_latency = (
            time.perf_counter() - fusion_start
        ) * 1000

        hybrid_latency = (
            vector_latency + keyword_latency + fusion_latency
        )

        vector_scores = {
            item["id"]: item["score"]
            for item in vector_ranked
        }

        keyword_scores = {
            item["id"]: item
            for item in keyword_ranked
        }

        rrf_scores = {
            item["id"]: item["rrf_score"]
            for item in hybrid_ranked
        }

        out_of_vocabulary = find_out_of_vocabulary_terms(
            question,
            vocabulary
        )

        print(f"Meaningful query terms: {tokenize(question)}")
        print(f"Out-of-vocabulary terms: {out_of_vocabulary}")

        method_data = {
            "Vector Search": (vector_ranked, vector_latency),
            "Keyword Search": (keyword_ranked, keyword_latency),
            "Hybrid Search": (hybrid_ranked, hybrid_latency)
        }

        for method in methods:

            ranked, latency_ms = method_data[method]

            accepted = get_accepted_documents(
                method,
                ranked,
                vector_scores,
                keyword_scores,
                out_of_vocabulary
            )

            metrics = calculate_metrics(accepted, expected)

            if expected:
                status = (
                    "PASS"
                    if all(doc in accepted for doc in expected)
                    else "FAIL"
                )
            else:
                status = "PASS" if not accepted else "FAIL"

            best_candidate = ranked[0]["id"]

            best_vector = vector_scores[best_candidate]
            best_keyword = keyword_scores[best_candidate]["coverage"]
            best_rrf = rrf_scores[best_candidate]

            print(f"\n{method}:")
            print(f"Best candidate: {best_candidate}")
            print(f"Vector similarity: {best_vector:.4f}")
            print(f"Keyword coverage: {best_keyword:.4f}")

            if method == "Hybrid Search":
                print(f"RRF score: {best_rrf:.6f}")

            print(f"Accepted documents: {accepted}")
            print(f"Status: {status}")
            print(f"Precision@{TOP_K}: {metrics['precision']:.4f}")
            print(f"Recall@{TOP_K}: {metrics['recall']:.4f}")
            print(f"MRR: {metrics['mrr']:.4f}")
            print(f"Top-1 accuracy: {metrics['top1']}")
            print(f"Retrieval-only latency: {latency_ms:.3f} ms")

            report_rows.append({
                "Test Case": test_case["name"],
                "Question": question,
                "Expected Documents": ";".join(expected),
                "Method": method,
                "Best Candidate": best_candidate,
                "Best Vector Similarity": round(best_vector, 6),
                "Best Keyword Coverage": round(best_keyword, 6),
                "Best RRF Score": round(best_rrf, 8),
                "Out of Vocabulary Terms": ";".join(out_of_vocabulary),
                "Accepted Documents": ";".join(accepted),
                "Precision@3": round(metrics["precision"], 4),
                "Recall@3": round(metrics["recall"], 4),
                "MRR": round(metrics["mrr"], 4),
                "Top1 Accuracy": metrics["top1"],
                "Latency (ms)": round(latency_ms, 3),
                "Status": status
            })

    # --------------------------------------------------------
    # SUMMARY CALCULATIONS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL HYBRID SEARCH SUMMARY")
    print("=" * 70)

    summary_rows = []

    for method in methods:

        method_rows = [
            row for row in report_rows
            if row["Method"] == method
        ]

        supported_rows = [
            row for row in method_rows
            if row["Expected Documents"]
        ]

        unsupported_rows = [
            row for row in method_rows
            if not row["Expected Documents"]
        ]

        passed = sum(
            row["Status"] == "PASS"
            for row in method_rows
        )

        failed = len(method_rows) - passed

        supported_count = len(supported_rows)

        average_precision = (
            sum(row["Precision@3"] for row in supported_rows)
            / supported_count
        )

        average_recall = (
            sum(row["Recall@3"] for row in supported_rows)
            / supported_count
        )

        average_mrr = (
            sum(row["MRR"] for row in supported_rows)
            / supported_count
        )

        top1_accuracy = (
            sum(row["Top1 Accuracy"] for row in supported_rows)
            / supported_count
        )

        rejection_rate = (
            sum(
                row["Status"] == "PASS"
                for row in unsupported_rows
            )
            / len(unsupported_rows)
            if unsupported_rows else 0.0
        )

        average_latency = (
            sum(row["Latency (ms)"] for row in method_rows)
            / len(method_rows)
        )

        pass_rate = passed / len(method_rows) * 100

        print(f"\n{method}:")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Overall pass rate: {pass_rate:.2f}%")
        print(f"Supported average Precision@3: {average_precision:.4f}")
        print(f"Supported average Recall@3: {average_recall:.4f}")
        print(f"Supported average MRR: {average_mrr:.4f}")
        print(f"Supported Top-1 accuracy: {top1_accuracy:.2%}")
        print(f"Unsupported-query rejection rate: {rejection_rate:.2%}")
        print(f"Average retrieval-only latency: {average_latency:.3f} ms")

        summary_rows.append({
            "Method": method,
            "Total Tests": len(method_rows),
            "Passed": passed,
            "Failed": failed,
            "Overall Pass Rate (%)": round(pass_rate, 2),
            "Supported Tests": supported_count,
            "Unsupported Tests": len(unsupported_rows),
            "Supported Precision@3": round(average_precision, 4),
            "Supported Recall@3": round(average_recall, 4),
            "Supported MRR": round(average_mrr, 4),
            "Supported Top1 Accuracy (%)": round(
                top1_accuracy * 100, 2
            ),
            "Unsupported Rejection Rate (%)": round(
                rejection_rate * 100, 2
            ),
            "Average Retrieval-only Latency (ms)": round(
                average_latency, 3
            )
        })

    # --------------------------------------------------------
    # EXPORT CSV REPORTS
    # --------------------------------------------------------

    if report_rows:
        with open(
            REPORT_FILE,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=report_rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(report_rows)

    if summary_rows:
        with open(
            SUMMARY_FILE,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=summary_rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    print("\nReports generated:")
    print(REPORT_FILE)
    print(SUMMARY_FILE)

    print("\nIMPORTANT EVALUATION NOTES:")
    print(
        "- Vector-only search uses cosine similarity and its own threshold."
    )
    print(
        "- Keyword search uses normalized token coverage and exact "
        "numeric matching."
    )
    print(
        "- Keyword and hybrid search reject queries containing "
        "meaningful terms absent from this corpus."
    )
    print(
        "- Hybrid ranking uses Reciprocal Rank Fusion (RRF)."
    )
    print(
        "- RRF scores are ranking signals, not confidence probabilities."
    )
    print(
        "- Latency measures retrieval and ranking only; query embedding "
        "generation and document indexing are excluded."
    )
    print(
        "- Vocabulary validation is a dataset-specific safeguard, "
        "not a complete semantic unsupported-query detector."
    )
    print(
        "- Thresholds are initial experimental values, not "
        "production-calibrated values."
    )

    print("\n===== HYBRID SEARCH TESTING COMPLETED =====")


if __name__ == "__main__":
    main()
