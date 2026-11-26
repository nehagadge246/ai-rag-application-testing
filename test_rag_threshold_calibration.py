
import requests
import csv
import time
from datetime import datetime

# ============================================================
# RAG RETRIEVAL THRESHOLD CALIBRATION
# ============================================================

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"

THRESHOLDS = [0.40, 0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.75, 0.80]

DOCUMENTS = [
    {
        "id": "DOC001",
        "text": "Employees receive 15 annual leave days every year.",
        "topic": "annual leave"
    },
    {
        "id": "DOC002",
        "text": "Employees receive 10 sick leave days annually.",
        "topic": "sick leave"
    },
    {
        "id": "DOC003",
        "text": "Employees can work remotely two days per week.",
        "topic": "remote work"
    },
    {
        "id": "DOC004",
        "text": "Employees are entitled to 12 weeks of parental leave.",
        "topic": "parental leave"
    },
    {
        "id": "DOC005",
        "text": "The company provides 10 public holidays every year.",
        "topic": "public holidays"
    },
    {
        "id": "DOC006",
        "text": "Employees must use multi-factor authentication for company systems.",
        "topic": "IT security"
    },
    {
        "id": "DOC007",
        "text": "Employees can claim approved business travel expenses.",
        "topic": "travel reimbursement"
    },
    {
        "id": "DOC008",
        "text": "Employee performance reviews are conducted twice a year.",
        "topic": "performance review"
    }
]

# expected_supported:
# 1 = relevant information exists in the document collection
# 0 = relevant information does not exist

TEST_CASES = [
    {
        "question": "How many annual leave days do employees receive?",
        "expected_supported": 1,
        "expected_doc": "DOC001"
    },
    {
        "question": "What is the yearly sick leave entitlement?",
        "expected_supported": 1,
        "expected_doc": "DOC002"
    },
    {
        "question": "How many days can employees work remotely?",
        "expected_supported": 1,
        "expected_doc": "DOC003"
    },
    {
        "question": "How long is parental leave?",
        "expected_supported": 1,
        "expected_doc": "DOC004"
    },
    {
        "question": "How many public holidays are provided?",
        "expected_supported": 1,
        "expected_doc": "DOC005"
    },
    {
        "question": "What security authentication is required?",
        "expected_supported": 1,
        "expected_doc": "DOC006"
    },
    {
        "question": "Can employees claim business travel expenses?",
        "expected_supported": 1,
        "expected_doc": "DOC007"
    },
    {
        "question": "How often are performance reviews conducted?",
        "expected_supported": 1,
        "expected_doc": "DOC008"
    },
    {
        "question": "What is the employee health insurance provider?",
        "expected_supported": 0,
        "expected_doc": None
    },
    {
        "question": "What is the company's annual employee bonus amount?",
        "expected_supported": 0,
        "expected_doc": None
    },
    {
        "question": "What is the current gold stock market price?",
        "expected_supported": 0,
        "expected_doc": None
    },
    {
        "question": "What is the company's monthly salary structure?",
        "expected_supported": 0,
        "expected_doc": None
    }
]


def get_embedding(text):
    """Generate an embedding using Ollama."""

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
        raise ValueError("Ollama returned an empty embedding.")

    return embeddings[0]


def cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity."""

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def calculate_metrics(tp, tn, fp, fn):
    """Calculate classification metrics."""

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0

    precision = tp / (tp + fp) if (tp + fp) else 0

    recall = tp / (tp + fn) if (tp + fn) else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }


def main():

    print("=" * 65)
    print("RAG RETRIEVAL THRESHOLD CALIBRATION")
    print("=" * 65)

    # --------------------------------------------------------
    # STEP 1: Check Ollama connection
    # --------------------------------------------------------

    print("\nSTEP 1: CHECKING OLLAMA CONNECTION")

    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=10
        )

        response.raise_for_status()

        print("Ollama connection: PASS")

    except requests.RequestException as error:
        print(f"Ollama connection: FAIL - {error}")
        return

    # --------------------------------------------------------
    # STEP 2: Generate document embeddings
    # --------------------------------------------------------

    print("\nSTEP 2: GENERATING DOCUMENT EMBEDDINGS")

    document_embeddings = {}

    start_time = time.perf_counter()

    try:
        for document in DOCUMENTS:
            embedding = get_embedding(document["text"])

            document_embeddings[document["id"]] = embedding

            print(f'{document["id"]}: Embedding generated')

    except Exception as error:
        print(f"Embedding generation failed: {error}")
        return

    embedding_time = round(
        time.perf_counter() - start_time, 3
    )

    print(f"Documents embedded: {len(document_embeddings)}")
    print(f"Embedding time: {embedding_time} seconds")

    # --------------------------------------------------------
    # STEP 3: Retrieve documents for every question
    # --------------------------------------------------------

    print("\nSTEP 3: RETRIEVING DOCUMENTS")

    retrieval_results = []

    for index, test_case in enumerate(TEST_CASES, start=1):

        question = test_case["question"]

        print(f"\nQuestion {index}: {question}")

        try:
            question_embedding = get_embedding(question)

            scores = []

            for document in DOCUMENTS:

                similarity = cosine_similarity(
                    question_embedding,
                    document_embeddings[document["id"]]
                )

                scores.append({
                    "document_id": document["id"],
                    "similarity": similarity
                })

            scores.sort(
                key=lambda item: item["similarity"],
                reverse=True
            )

            best_match = scores[0]

            expected_doc = test_case["expected_doc"]

            # Relevant document must rank first.
            # Unsupported queries must not retrieve a relevant document.
            if test_case["expected_supported"] == 1:
                retrieval_correct = (
                    best_match["document_id"] == expected_doc
                )
            else:
                retrieval_correct = True

            print(
                f'Best match: {best_match["document_id"]} '
                f'| Similarity: {best_match["similarity"]:.4f}'
            )

            print(
                f"Expected supported: "
                f'{test_case["expected_supported"]}'
            )

            retrieval_results.append({
                "question": question,
                "expected_supported": test_case["expected_supported"],
                "expected_doc": expected_doc,
                "best_document": best_match["document_id"],
                "best_similarity": best_match["similarity"],
                "retrieval_correct": retrieval_correct
            })

        except Exception as error:
            print(f"Question evaluation failed: {error}")
            return

    # --------------------------------------------------------
    # STEP 4: Evaluate multiple thresholds
    # --------------------------------------------------------

    print("\nSTEP 4: THRESHOLD EVALUATION")

    report_rows = []
    summary_rows = []

    for threshold in THRESHOLDS:

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        correct = 0

        for result in retrieval_results:

            predicted_supported = (
                result["best_similarity"] >= threshold
            )

            actual_supported = result["expected_supported"] == 1

            if predicted_supported and actual_supported:
                tp += 1

            elif not predicted_supported and not actual_supported:
                tn += 1

            elif predicted_supported and not actual_supported:
                fp += 1

            elif not predicted_supported and actual_supported:
                fn += 1

            # A supported query is correctly classified only when
            # its expected document is also the top-ranked document.
            if actual_supported:
                case_correct = (
                    predicted_supported
                    and result["retrieval_correct"]
                )
            else:
                case_correct = not predicted_supported

            if case_correct:
                correct += 1

            report_rows.append({
                "Question": result["question"],
                "Expected Supported": int(actual_supported),
                "Best Document": result["best_document"],
                "Best Similarity": round(
                    result["best_similarity"], 4
                ),
                "Threshold": threshold,
                "Predicted Supported": int(predicted_supported),
                "Retrieval Correct": result["retrieval_correct"],
                "Classification Status": (
                    "PASS"
                    if predicted_supported == actual_supported
                    else "FAIL"
                )
            })

        metrics = calculate_metrics(tp, tn, fp, fn)

        summary_rows.append({
            "Threshold": threshold,
            "True Positive": tp,
            "True Negative": tn,
            "False Positive": fp,
            "False Negative": fn,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1_score"],
            "Retrieval Aware Correct": correct,
            "Total Cases": len(TEST_CASES)
        })

        print(
            f"\nThreshold: {threshold:.2f}"
            f"\nTP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}"
            f"\nAccuracy: {metrics['accuracy']:.4f}"
            f"\nPrecision: {metrics['precision']:.4f}"
            f"\nRecall: {metrics['recall']:.4f}"
            f"\nF1 Score: {metrics['f1_score']:.4f}"
        )

    # --------------------------------------------------------
    # STEP 5: Identify best threshold
    # --------------------------------------------------------

    best_result = max(
        summary_rows,
        key=lambda row: (
            row["F1 Score"],
            row["Accuracy"],
            row["Precision"]
        )
    )

    print("\nSTEP 5: BEST THRESHOLD")

    print(f'Best threshold: {best_result["Threshold"]:.2f}')
    print(f'Best F1 Score: {best_result["F1 Score"]:.4f}')
    print(f'Best Accuracy: {best_result["Accuracy"]:.4f}')

    # --------------------------------------------------------
    # STEP 6: Export CSV reports
    # --------------------------------------------------------

    with open(
        "rag_threshold_calibration_report.csv",
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
        "rag_threshold_calibration_summary.csv",
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

    print("\nSTEP 6: REPORTS GENERATED")

    print("1. rag_threshold_calibration_report.csv")
    print("2. rag_threshold_calibration_summary.csv")

    print("\n" + "=" * 65)
    print("RAG THRESHOLD CALIBRATION COMPLETED")
    print("=" * 65)

    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"Thresholds evaluated: {len(THRESHOLDS)}")
    print(f"Best threshold: {best_result['Threshold']:.2f}")


if __name__ == "__main__":
    main()
