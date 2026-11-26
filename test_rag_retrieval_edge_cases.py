
import requests
import csv
import time
from datetime import datetime

# ============================================================
# LESSON 57: RAG RETRIEVAL EDGE CASE TESTING
# ============================================================

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"

TOP_K = 3
SIMILARITY_THRESHOLD = 0.68

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
# 1 = relevant information exists
# 0 = relevant information does not exist
# None = invalid input; retrieval should be skipped

TEST_CASES = [
    {
        "scenario": "Normal Query",
        "question": "How many annual leave days do employees receive?",
        "expected_supported": 1,
        "expected_doc": "DOC001"
    },
    {
        "scenario": "Empty Query",
        "question": "",
        "expected_supported": None,
        "expected_doc": None
    },
    {
        "scenario": "Whitespace Query",
        "question": "     ",
        "expected_supported": None,
        "expected_doc": None
    },
    {
        "scenario": "Very Short Query",
        "question": "Leave?",
        "expected_supported": 1,
        "expected_doc": "DOC001"
    },
    {
        "scenario": "Misspelled Query",
        "question": "How meny anuel leev dayz do employes get?",
        "expected_supported": 1,
        "expected_doc": "DOC001"
    },
    {
        "scenario": "Paraphrased Query",
        "question": "How much time off can staff take each year?",
        "expected_supported": 1,
        "expected_doc": "DOC001"
    },
    {
        "scenario": "Ambiguous Query",
        "question": "What is the leave policy?",
        "expected_supported": 1,
        "expected_doc": None
    },
    {
        "scenario": "Long Query",
        "question": (
            "I am an employee working in the company and I would like "
            "to understand the annual vacation entitlement available "
            "to me during the year, including the total number of "
            "paid leave days that I can request from my manager."
        ),
        "expected_supported": 1,
        "expected_doc": "DOC001"
    },
    {
        "scenario": "Multiple Topic Query",
        "question": "Tell me about annual leave and sick leave.",
        "expected_supported": 1,
        "expected_doc": None
    },
    {
        "scenario": "Unsupported Insurance Query",
        "question": "Which health insurance company covers employees?",
        "expected_supported": 0,
        "expected_doc": None
    },
    {
        "scenario": "Unsupported Bonus Query",
        "question": "How much annual performance bonus is paid?",
        "expected_supported": 0,
        "expected_doc": None
    },
    {
        "scenario": "Unrelated Domain Query",
        "question": "What is the current gold market price?",
        "expected_supported": 0,
        "expected_doc": None
    },
    {
        "scenario": "Special Characters",
        "question": "Annual leave @#$% entitlement ???",
        "expected_supported": 1,
        "expected_doc": "DOC001"
    },
    {
        "scenario": "Numeric Query",
        "question": "15 days?",
        "expected_supported": 1,
        "expected_doc": "DOC001"
    },
    {
        "scenario": "Case Variation",
        "question": "HOW MANY DAYS CAN EMPLOYEES WORK REMOTELY?",
        "expected_supported": 1,
        "expected_doc": "DOC003"
    }
]


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
        raise ValueError("Empty embedding returned by Ollama")

    return embeddings[0]


def cosine_similarity(vector_a, vector_b):

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def main():

    print("=" * 65)
    print("RAG RETRIEVAL EDGE CASE TESTING")
    print("=" * 65)

    # STEP 1: Check Ollama connection

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

    # STEP 2: Generate document embeddings

    print("\nSTEP 2: GENERATING DOCUMENT EMBEDDINGS")

    document_embeddings = {}

    start_time = time.perf_counter()

    try:
        for document in DOCUMENTS:
            document_embeddings[document["id"]] = get_embedding(
                document["text"]
            )

            print(f'{document["id"]}: Embedding generated')

    except Exception as error:
        print(f"Document embedding failed: {error}")
        return

    embedding_time = round(
        time.perf_counter() - start_time, 3
    )

    print(f"Total documents: {len(document_embeddings)}")
    print(f"Embedding time: {embedding_time} seconds")

    # STEP 3: Execute edge case tests

    print("\nSTEP 3: EXECUTING EDGE CASE TESTS")

    report_rows = []

    for index, test_case in enumerate(TEST_CASES, start=1):

        question = test_case["question"]
        scenario = test_case["scenario"]
        expected_supported = test_case["expected_supported"]
        expected_doc = test_case["expected_doc"]

        print(f"\nTEST {index}: {scenario}")
        print(f"Question: {question!r}")

        started = time.perf_counter()

        best_document = ""
        best_similarity = 0.0
        retrieved_documents = ""
        predicted_supported = ""
        retrieval_status = ""
        status = "FAIL"
        error_message = ""

        # Empty and whitespace-only input validation

        if expected_supported is None:

            if not question.strip():
                status = "PASS"
                retrieval_status = "SKIPPED"
                predicted_supported = "N/A"
                error_message = "Invalid input correctly rejected"

            else:
                status = "FAIL"
                retrieval_status = "UNEXPECTED"

        else:

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

                top_results = scores[:TOP_K]

                best_document = top_results[0]["document_id"]
                best_similarity = round(
                    top_results[0]["similarity"], 4
                )

                retrieved_documents = ", ".join(
                    item["document_id"]
                    for item in top_results
                )

                predicted_supported = int(
                    best_similarity >= SIMILARITY_THRESHOLD
                )

                if expected_supported == 1:

                    if predicted_supported == 1:

                        if expected_doc is None:
                            # Ambiguous/multiple-topic query:
                            # check that at least one document
                            # passes the threshold.
                            status = "PASS"
                        elif best_document == expected_doc:
                            status = "PASS"
                        else:
                            status = "REVIEW"

                    else:
                        status = "FAIL"

                else:

                    if predicted_supported == 0:
                        status = "PASS"
                    else:
                        status = "FAIL"

                retrieval_status = (
                    "RETRIEVED"
                    if predicted_supported == 1
                    else "REJECTED"
                )

            except Exception as error:
                status = "FAIL"
                retrieval_status = "ERROR"
                error_message = str(error)

        latency = round(
            time.perf_counter() - started, 3
        )

        print(f"Best document: {best_document or 'N/A'}")
        print(f"Best similarity: {best_similarity}")
        print(f"Retrieved documents: {retrieved_documents or 'N/A'}")
        print(f"Predicted supported: {predicted_supported}")
        print(f"Status: {status}")
        print(f"Latency: {latency} seconds")

        report_rows.append({
            "Scenario": scenario,
            "Question": question,
            "Expected Supported": (
                expected_supported
                if expected_supported is not None
                else "N/A"
            ),
            "Expected Document": expected_doc or "N/A",
            "Best Document": best_document or "N/A",
            "Best Similarity": best_similarity,
            "Retrieved Documents": retrieved_documents or "N/A",
            "Predicted Supported": predicted_supported,
            "Retrieval Status": retrieval_status,
            "Latency Seconds": latency,
            "Status": status,
            "Error": error_message
        })

    # STEP 4: Generate summary

    print("\nSTEP 4: GENERATING SUMMARY")

    total = len(report_rows)

    passed = sum(
        1 for row in report_rows
        if row["Status"] == "PASS"
    )

    failed = sum(
        1 for row in report_rows
        if row["Status"] == "FAIL"
    )

    review = sum(
        1 for row in report_rows
        if row["Status"] == "REVIEW"
    )

    skipped = sum(
        1 for row in report_rows
        if row["Retrieval Status"] == "SKIPPED"
    )

    evaluated = total - skipped

    avg_latency = (
        sum(row["Latency Seconds"] for row in report_rows)
        / total
    )

    pass_rate = (
        passed / total * 100
        if total else 0
    )

    summary = {
        "Execution Date": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Total Test Cases": total,
        "Evaluated Cases": evaluated,
        "Skipped Invalid Inputs": skipped,
        "Passed": passed,
        "Failed": failed,
        "Review": review,
        "Pass Rate Percent": round(pass_rate, 2),
        "Average Latency Seconds": round(avg_latency, 3),
        "Similarity Threshold": SIMILARITY_THRESHOLD,
        "Top K": TOP_K,
        "Embedding Model": EMBEDDING_MODEL
    }

    print(f"Total cases: {total}")
    print(f"Evaluated cases: {evaluated}")
    print(f"Skipped invalid inputs: {skipped}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Review: {review}")
    print(f"Pass rate: {pass_rate:.2f}%")
    print(f"Average latency: {avg_latency:.3f} seconds")

    # STEP 5: Export CSV reports

    print("\nSTEP 5: EXPORTING CSV REPORTS")

    with open(
        "rag_retrieval_edge_cases_report.csv",
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
        "rag_retrieval_edge_cases_summary.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary.keys()
        )

        writer.writeheader()
        writer.writerow(summary)

    print("Detailed report: rag_retrieval_edge_cases_report.csv")
    print("Summary report: rag_retrieval_edge_cases_summary.csv")

    print("\n" + "=" * 65)
    print("RAG RETRIEVAL EDGE CASE TESTING COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
