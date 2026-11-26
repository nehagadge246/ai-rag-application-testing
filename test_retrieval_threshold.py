
import requests
import numpy as np
import pandas as pd
from pathlib import Path

print("===== RETRIEVAL THRESHOLD AND NO-ANSWER TESTING =====")

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text:latest"

SIMILARITY_THRESHOLD = 0.68

documents = [
    "Employees receive 20 annual leave days per year.",
    "Employees receive 10 sick leave days per year.",
    "Employees can work remotely twice a week.",
    "Office parking is available for employees.",
    "Employees receive health insurance benefits."
]

test_cases = [
    {
        "question": "How much paid vacation do employees get?",
        "expected_document": documents[0],
        "expected_status": "FOUND"
    },
    {
        "question": "What is the sick leave entitlement?",
        "expected_document": documents[1],
        "expected_status": "FOUND"
    },
    {
        "question": "How often can staff work from home?",
        "expected_document": documents[2],
        "expected_status": "FOUND"
    },
    {
        "question": "What is the employee's annual bonus amount?",
        "expected_document": None,
        "expected_status": "NOT_FOUND"
    }
]


def generate_embedding(text):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": text
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embedding"]


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


print("\n===== GENERATING DOCUMENT EMBEDDINGS =====")

document_embeddings = [
    generate_embedding(document)
    for document in documents
]

results = []

print("\n===== THRESHOLD VALIDATION =====")

for index, test in enumerate(test_cases, start=1):

    question = test["question"]

    query_embedding = generate_embedding(question)

    scores = [
        cosine_similarity(query_embedding, embedding)
        for embedding in document_embeddings
    ]

    best_index = int(np.argmax(scores))
    best_score = scores[best_index]

    retrieved_document = (
        documents[best_index]
        if best_score >= SIMILARITY_THRESHOLD
        else "No relevant information found."
    )

    actual_status = (
        "FOUND"
        if best_score >= SIMILARITY_THRESHOLD
        else "NOT_FOUND"
    )

    status = (
        "PASS"
        if actual_status == test["expected_status"]
        and (
            test["expected_document"] is None
            or documents[best_index] == test["expected_document"]
        )
        else "FAIL"
    )

    print(f"\nTest Case {index}")
    print("Question:", question)
    print("Best Similarity:", round(best_score, 4))
    print("Threshold:", SIMILARITY_THRESHOLD)
    print("Retrieved Result:", retrieved_document)
    print("Expected Status:", test["expected_status"])
    print("Actual Status:", actual_status)
    print("Test Result:", status)

    results.append({
        "Test Case": index,
        "Question": question,
        "Best Similarity": round(best_score, 4),
        "Threshold": SIMILARITY_THRESHOLD,
        "Expected Status": test["expected_status"],
        "Actual Status": actual_status,
        "Retrieved Document": retrieved_document,
        "Test Result": status
    })


report = pd.DataFrame(results)

report_path = (
    Path(__file__).parent /
    "retrieval_threshold_report.csv"
)

report.to_csv(report_path, index=False)

print("\n===== FINAL QA SUMMARY =====")
print(report[["Test Case", "Best Similarity", "Expected Status", "Actual Status", "Test Result"]].to_string(index=False))

print("\nTotal Tests:", len(results))
print("Passed:", sum(r["Test Result"] == "PASS" for r in results))
print("Failed:", sum(r["Test Result"] == "FAIL" for r in results))

print("\nReport saved:", report_path.name)
print("===== THRESHOLD TESTING COMPLETED =====")
