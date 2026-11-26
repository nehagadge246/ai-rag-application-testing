
import requests
import numpy as np
import pandas as pd
from pathlib import Path

print("===== TOP-K RETRIEVAL AND RANKING TESTING =====")

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text:latest"

documents = [
    "Employees receive 20 annual leave days per year.",
    "Employees receive 10 sick leave days per year.",
    "Employees can work remotely twice a week.",
    "Office parking is available for employees.",
    "Employees receive health insurance benefits."
]

test_cases = [
    {
        "id": 1,
        "question": "How much paid vacation do employees get?",
        "expected_document": documents[0],
        "top_k": 3,
        "expected_relevant": True
    },
    {
        "id": 2,
        "question": "What is the sick leave entitlement?",
        "expected_document": documents[1],
        "top_k": 2,
        "expected_relevant": True
    },
    {
        "id": 3,
        "question": "How often can staff work from home?",
        "expected_document": documents[2],
        "top_k": 3,
        "expected_relevant": True
    },
    {
        "id": 4,
        "question": "What is the employee's annual bonus amount?",
        "expected_document": None,
        "top_k": 3,
        "expected_relevant": False
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


def cosine_similarity(vector_a, vector_b):
    a = np.array(vector_a)
    b = np.array(vector_b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# 1. Generate document embeddings
print("\n===== DOCUMENT EMBEDDING GENERATION =====")

document_embeddings = []

for index, document in enumerate(documents, start=1):
    embedding = generate_embedding(document)
    document_embeddings.append(embedding)

    print(f"Document {index}: Embedding generated")


# 2. Execute top-K retrieval tests
print("\n===== TOP-K RETRIEVAL TESTING =====")

results = []

for test in test_cases:

    question = test["question"]
    expected_document = test["expected_document"]
    top_k = test["top_k"]

    query_embedding = generate_embedding(question)

    scores = [
        cosine_similarity(query_embedding, embedding)
        for embedding in document_embeddings
    ]

    ranked_indices = np.argsort(scores)[::-1]

    retrieved_indices = ranked_indices[:top_k]

    retrieved_documents = [
        documents[index]
        for index in retrieved_indices
    ]

    retrieved_scores = [
        scores[index]
        for index in retrieved_indices
    ]

    if test["expected_relevant"]:
        ranking_status = (
            "PASS"
            if retrieved_documents[0] == expected_document
            else "FAIL"
        )

        relevance_status = (
            "PASS"
            if expected_document in retrieved_documents
            else "FAIL"
        )

    else:
        # This negative test checks whether a known irrelevant
        # document is incorrectly ranked first.
        ranking_status = (
            "PASS"
            if retrieved_documents[0] not in [
                documents[0],
                documents[1],
                documents[2],
                documents[3],
                documents[4]
            ]
            else "REVIEW"
        )

        relevance_status = "REVIEW"

    top_k_status = (
        "PASS"
        if len(retrieved_documents) == top_k
        else "FAIL"
    )

    print(f"\n===== TEST CASE {test['id']} =====")
    print("Question:", question)
    print("Requested Top-K:", top_k)

    for rank, (document, score) in enumerate(
        zip(retrieved_documents, retrieved_scores),
        start=1
    ):
        print(f"\nRank {rank}: {document}")
        print("Similarity:", round(score, 4))

    print("\nTop-K Count:", top_k_status)

    if test["expected_relevant"]:
        print("Expected Document Ranking:", ranking_status)
        print("Expected Document Retrieval:", relevance_status)
    else:
        print("Negative Query: REVIEW - requires relevance threshold calibration")

    results.append({
        "Test Case": test["id"],
        "Question": question,
        "Requested Top-K": top_k,
        "Retrieved Count": len(retrieved_documents),
        "Top Document": retrieved_documents[0],
        "Top Similarity": round(retrieved_scores[0], 4),
        "Ranking Status": ranking_status,
        "Top-K Count Status": top_k_status,
        "Retrieval Status": relevance_status
    })


# 3. Generate report
report = pd.DataFrame(results)

report_path = (
    Path(__file__).parent /
    "top_k_retrieval_report.csv"
)

report.to_csv(report_path, index=False)

print("\n===== FINAL QA SUMMARY =====")
print(report.to_string(index=False))

print("\nTotal Tests:", len(results))
print(
    "Top-K Count Passed:",
    sum(r["Top-K Count Status"] == "PASS" for r in results)
)
print(
    "Ranking Passed:",
    sum(r["Ranking Status"] == "PASS" for r in results)
)
print(
    "Ranking Failed:",
    sum(r["Ranking Status"] == "FAIL" for r in results)
)

print("\nReport saved:", report_path.name)
print("===== TOP-K TESTING COMPLETED =====")
