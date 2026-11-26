
import requests
import numpy as np
import pandas as pd
from pathlib import Path

print("===== LOCAL EMBEDDING AND SEMANTIC RETRIEVAL TESTING =====")

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text:latest"

documents = [
    "Employees receive 20 annual leave days per year.",
    "Employees receive 10 sick leave days per year.",
    "Employees can work remotely twice a week.",
    "Office parking is available for employees."
]

test_cases = [
    {
        "question": "How much paid vacation do employees get?",
        "expected_document": documents[0]
    },
    {
        "question": "What is the sick leave entitlement?",
        "expected_document": documents[1]
    },
    {
        "question": "How often can staff work from home?",
        "expected_document": documents[2]
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

    print(
        f"Document {index}: "
        f"Embedding generated ({len(embedding)} dimensions)"
    )

# 2. Test semantic retrieval
print("\n===== SEMANTIC RETRIEVAL TESTING =====")

results = []

for index, test in enumerate(test_cases, start=1):

    question = test["question"]
    expected_document = test["expected_document"]

    query_embedding = generate_embedding(question)

    similarities = [
        cosine_similarity(query_embedding, embedding)
        for embedding in document_embeddings
    ]

    best_index = int(np.argmax(similarities))
    retrieved_document = documents[best_index]
    best_score = similarities[best_index]

    status = (
        "PASS"
        if retrieved_document == expected_document
        else "FAIL"
    )

    print(f"\nTest Case {index}")
    print("Question:", question)
    print("Retrieved Document:", retrieved_document)
    print("Expected Document:", expected_document)
    print("Similarity Score:", round(best_score, 4))
    print("Status:", status)

    results.append({
        "Test Case": index,
        "Question": question,
        "Retrieved Document": retrieved_document,
        "Expected Document": expected_document,
        "Similarity Score": round(best_score, 4),
        "Status": status
    })

# 3. Generate QA report
report = pd.DataFrame(results)

report_path = (
    Path(__file__).parent /
    "embedding_semantic_retrieval_report.csv"
)

report.to_csv(report_path, index=False)

print("\n===== FINAL QA SUMMARY =====")
print(report[["Test Case", "Similarity Score", "Status"]].to_string(index=False))

print("\nTotal Tests:", len(results))
print("Passed:", sum(r["Status"] == "PASS" for r in results))
print("Failed:", sum(r["Status"] == "FAIL" for r in results))

print("\nReport saved:", report_path.name)
print("===== EMBEDDING TESTING COMPLETED =====")
