
import requests
import numpy as np
import pandas as pd
import time

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("===== INTEGRATED END-TO-END RAG TESTING =====")

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "qwen2.5-coder:3b"

SIMILARITY_THRESHOLD = 0.69
TOP_K = 2

# --------------------------------------------------
# 2. KNOWLEDGE BASE
# --------------------------------------------------

documents = [
    """
    HR EMPLOYEE POLICY

    Annual Leave:
    Employees receive 20 annual leave days per year.
    Employees can carry forward up to 5 unused leave days.

    Sick Leave:
    Employees receive 10 sick leave days per year.

    Remote Work:
    Employees can work remotely twice a week.

    Office Facilities:
    Office parking is available for employees.
    """
]

# --------------------------------------------------
# 3. TEST CASES
# Each group contains equivalent acceptable phrases.
# Every group must have at least one matching phrase.
# --------------------------------------------------

test_cases = [
    {
        "id": 1,
        "question": "How much paid vacation do employees get?",
        "expected_keyword_groups": [
            ["20"],
            ["annual leave", "paid vacation", "vacation"]
        ],
        "expected_answer": True
    },
    {
        "id": 2,
        "question": "What is the sick leave entitlement?",
        "expected_keyword_groups": [
            ["10"],
            ["sick leave"]
        ],
        "expected_answer": True
    },
    {
        "id": 3,
        "question": "How often can employees work from home?",
        "expected_keyword_groups": [
            ["twice"],
            ["week"]
        ],
        "expected_answer": True
    },
    {
        "id": 4,
        "question": "What is the employee annual bonus?",
        "expected_keyword_groups": [],
        "expected_answer": False
    }
]

# --------------------------------------------------
# 4. EMBEDDING GENERATION
# --------------------------------------------------

def generate_embedding(text):

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": text
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["embedding"]


# --------------------------------------------------
# 5. COSINE SIMILARITY
# --------------------------------------------------

def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# --------------------------------------------------
# 6. LLM ANSWER GENERATION
# --------------------------------------------------

def generate_answer(question, context):

    prompt = f"""
Answer the question using ONLY the supplied context.

Do not use outside knowledge.
Do not invent information.

If the answer is unavailable, clearly say:
"No relevant information found."

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"].strip()


# --------------------------------------------------
# 7. DOCUMENT INGESTION
# --------------------------------------------------

print("\n===== STEP 1: DOCUMENT INGESTION =====")

all_text = "\n".join(documents)

ingestion_status = (
    "PASS" if all_text.strip() else "FAIL"
)

print("Documents loaded:", len(documents))
print("Total characters:", len(all_text))
print("Ingestion Status:", ingestion_status)

if ingestion_status == "FAIL":
    raise RuntimeError("Document ingestion failed.")


# --------------------------------------------------
# 8. TEXT CHUNKING
# --------------------------------------------------

print("\n===== STEP 2: TEXT CHUNKING =====")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30
)

chunks = splitter.split_text(all_text)

print("Total chunks generated:", len(chunks))

for index, chunk in enumerate(chunks, start=1):

    print(f"\nChunk {index}:")
    print(chunk)

if not chunks:
    raise RuntimeError("No document chunks generated.")


# --------------------------------------------------
# 9. CHUNK EMBEDDINGS
# --------------------------------------------------

print("\n===== STEP 3: EMBEDDING GENERATION =====")

chunk_embeddings = []

for index, chunk in enumerate(chunks, start=1):

    embedding = generate_embedding(chunk)

    chunk_embeddings.append(embedding)

    print(
        f"Chunk {index}: "
        f"{len(embedding)} embedding dimensions generated"
    )


# --------------------------------------------------
# 10. RETRIEVAL AND ANSWER GENERATION
# --------------------------------------------------

print("\n===== STEP 4: RETRIEVAL AND ANSWER GENERATION =====")

results = []

for test in test_cases:

    print("\n" + "=" * 55)

    print("TEST CASE:", test["id"])
    print("Question:", test["question"])

    start_time = time.perf_counter()

    # Generate question embedding
    query_embedding = generate_embedding(test["question"])

    # Calculate similarity with every chunk
    similarities = [
        cosine_similarity(query_embedding, embedding)
        for embedding in chunk_embeddings
    ]

    # Rank chunks by similarity
    ranked_indices = np.argsort(similarities)[::-1][:TOP_K]

    # Apply similarity threshold
    retrieved_chunks = [
        chunks[index]
        for index in ranked_indices
        if similarities[index] >= SIMILARITY_THRESHOLD
    ]

    top_score = float(max(similarities))

    retrieval_status = (
        "FOUND" if retrieved_chunks else "NOT_FOUND"
    )

    # Prepare context
    context = "\n".join(retrieved_chunks)

    # Generate answer only when relevant context exists
    if retrieved_chunks:

        answer = generate_answer(
            test["question"],
            context
        )

    else:

        answer = "No relevant information found."

    answer_lower = answer.lower()

    # --------------------------------------------------
    # 11. ANSWER QUALITY VALIDATION
    # --------------------------------------------------

    if test["expected_answer"]:

        matched_keywords = []

        for group in test["expected_keyword_groups"]:

            matched = next(
                (
                    keyword
                    for keyword in group
                    if keyword.lower() in answer_lower
                ),
                None
            )

            if matched:
                matched_keywords.append(matched)

        total_groups = len(test["expected_keyword_groups"])

        keyword_coverage = (
            len(matched_keywords) / total_groups
            if total_groups else 0
        )

        answer_status = (
            "PASS"
            if keyword_coverage == 1.0
            else "FAIL"
        )

        overall_status = (
            "PASS"
            if retrieval_status == "FOUND"
            and answer_status == "PASS"
            else "FAIL"
        )

    else:

        matched_keywords = []

        keyword_coverage = (
            1.0
            if retrieval_status == "NOT_FOUND"
            and "no relevant information found" in answer_lower
            else 0.0
        )

        answer_status = (
            "PASS" if keyword_coverage == 1.0 else "FAIL"
        )

        overall_status = answer_status

    # --------------------------------------------------
    # 12. DISPLAY RESULTS
    # --------------------------------------------------

    latency = round(
        time.perf_counter() - start_time,
        2
    )

    print("\nRetrieved Context:")

    print(
        context
        if context
        else "No relevant context retrieved."
    )

    print("\nSimilarity Score:", round(top_score, 4))
    print("Similarity Threshold:", SIMILARITY_THRESHOLD)
    print("Retrieval Status:", retrieval_status)
    print("Generated Answer:", answer)

    print("Matched Keywords:", matched_keywords)

    print(
        "Keyword Coverage:",
        f"{keyword_coverage:.2%}"
    )

    print("Answer Status:", answer_status)
    print("Response Time:", f"{latency}s")
    print("Overall Result:", overall_status)

    results.append({
        "Test Case": test["id"],
        "Question": test["question"],
        "Retrieved Context": context,
        "Similarity Score": round(top_score, 4),
        "Similarity Threshold": SIMILARITY_THRESHOLD,
        "Retrieval Status": retrieval_status,
        "Generated Answer": answer,
        "Matched Keywords": ", ".join(matched_keywords),
        "Keyword Coverage": round(keyword_coverage, 2),
        "Answer Status": answer_status,
        "Latency Seconds": latency,
        "Overall Status": overall_status
    })


# --------------------------------------------------
# 13. CSV REPORT
# --------------------------------------------------

report = pd.DataFrame(results)

report_path = (
    Path(__file__).parent /
    "integrated_rag_pipeline_report.csv"
)

report.to_csv(
    report_path,
    index=False
)


# --------------------------------------------------
# 14. FINAL QA SUMMARY
# --------------------------------------------------

print("\n===== FINAL INTEGRATED RAG QA SUMMARY =====")

print(
    report[
        [
            "Test Case",
            "Retrieval Status",
            "Answer Status",
            "Overall Status"
        ]
    ].to_string(index=False)
)

passed = sum(
    result["Overall Status"] == "PASS"
    for result in results
)

failed = len(results) - passed

print("\nDocument Ingestion:", ingestion_status)

print("Total Test Cases:", len(results))
print("Passed:", passed)
print("Failed:", failed)

print(
    "Success Rate:",
    f"{(passed / len(results)) * 100:.2f}%"
)

print("\nReport saved:", report_path.name)

print("\n===== INTEGRATED RAG TESTING COMPLETED =====")
