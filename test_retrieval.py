
# Lesson 14: RAG Retrieval Testing

# Knowledge base
documents = [
    "Employees receive 20 annual leave days per year.",
    "Employees receive 10 sick leave days per year.",
    "Employees can work remotely twice a week.",
    "Office parking is available for employees."
]

# User question
question = "How many annual leave days do employees receive?"

# Simple keyword-based retrieval
def retrieve_documents(question, documents, top_k=2):

    question_words = set(
        question.lower().replace("?", "").split()
    )

    results = []

    for document in documents:

        document_words = set(
            document.lower().replace(".", "").split()
        )

        score = len(question_words.intersection(document_words))

        results.append((document, score))

    # Sort by relevance score
    results.sort(key=lambda item: item[1], reverse=True)

    # Return top K documents
    return [
        document
        for document, score in results[:top_k]
        if score > 0
    ]


# Execute retrieval
retrieved_context = retrieve_documents(
    question,
    documents,
    top_k=1
)

# Expected relevant information
expected_keyword = "20 annual leave days"

# Display results
print("\n===== RAG RETRIEVAL TEST =====")

print("\nQuestion:")
print(question)

print("\nRetrieved Documents:")

for index, document in enumerate(retrieved_context, start=1):
    print(f"{index}. {document}")

# Validate retrieval
assert any(
    expected_keyword in document.lower()
    for document in retrieved_context
), "FAIL: Relevant annual leave information was not retrieved!"

print("\nPASS: Relevant information retrieved successfully!")

# Calculate retrieval precision

relevant_documents = [
    document for document in retrieved_context
    if "annual leave" in document.lower()
]

precision = (
    len(relevant_documents) / len(retrieved_context)
    if retrieved_context else 0
)

print(f"\nRetrieval Precision: {precision:.2f}")

assert precision == 1.0, "FAIL: Retrieval precision is below 100%!"

print("PASS: Retrieval precision is 100%!")

# Lesson 16: Retrieval Recall Testing

expected_documents = [
    "Employees receive 20 annual leave days per year.",
    "Employees receive 10 sick leave days per year."
]

question = "What leave benefits do employees receive?"

retrieved_context = retrieve_documents(
    question,
    documents,
    top_k=4
)

relevant_retrieved = [
    doc for doc in expected_documents
    if doc in retrieved_context
]

recall = len(relevant_retrieved) / len(expected_documents)

print("\n===== RETRIEVAL RECALL TEST =====")
print(f"Expected relevant documents: {len(expected_documents)}")
print(f"Retrieved relevant documents: {len(relevant_retrieved)}")
print(f"Retrieval Recall: {recall:.2f}")

assert recall == 1.0, "FAIL: Some relevant documents were not retrieved!"

print("PASS: Retrieval recall is 100%!")
