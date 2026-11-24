
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
    top_k=2
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
