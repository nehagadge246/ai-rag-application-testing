
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import SemanticSimilarity

# Connect to local Ollama embeddings
embedding_client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

evaluator_embeddings = OpenAIEmbeddings(
    client=embedding_client,
    model="nomic-embed-text:latest"
)

# Initialize Semantic Similarity metric
semantic_similarity = SemanticSimilarity(
    embeddings=evaluator_embeddings
)

reference = "Employees receive 20 annual leave days per year."

response = (
    "Staff members are entitled to 20 days of paid annual "
    "leave every year."
)

# Evaluate semantic similarity
result = semantic_similarity.score(
    response=response,
    reference=reference
)

print("\n===== RAGAS SEMANTIC SIMILARITY EVALUATION =====")
print("Expected Answer:", reference)
print("Generated Answer:", response)
print("Semantic Similarity Score:", result.value)
