
from openai import OpenAI, AsyncOpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import AnswerCorrectness

# Configure Ollama LLM
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

evaluator_llm = llm_factory(
    model="qwen2.5-coder:3b",
    provider="openai",
    client=client,
    mode=Mode.MD_JSON
)

# Configure Ollama embeddings
embedding_client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

evaluator_embeddings = OpenAIEmbeddings(
    client=embedding_client,
    model="nomic-embed-text:latest"
)

# Initialize Answer Correctness
answer_correctness = AnswerCorrectness(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

# Test data
user_input = "How many annual leave days do employees receive?"

reference = "Employees receive 20 annual leave days per year."

response = "Employees receive 20 annual leave days per year."

# Evaluate
result = answer_correctness.score(
    user_input=user_input,
    response=response,
    reference=reference
)

print("\n===== RAGAS ANSWER CORRECTNESS EVALUATION =====")
print("Question:", user_input)
print("Generated Answer:", response)
print("Expected Answer:", reference)
print("\nAnswer Correctness Score:", result.value)
