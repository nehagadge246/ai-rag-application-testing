
from openai import OpenAI, AsyncOpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import AnswerRelevancy

# Step 1: Synchronous client for the evaluation LLM
llm_client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

# Step 2: Asynchronous client for embeddings
embedding_client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

# Step 3: Configure the evaluation LLM
evaluator_llm = llm_factory(
    model="qwen2.5-coder:3b",
    provider="openai",
    client=llm_client,
    mode=Mode.MD_JSON
)

# Step 4: Configure modern asynchronous embeddings
evaluator_embeddings = OpenAIEmbeddings(
    client=embedding_client,
    model="nomic-embed-text:latest"
)

# Step 5: Initialize Answer Relevancy
answer_relevancy = AnswerRelevancy(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

# Step 6: Define test data
user_input = "How many annual leave days do employees receive?"

response = "Employees receive 20 annual leave days per year."

# Step 7: Evaluate Answer Relevancy
result = answer_relevancy.score(
    user_input=user_input,
    response=response
)

# Step 8: Display results
print("\n===== RAGAS ANSWER RELEVANCY EVALUATION =====")
print("Question:", user_input)
print("Response:", response)
print("Answer Relevancy Score:", result.value)
