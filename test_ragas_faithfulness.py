
from openai import OpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

# Step 1: Connect to local Ollama API
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

# Step 2: Configure RAGAs evaluation model
evaluator_llm = llm_factory(
    model="qwen2.5-coder:3b",
    provider="openai",
    client=client,
    mode=Mode.MD_JSON
)

# Step 3: Initialize Faithfulness metric
faithfulness = Faithfulness(llm=evaluator_llm)

# Step 4: Define RAG test data
user_input = "How many annual leave days do employees receive?"

response = "Employees receive 20 annual leave days per year."

retrieved_contexts = [
    "Employees receive 20 annual leave days per year."
]

# Step 5: Evaluate faithfulness
result = faithfulness.score(
    user_input=user_input,
    response=response,
    retrieved_contexts=retrieved_contexts
)

# Step 6: Display results
print("\n===== RAGAS FAITHFULNESS EVALUATION =====")
print("Question:", user_input)
print("Response:", response)
print("Retrieved Context:", retrieved_contexts)
print("Faithfulness Score:", result.value)
