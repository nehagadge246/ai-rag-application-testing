
from openai import OpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.metrics.collections import FactualCorrectness

# Step 1: Connect to local Ollama API
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

# Step 2: Configure evaluation LLM
evaluator_llm = llm_factory(
    model="qwen2.5-coder:3b",
    provider="openai",
    client=client,
    mode=Mode.MD_JSON
)

# Step 3: Initialize Factual Correctness
factual_correctness = FactualCorrectness(
    llm=evaluator_llm
)

# Step 4: Define test question
user_input = "How many annual leave days do employees receive?"

# Step 5: Define expected answer
reference = "Employees receive 20 annual leave days per year."

# Step 6: Define AI-generated answer
response = "Employees receive 30 annual leave days per year."

# Step 7: Evaluate factual correctness
result = factual_correctness.score(
    response=response,
    reference=reference
)

# Step 8: Display results
print("\n===== RAGAS FACTUAL CORRECTNESS EVALUATION =====")
print("Question:", user_input)
print("Expected Answer:", reference)
print("Generated Answer:", response)
print("Factual Correctness Score:", result.value)
