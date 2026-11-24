
from openai import OpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision

# Step 1: Connect to local Ollama API
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

# Step 2: Configure RAGAs evaluation LLM
evaluator_llm = llm_factory(
    model="qwen2.5-coder:3b",
    provider="openai",
    client=client,
    mode=Mode.MD_JSON
)

# Step 3: Initialize Context Precision
context_precision = ContextPrecision(
    llm=evaluator_llm
)

# Step 4: Define test question
user_input = "How many annual leave days do employees receive?"

# Step 5: Define expected answer
reference = "Employees receive 20 annual leave days per year."

# Step 6: Define retrieved contexts
retrieved_contexts = [
    "Employees receive 20 annual leave days per year.",
    "Employees receive 10 sick leave days per year.",
    "Annual leave requests must be approved by the manager."
]

# Step 7: Evaluate Context Precision
result = context_precision.score(
    user_input=user_input,
    reference=reference,
    retrieved_contexts=retrieved_contexts
)

# Step 8: Display results
print("\n===== RAGAS CONTEXT PRECISION EVALUATION =====")
print("Question:", user_input)
print("Reference Answer:", reference)
print("\nRetrieved Contexts:")

for index, context in enumerate(retrieved_contexts, start=1):
    print(f"{index}. {context}")

print("\nContext Precision Score:", result.value)
