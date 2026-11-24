
from openai import OpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall

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

# Step 3: Initialize Context Recall
context_recall = ContextRecall(
    llm=evaluator_llm
)

# Step 4: Define test question
user_input = "What are the annual leave entitlement and carry-forward policy?"

# Step 5: Define expected reference answer
reference = (
    "Employees receive 20 annual leave days per year. "
    "Up to 5 unused days can be carried forward."
)

# Step 6: Define retrieved contexts
retrieved_contexts = [
    "Employees receive 20 annual leave days per year."
]

# Step 7: Evaluate Context Recall
result = context_recall.score(
    user_input=user_input,
    reference=reference,
    retrieved_contexts=retrieved_contexts
)

# Step 8: Display results
print("\n===== RAGAS CONTEXT RECALL EVALUATION =====")
print("Question:", user_input)
print("Reference Answer:", reference)
print("\nRetrieved Contexts:")

for index, context in enumerate(retrieved_contexts, start=1):
    print(f"{index}. {context}")

print("\nContext Recall Score:", result.value)
