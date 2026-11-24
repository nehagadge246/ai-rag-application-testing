
from openai import OpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextEntityRecall

# Connect to local Ollama
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

# Configure evaluator LLM
evaluator_llm = llm_factory(
    model="qwen2.5-coder:3b",
    provider="openai",
    client=client,
    mode=Mode.MD_JSON
)

# Initialize Context Entity Recall
context_entity_recall = ContextEntityRecall(
    llm=evaluator_llm
)

# Reference answer
reference = (
    "Employees receive 20 annual leave days per year. "
    "Up to 5 unused days can be carried forward."
)

# Retrieved information is intentionally incomplete
retrieved_contexts = [
    "Employees receive 20 annual leave days per year.",
    "Employees can request leave through the HR portal."
]

# Evaluate context entity recall
result = context_entity_recall.score(
    reference=reference,
    retrieved_contexts=retrieved_contexts
)

print("\n===== RAGAS CONTEXT ENTITY RECALL EVALUATION =====")
print("Reference:", reference)

print("\nRetrieved Contexts:")
for context in retrieved_contexts:
    print("-", context)

print("\nContext Entity Recall Score:", result.value)
