
from openai import OpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.metrics.collections import ResponseGroundedness

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

# Initialize Response Groundedness
response_groundedness = ResponseGroundedness(
    llm=evaluator_llm
)

# Test question
user_input = "How many annual leave days do employees receive?"

# Retrieved context
retrieved_contexts = [
    "Employees receive 20 annual leave days per year.",
    "Employees can request leave through the HR portal."
]

# Generated response supported by context
response = "Employees receive 0 annual leave days per year."

# Evaluate response groundedness
result = response_groundedness.score(
    response=response,
    retrieved_contexts=retrieved_contexts
)

print("\n===== RAGAS RESPONSE GROUNDEDNESS EVALUATION =====")
print("Question:", user_input)
print("Generated Answer:", response)

print("\nRetrieved Contexts:")
for context in retrieved_contexts:
    print("-", context)

print("\nResponse Groundedness Score:", result.value)
