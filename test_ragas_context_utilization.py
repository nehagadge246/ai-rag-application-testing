
from openai import OpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextUtilization

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

# Initialize Context Utilization metric
context_utilization = ContextUtilization(
    llm=evaluator_llm
)

user_input = "How many annual leave days do employees receive?"

response = "Employees receive 20 annual leave days per year."

retrieved_contexts = [
    "Employees receive 20 annual leave days per year.",
    "Employees receive 10 sick leave days per year.",
    "The company organizes an annual sports competition."
]

# Evaluate context utilization
result = context_utilization.score(
    user_input=user_input,
    response=response,
    retrieved_contexts=retrieved_contexts
)

print("\n===== RAGAS CONTEXT UTILIZATION EVALUATION =====")
print("Question:", user_input)
print("Generated Answer:", response)

print("\nRetrieved Contexts:")
for context in retrieved_contexts:
    print("-", context)

print("\nContext Utilization Score:", result.value)

