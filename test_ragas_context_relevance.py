
from openai import OpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRelevance

# Connect to Ollama
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

# Initialize Context Relevance
context_relevance = ContextRelevance(llm=evaluator_llm)

# Test data
user_input = "How many annual leave days do employees receive?"

retrieved_contexts = [
    "The company organizes annual sports competitions.",
    "Employees participate in football and cricket tournaments.",
    "The company conducts team-building activities every year."
]
# Evaluate retrieved contexts
result = context_relevance.score(
    user_input=user_input,
    retrieved_contexts=retrieved_contexts
)

# Display results
print("\n===== RAGAS CONTEXT RELEVANCE EVALUATION =====")

print("\nUser Question:", user_input)

print("\nRetrieved Contexts:")
for context in retrieved_contexts:
    print("-", context)

print("\nContext Relevance Score:", result.value)
