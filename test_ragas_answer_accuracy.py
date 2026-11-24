
from openai import OpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerAccuracy

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

# Initialize Answer Accuracy metric
answer_accuracy = AnswerAccuracy(llm=evaluator_llm)

# Test data
user_input = "How many annual leave days do employees receive?"

reference = "Employees receive 20 annual leave days per year."

response = "Employees receive 30 annual leave days per year."

# Evaluate answer accuracy
result = answer_accuracy.score(
    user_input=user_input,
    response=response,
    reference=reference
)

# Display results
print("\n===== RAGAS ANSWER ACCURACY EVALUATION =====")

print("\nQuestion:", user_input)
print("Generated Answer:", response)
print("Expected Answer:", reference)

print("\nAnswer Accuracy Score:", result.value)
