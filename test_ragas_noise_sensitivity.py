
from openai import OpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.metrics.collections import NoiseSensitivity

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

# Initialize Noise Sensitivity metric
noise_sensitivity = NoiseSensitivity(
    llm=evaluator_llm,
    mode="irrelevant"
)

user_input = "How many annual leave days do employees receive?"

reference = "Employees receive 20 annual leave days per year."

response = "Employees receive 20 annual leave days per year."

retrieved_contexts = [
    "Employees receive 20 annual leave days per year.",
    "The company organizes an annual sports competition.",
    "Employees can participate in team-building activities."
]

# Evaluate noise sensitivity
result = noise_sensitivity.score(
    user_input=user_input,
    response=response,
    reference=reference,
    retrieved_contexts=retrieved_contexts
)

print("\n===== RAGAS NOISE SENSITIVITY EVALUATION =====")
print("Question:", user_input)
print("Reference:", reference)
print("Generated Answer:", response)
print("\nRetrieved Contexts:")

for context in retrieved_contexts:
    print("-", context)

print("\nNoise Sensitivity Score:", result.value)
