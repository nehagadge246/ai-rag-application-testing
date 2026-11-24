
from openai import OpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.metrics.collections import SummaryScore

# Configure Ollama LLM
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

evaluator_llm = llm_factory(
    model="qwen2.5-coder:3b",
    provider="openai",
    client=client,
    mode=Mode.MD_JSON
)

# Initialize Summary Score metric
summary_score = SummaryScore(
    llm=evaluator_llm,
    length_penalty=False
)

# Original source information
reference_contexts = [
    "Employees receive 20 annual leave days per year.",
    "Employees can carry forward up to 5 unused annual leave days.",
    "Annual leave requests must be approved by the employee's manager."
]

# Positive summary: Covers all important information
positive_summary = (
    "Employees receive 20 annual leave days per year, "
    "can carry forward up to 5 unused days, and need "
    "manager approval for leave requests."
)

# Negative summary: Misses important information
negative_summary = (
    "Employees receive 20 annual leave days per year."
)

print("===== RAGAS SUMMARY SCORE EVALUATION =====")

# Positive evaluation
print("\n===== POSITIVE TEST =====")

positive_result = summary_score.score(
    reference_contexts=reference_contexts,
    response=positive_summary
)

print("Positive Summary Score:", positive_result.value)

# Negative evaluation
print("\n===== NEGATIVE TEST =====")

negative_result = summary_score.score(
    reference_contexts=reference_contexts,
    response=negative_summary
)

print("Negative Summary Score:", negative_result.value)
