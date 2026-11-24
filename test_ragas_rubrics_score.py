
from openai import OpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.metrics.collections import RubricsScoreWithReference

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

# Define custom scoring rubrics
custom_rubrics = {
    "score1_description": (
        "The answer is completely incorrect, irrelevant, "
        "or contradicts the reference information."
    ),
    "score2_description": (
        "The answer contains major factual errors and "
        "provides very little correct information."
    ),
    "score3_description": (
        "The answer is partially correct but misses "
        "important information from the reference."
    ),
    "score4_description": (
        "The answer is mostly correct and relevant, "
        "but has minor omissions."
    ),
    "score5_description": (
        "The answer is completely accurate, relevant, "
        "and covers all important information from the reference."
    )
}

# Initialize rubric-based metric
rubrics_metric = RubricsScoreWithReference(
    llm=evaluator_llm,
    rubrics=custom_rubrics
)

# Test question
user_input = (
    "What is the annual leave entitlement and carry-forward policy?"
)

# Expected answer
reference = (
    "Employees receive 20 annual leave days per year. "
    "Up to 5 unused days can be carried forward."
)

# Positive test
positive_response = (
    "Employees receive 20 annual leave days per year. "
    "They can carry forward up to 5 unused annual leave days."
)

# Negative test
negative_response = (
    "Employees receive 30 annual leave days per year. "
    "They can carry forward up to 15 unused days."
)

print("===== RAGAS RUBRICS SCORE EVALUATION =====")

# Positive evaluation
print("\n===== POSITIVE TEST =====")

positive_result = rubrics_metric.score(
    user_input=user_input,
    response=positive_response,
    reference=reference
)

print("Positive Rubrics Score:", positive_result.value)
print("Positive Feedback:", positive_result.reason)

# Negative evaluation
print("\n===== NEGATIVE TEST =====")

negative_result = rubrics_metric.score(
    user_input=user_input,
    response=negative_response,
    reference=reference
)

print("Negative Rubrics Score:", negative_result.value)
print("Negative Feedback:", negative_result.reason)
