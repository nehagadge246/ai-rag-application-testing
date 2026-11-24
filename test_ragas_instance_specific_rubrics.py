
from openai import OpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.metrics.collections import InstanceSpecificRubrics

# Configure Ollama LLM
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

evaluator_llm = llm_factory(
    model="qwen3:latest",
    provider="openai",
    client=client,
    mode=Mode.MD_JSON
)

# Initialize instance-specific rubric metric
rubrics_metric = InstanceSpecificRubrics(
    llm=evaluator_llm
)

# Test question
user_input = (
    "What is the annual leave entitlement and carry-forward policy?"
)

# Reference answer
reference = (
    "Employees receive 20 annual leave days per year. "
    "Up to 5 unused days can be carried forward."
)

# Instance-specific scoring criteria
custom_rubrics = {
    "score1_description": (
        "The response is completely incorrect and contradicts "
        "both annual leave facts."
    ),
    "score2_description": (
        "The response contains major factual errors about "
        "annual leave entitlement and carry-forward."
    ),
    "score3_description": (
        "The response correctly mentions one policy fact "
        "but incorrectly states or misses the other."
    ),
    "score4_description": (
        "The response correctly explains both policy facts "
        "but has a minor omission."
    ),
    "score5_description": (
        "The response accurately states 20 annual leave days "
        "and the maximum carry-forward of 5 unused days."
    )
}

# Positive test
positive_response = (
    "Employees receive 20 annual leave days per year. "
    "They can carry forward up to 5 unused days."
)

# Negative test
negative_response = (
    "Employees receive 30 annual leave days per year. "
    "They can carry forward up to 15 unused days."
)

print("===== RAGAS INSTANCE-SPECIFIC RUBRICS =====")

print("\n===== POSITIVE TEST =====")

positive_result = rubrics_metric.score(
    rubrics=custom_rubrics,
    user_input=user_input,
    response=positive_response,
    reference=reference
)

print("Positive Rubrics Score:", positive_result.value)
print("Positive Feedback:", positive_result.reason)

print("\n===== NEGATIVE TEST =====")

negative_result = rubrics_metric.score(
    rubrics=custom_rubrics,
    user_input=user_input,
    response=negative_response,
    reference=reference
)

print("Negative Rubrics Score:", negative_result.value)
print("Negative Feedback:", negative_result.reason)
