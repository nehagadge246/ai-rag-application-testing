
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric
from deepeval.models import OllamaModel

# User question
question = "How many annual leave and sick leave days are provided?"

# AI-generated answer
actual_output = (
    "Employees receive 20 annual leave days "
    "and 10 sick leave days."
)

# Expected answer
expected_output = (
    "Employees receive 20 annual leave days "
    "and 10 sick leave days."
)

# Retrieved context
retrieval_context = [
    "Employees receive 20 annual leave days per year.",
    "Employees receive 10 sick leave days per year."
]

# Create test case
test_case = LLMTestCase(
    input=question,
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=retrieval_context
)

# Configure Ollama
eval_model = OllamaModel(
    model="qwen2.5-coder:3b",
    base_url="http://localhost:11434",
    temperature=0
)

# Define Contextual Recall metric
metric = ContextualRecallMetric(
    threshold=0.7,
    model=eval_model,
    include_reason=True
)

# Run evaluation
evaluate(
    test_cases=[test_case],
    metrics=[metric]
)
