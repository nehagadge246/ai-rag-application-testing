
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.models import OllamaModel

# User question
question = "How many annual leave days do employees receive?"

# AI-generated answer
actual_output = "Employees receive 20 annual leave days per year."

# Expected answer
expected_output = "Employees receive 20 annual leave days per year."

# Retrieved documents
retrieval_context = [
    "Employees receive 20 annual leave days per year.",
    "The company provides free office parking.",
    "Employees can work remotely twice a week."
]

# Create test case
test_case = LLMTestCase(
    input=question,
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=retrieval_context
)

# Configure evaluation model
eval_model = OllamaModel(
    model="qwen2.5-coder:3b",
    base_url="http://localhost:11434",
    temperature=0
)

# Define metric
metric = ContextualPrecisionMetric(
    threshold=0.7,
    model=eval_model,
    include_reason=True
)

# Run evaluation
evaluate(
    test_cases=[test_case],
    metrics=[metric]
)
