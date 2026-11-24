
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric
from deepeval.models import OllamaModel

# User question
question = "How many annual leave days do employees receive?"

# Retrieved context
context = [
    "Employees receive 20 annual leave days per year."
]

# AI-generated response
actual_output = "Employees receive 20 annual leave days per year."

# Create test case
test_case = LLMTestCase(
    input=question,
    actual_output=actual_output,
    context=context
)

# Configure Ollama
eval_model = OllamaModel(
    model="qwen2.5-coder:3b",
    base_url="http://localhost:11434",
    temperature=0
)

# Define hallucination metric
metric = HallucinationMetric(
    threshold=0.5,
    model=eval_model,
    include_reason=True
)

# Execute evaluation
evaluate(
    test_cases=[test_case],
    metrics=[metric]
)
