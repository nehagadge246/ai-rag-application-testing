
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval.models import OllamaModel

# Context provided to the AI
context = [
    "The company was founded in 2010.",
    "The company headquarters are located in Pune, India."
]

# AI-generated answer
actual_output = "The company was founded in 2010 and is headquartered in Pune, India."

# Create test case
test_case = LLMTestCase(
    input="When was the company founded and where is its headquarters?",
    actual_output=actual_output,
    retrieval_context=context
)

# Configure local evaluation model
eval_model = OllamaModel(
    model="qwen2.5-coder:3b",
    base_url="http://localhost:11434",
    temperature=0
)

# Define faithfulness metric
metric = FaithfulnessMetric(
    threshold=0.7,
    model=eval_model,
    include_reason=True
)

# Run evaluation
evaluate(
    test_cases=[test_case],
    metrics=[metric]
)
