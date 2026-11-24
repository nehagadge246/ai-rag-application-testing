
import ollama
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OllamaModel

# Step 1: Generate response using Ollama
prompt = "What is the capital of France?"

response = ollama.generate(
    model="qwen2.5-coder:3b",
    prompt=prompt
)

actual_output = response["response"]
print("\nAI Response:", actual_output)

# Step 2: Configure DeepEval's evaluation model
eval_model = OllamaModel(
    model="qwen2.5-coder:3b",
    base_url="http://localhost:11434",
    temperature=0
)

# Step 3: Create test case
test_case = LLMTestCase(
    input=prompt,
    actual_output=actual_output
)

# Step 4: Define evaluation metric
metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=eval_model,
    include_reason=True
)

# Step 5: Execute evaluation
evaluate(
    test_cases=[test_case],
    metrics=[metric]
)
