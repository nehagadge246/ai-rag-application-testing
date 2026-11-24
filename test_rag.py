
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric
)
from deepeval.models import OllamaModel
from deepeval import evaluate

# User question
question = "How many annual leave days do employees receive?"

# Retrieved document context
retrieved_context = [
    "Employees are entitled to 20 annual leave days per year."
]

# Simulated AI-generated answer
actual_output = "Employees receive 20 annual leave days per year."

# Create RAG test case
test_case = LLMTestCase(
    input=question,
    actual_output=actual_output,
    retrieval_context=retrieved_context
)

# Configure Ollama evaluation model
eval_model = OllamaModel(
    model="qwen2.5-coder:3b",
    base_url="http://localhost:11434",
    temperature=0
)

# Define evaluation metrics
faithfulness = FaithfulnessMetric(
    threshold=0.7,
    model=eval_model
)

relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=eval_model
)

# Run both metrics
evaluate(
    test_cases=[test_case],
    metrics=[faithfulness, relevancy]
)

