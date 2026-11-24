
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric
)
from deepeval.models import OllamaModel

question = "How many annual leave days do employees receive?"

context = [
    "Employees receive 20 annual leave days per year.",
    "Employees receive 10 sick leave days per year."
]

actual_output = "Employees receive 20 annual leave days per year."

test_case = LLMTestCase(
    input=question,
    actual_output=actual_output,
    retrieval_context=context,
    context=context
)

eval_model = OllamaModel(
    model="qwen2.5-coder:3b",
    base_url="http://localhost:11434",
    temperature=0
)

answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=eval_model
)

faithfulness = FaithfulnessMetric(
    threshold=0.7,
    model=eval_model
)

hallucination = HallucinationMetric(
    threshold=0.5,
    model=eval_model
)

evaluate(
    test_cases=[test_case],
    metrics=[
        answer_relevancy,
        faithfulness,
        hallucination
    ]
)

