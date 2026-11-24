
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import OllamaModel

eval_model = OllamaModel(
    model="qwen2.5-coder:3b",
    base_url="http://localhost:11434",
    temperature=0
)

context = [
    "Employees receive 20 annual leave days per year.",
    "Employees receive 10 sick leave days per year."
]

test_cases = [
    LLMTestCase(
        input="How many annual leave days do employees receive?",
        actual_output="Employees receive 20 annual leave days per year.",
        retrieval_context=context
    ),
    LLMTestCase(
        input="How many annual leave days do employees receive?",
        actual_output="Employees receive 30 annual leave days per year.",
        retrieval_context=context
    ),
    LLMTestCase(
        input="How many sick leave days do employees receive?",
        actual_output="Employees receive 10 sick leave days per year.",
        retrieval_context=context
    ),
    LLMTestCase(
        input="How many annual leave days do employees receive?",
        actual_output="Python is a programming language.",
        retrieval_context=context
    )
]

metrics = [
    AnswerRelevancyMetric(threshold=0.7, model=eval_model),
    FaithfulnessMetric(threshold=0.7, model=eval_model)
]

evaluate(
    test_cases=test_cases,
    metrics=metrics
)
