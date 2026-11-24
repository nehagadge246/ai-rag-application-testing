
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.models import OllamaModel

question = "What is software testing?"

actual_output = """
Software testing is the process of checking an application
to identify defects and verify that it works as expected.
It helps improve software quality and reliability.
"""

test_case = LLMTestCase(
    input=question,
    actual_output=actual_output
)

eval_model = OllamaModel(
    model="qwen2.5-coder:3b",
    base_url="http://localhost:11434",
    temperature=0
)

professionalism_metric = GEval(
    name="Professionalism",
    criteria=(
        "Evaluate whether the response uses professional, "
        "clear, and appropriate language. It should be easy "
        "for a beginner to understand."
    ),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT
    ],
    threshold=0.7,
    model=eval_model
)

evaluate(
    test_cases=[test_case],
    metrics=[professionalism_metric]
)
