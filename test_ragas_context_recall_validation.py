
from openai import OpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall

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

context_recall = ContextRecall(
    llm=evaluator_llm
)

user_input = "How many annual leave days do employees receive?"

reference = (
    "Employees receive 20 annual leave days per year."
)

scenarios = [
    {
        "name": "Positive",
        "contexts": [
            "Employees receive 20 annual leave days per year."
        ]
    },
    {
        "name": "Negative",
        "contexts": [
            "The company organizes annual sports competitions.",
            "Employees participate in football tournaments."
        ]
    }
]

print("\n===== CONTEXT RECALL VALIDATION =====")

for scenario in scenarios:

    print("\nScenario:", scenario["name"])

    result = context_recall.score(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=scenario["contexts"]
    )

    print("Retrieved Contexts:", scenario["contexts"])
    print("Context Recall Score:", result.value)
