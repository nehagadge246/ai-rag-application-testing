
from openai import OpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.metrics.collections import TopicAdherence
from langchain_core.messages import HumanMessage, AIMessage

# Configure Ollama LLM
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

evaluator_llm = llm_factory(
    model="qwen2.5-coder:3b",
    provider="openai",
    client=client,
    mode=Mode.MD_JSON
)

# Initialize Topic Adherence metric
topic_adherence = TopicAdherence(
    llm=evaluator_llm,
    mode="f1"
)

# Expected topics
reference_topics = [
    "Employee annual leave policy",
    "Employee sick leave policy",
    "Employee leave approval process"
]

# Positive conversation
positive_conversation = [
    HumanMessage(
        content="How many annual leave days do employees receive?"
    ),
    AIMessage(
        content="Employees receive 20 annual leave days per year. "
                "Leave requests must be approved by their manager."
    )
]

# Negative conversation
negative_conversation = [
    HumanMessage(
        content="Tell me about company activities."
    ),
    AIMessage(
        content="The company organizes annual cricket and football tournaments."
    )
]

print("===== RAGAS TOPIC ADHERENCE EVALUATION =====")

# Positive evaluation
print("\n===== POSITIVE TEST =====")

positive_result = topic_adherence.score(
    user_input=positive_conversation,
    reference_topics=reference_topics
)

print("Positive Topic Adherence Score:", positive_result.value)

# Negative evaluation
print("\n===== NEGATIVE TEST =====")

negative_result = topic_adherence.score(
    user_input=negative_conversation,
    reference_topics=reference_topics
)

print("Negative Topic Adherence Score:", negative_result.value)
