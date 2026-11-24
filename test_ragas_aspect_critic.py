
from openai import OpenAI
from instructor import Mode
from ragas.llms import llm_factory
from ragas.metrics import _AspectCritic
from ragas.dataset_schema import SingleTurnSample

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

# Define Aspect Critic
aspect_critic = _AspectCritic(
    name="annual_leave_accuracy",
    definition=(
        "Does the response provide accurate information about "
        "the employee annual leave policy without introducing "
        "unsupported facts?"
    ),
    llm=evaluator_llm,
    strictness=1
)

# Positive test: Answer supported by context
positive_sample = SingleTurnSample(
    user_input="How many annual leave days do employees receive?",
    response="Employees receive 20 annual leave days per year.",
    retrieved_contexts=[
        "Employees receive 20 annual leave days per year."
    ],
    reference="Employees receive 20 annual leave days per year."
)

# Negative test: Answer contradicts context
negative_sample = SingleTurnSample(
    user_input="How many annual leave days do employees receive?",
    response="Employees receive 30 annual leave days per year.",
    retrieved_contexts=[
        "Employees receive 20 annual leave days per year."
    ],
    reference="Employees receive 20 annual leave days per year."
)

print("===== RAGAS ASPECT CRITIC EVALUATION =====")

# Positive evaluation
print("\n===== POSITIVE TEST =====")

positive_score = aspect_critic.single_turn_score(positive_sample)

print("Positive Aspect Critic Score:", positive_score)

# Negative evaluation
print("\n===== NEGATIVE TEST =====")

negative_score = aspect_critic.single_turn_score(negative_sample)

print("Negative Aspect Critic Score:", negative_score)
