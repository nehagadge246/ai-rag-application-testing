
import ollama

# Test input
prompt = "What is the capital of France?"

# Expected answer
expected_answer = "Paris"

# Generate AI response
response = ollama.generate(
    model="qwen2.5-coder:3b",
    prompt=prompt
)

actual_answer = response["response"]

print("\nAI Response:")
print(actual_answer)

# Test 1: Check response is not empty
assert actual_answer.strip() != "", "FAIL: Empty response"

print("\nPASS: Response is not empty")

# Test 2: Check expected information
assert expected_answer.lower() in actual_answer.lower(), \
    "FAIL: Expected answer not found"

print("PASS: Expected answer found")

print("\nAll basic LLM tests passed!")
