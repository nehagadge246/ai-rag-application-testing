
import evaluate
import pandas as pd
from pathlib import Path

print("===== HUGGING FACE ADVANCED EVALUATION =====")

# Test dataset
references = [1, 0, 1, 1, 0, 0]
predictions = [1, 0, 1, 0, 0, 1]

# Load Hugging Face evaluation metrics
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

# Calculate metrics
accuracy = accuracy_metric.compute(
    predictions=predictions,
    references=references
)["accuracy"]

precision = precision_metric.compute(
    predictions=predictions,
    references=references
)["precision"]

recall = recall_metric.compute(
    predictions=predictions,
    references=references
)["recall"]

f1 = f1_metric.compute(
    predictions=predictions,
    references=references
)["f1"]

# Display results
print("\n===== EVALUATION RESULTS =====")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 Score:  {f1:.2%}")

# QA thresholds
thresholds = {
    "Accuracy": 0.80,
    "Precision": 0.70,
    "Recall": 0.70,
    "F1 Score": 0.70
}

scores = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

# Validate each metric
results = []

for metric, score in scores.items():
    status = "PASS" if score >= thresholds[metric] else "FAIL"

    results.append({
        "Metric": metric,
        "Score": round(score, 4),
        "Threshold": thresholds[metric],
        "Status": status
    })

report = pd.DataFrame(results)

print("\n===== QA THRESHOLD VALIDATION =====")
print(report.to_string(index=False))

# Save report
report_path = (
    Path(__file__).parent /
    "huggingface_advanced_evaluation_report.csv"
)

report.to_csv(report_path, index=False)

print("\nReport saved:", report_path.name)
print("\n===== EVALUATION COMPLETED =====")
