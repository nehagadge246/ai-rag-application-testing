
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

print("===== HUGGING FACE CONFUSION MATRIX TESTING =====")

# Ground-truth labels
references = [1, 0, 1, 1, 0, 0]

# Predicted labels
predictions = [1, 0, 1, 0, 0, 1]

# Generate confusion matrix
matrix = confusion_matrix(
    references,
    predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = matrix.ravel()

print("\n===== CONFUSION MATRIX =====")
print("                 Predicted")
print("                 Negative  Positive")
print(f"Actual Negative      {tn}         {fp}")
print(f"Actual Positive      {fn}         {tp}")

# Display error counts
print("\n===== ERROR ANALYSIS =====")
print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)

# Generate classification report
report = classification_report(
    references,
    predictions,
    labels=[0, 1],
    target_names=["Negative", "Positive"],
    output_dict=True,
    zero_division=0
)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    references,
    predictions,
    labels=[0, 1],
    target_names=["Negative", "Positive"],
    zero_division=0
))

# Export confusion matrix
matrix_df = pd.DataFrame(
    matrix,
    index=["Actual Negative", "Actual Positive"],
    columns=["Predicted Negative", "Predicted Positive"]
)

matrix_path = (
    Path(__file__).parent /
    "huggingface_confusion_matrix.csv"
)

matrix_df.to_csv(matrix_path)

# Export detailed classification report
report_df = pd.DataFrame(report).transpose()

report_path = (
    Path(__file__).parent /
    "huggingface_classification_report.csv"
)

report_df.to_csv(report_path)

print("===== REPORTS GENERATED =====")
print(matrix_path.name)
print(report_path.name)

print("\n===== CONFUSION MATRIX TEST COMPLETED =====")
