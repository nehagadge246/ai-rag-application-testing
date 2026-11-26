
from pathlib import Path
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("===== DOCUMENT INGESTION AND CHUNKING TESTING =====")

# 1. Create sample HR policy document
document = """
HR EMPLOYEE POLICY

Annual Leave:
Employees receive 20 annual leave days per year.
Employees can carry forward up to 5 unused leave days.

Sick Leave:
Employees receive 10 sick leave days per year.

Remote Work:
Employees can work remotely twice a week.

Office Facilities:
Office parking is available for employees.
"""

# 2. Document ingestion
print("\n===== DOCUMENT INGESTION TEST =====")

if document.strip():
    print("PASS: Document loaded successfully.")
    ingestion_status = "PASS"
else:
    print("FAIL: Document is empty.")
    ingestion_status = "FAIL"

# 3. Configure text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# 4. Generate chunks
chunks = splitter.split_text(document)

print("\n===== DOCUMENT CHUNKING TEST =====")
print("Total chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}:")
    print(chunk)
    print("Character count:", len(chunk))

# 5. Validate chunk generation
chunk_status = (
    "PASS"
    if len(chunks) > 0 and all(chunk.strip() for chunk in chunks)
    else "FAIL"
)

print("\nChunk Generation:", chunk_status)

# 6. Validate chunk size
oversized_chunks = [
    chunk for chunk in chunks
    if len(chunk) > 150
]

size_status = "PASS" if not oversized_chunks else "FAIL"

print("Chunk Size Validation:", size_status)

# 7. Validate content preservation
original_content = document.replace("\n", " ").strip()

normalized_chunks = " ".join(chunks).replace("\n", " ")

content_status = (
    "PASS"
    if all(
        phrase in normalized_chunks
        for phrase in [
            "20 annual leave days",
            "5 unused leave days",
            "10 sick leave days",
            "twice a week"
        ]
    )
    else "FAIL"
)

print("Content Preservation:", content_status)

# 8. Generate QA report
results = [
    {"Test": "Document Ingestion", "Status": ingestion_status},
    {"Test": "Chunk Generation", "Status": chunk_status},
    {"Test": "Chunk Size Validation", "Status": size_status},
    {"Test": "Content Preservation", "Status": content_status}
]

report = pd.DataFrame(results)

report_path = (
    Path(__file__).parent /
    "document_ingestion_chunking_report.csv"
)

report.to_csv(report_path, index=False)

print("\n===== FINAL QA SUMMARY =====")
print(report.to_string(index=False))

print("\nPassed:", sum(r["Status"] == "PASS" for r in results))
print("Failed:", sum(r["Status"] == "FAIL" for r in results))
print("Report saved:", report_path.name)

print("\n===== TESTING COMPLETED =====")
