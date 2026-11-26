
import streamlit as st
import pandas as pd
from pathlib import Path

# 1. Page configuration
st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Application Evaluation Dashboard")
st.caption("RAGAs + Ollama + Qwen 2.5 Coder")

# 2. Load CSV report
REPORT_PATH = Path(__file__).parent / "ragas_dataset_validation_report.csv"

@st.cache_data
def load_report():
    return pd.read_csv(REPORT_PATH)

if not REPORT_PATH.exists():
    st.error("Evaluation report CSV not found.")
    st.stop()

df = load_report()

if df.empty:
    st.warning("The evaluation report is empty.")
    st.stop()

# 3. Sidebar filters
st.sidebar.header("Dashboard Filters")

scenarios = ["All"] + sorted(df["Scenario"].dropna().unique().tolist())

selected_scenario = st.sidebar.selectbox(
    "Select Test Scenario",
    scenarios
)

filtered_df = (
    df.copy() if selected_scenario == "All"
    else df[df["Scenario"] == selected_scenario].copy()
)

# 4. KPI summary
total = len(filtered_df)
passed = (filtered_df["Overall Status"] == "PASS").sum()
failed = (filtered_df["Overall Status"] == "FAIL").sum()

pass_rate = (passed / total * 100) if total else 0
average_score = filtered_df["Average Score"].mean() if total else 0

st.subheader("Evaluation Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Scenarios", total)
col2.metric("Passed", int(passed))
col3.metric("Failed", int(failed))
col4.metric("Pass Rate", f"{pass_rate:.1f}%")

st.metric("Average Evaluation Score", f"{average_score:.2%}")

st.divider()

# 5. Scenario-wise results
st.subheader("Scenario-wise Evaluation Results")

score_columns = [
    "Faithfulness Score",
    "Answer Relevancy Score",
    "Context Precision Score",
    "Context Recall Score",
    "Factual Correctness Score",
    "Context Relevance Score"
]

display_columns = [
    "Scenario",
    *score_columns,
    "Average Score",
    "Overall Status"
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True,
    hide_index=True
)

# 6. RAGAs metric comparison
st.subheader("RAGAs Metric Comparison")

metric_scores = filtered_df[score_columns].mean()

st.bar_chart(metric_scores)

# 7. Scenario average scores
st.subheader("Average Score by Scenario")

scenario_scores = filtered_df.set_index("Scenario")[["Average Score"]]

st.bar_chart(scenario_scores)

# 8. Failed scenarios
st.subheader("Failed Test Scenarios")

failed_df = filtered_df[
    filtered_df["Overall Status"] == "FAIL"
]

if failed_df.empty:
    st.success("No failed scenarios in the selected filter.")
else:
    st.dataframe(
        failed_df,
        use_container_width=True,
        hide_index=True
    )

# 9. Metric validation statuses
st.subheader("Metric Validation Status")

status_columns = [
    "Faithfulness Status",
    "Answer Relevancy Status",
    "Context Precision Status",
    "Context Recall Status",
    "Factual Correctness Status",
    "Context Relevance Status"
]

status_summary = pd.DataFrame({
    column.replace(" Status", ""):
    filtered_df[column].value_counts()
    for column in status_columns
}).fillna(0).astype(int)

st.dataframe(status_summary, use_container_width=True)


# --------------------------------------------------
# 10. ADVANCED METRIC-LEVEL QUALITY ANALYSIS
# --------------------------------------------------

st.subheader("Advanced Metric-Level Quality Analysis")

metric_status_map = {
    "Faithfulness": "Faithfulness Status",
    "Answer Relevancy": "Answer Relevancy Status",
    "Context Precision": "Context Precision Status",
    "Context Recall": "Context Recall Status",
    "Factual Correctness": "Factual Correctness Status",
    "Context Relevance": "Context Relevance Status"
}

analysis_rows = []

for metric, status_column in metric_status_map.items():

    if status_column not in filtered_df.columns:
        continue

    statuses = filtered_df[status_column].astype(str)

    analysis_rows.append({
        "Metric": metric,
        "PASS": (statuses == "PASS").sum(),
        "FAIL": (statuses == "FAIL").sum(),
        "REVIEW": (statuses == "REVIEW").sum(),
        "NOT APPLICABLE": (statuses == "NOT APPLICABLE").sum()
    })

analysis_df = pd.DataFrame(analysis_rows)

st.dataframe(
    analysis_df,
    use_container_width=True,
    hide_index=True
)

# Identify metrics requiring attention
st.subheader("QA Recommendations")

failed_metrics = analysis_df[
    analysis_df["FAIL"] > 0
]

if failed_metrics.empty:
    st.success("No metric-level validation failures found.")
else:
    for _, row in failed_metrics.iterrows():
        st.warning(
            f"{row['Metric']}: {int(row['FAIL'])} validation failure(s). "
            "Review the affected scenarios and investigate the evaluation results."
        )

# Context Recall diagnostic warning
st.info(
    "Context Recall is currently diagnostic-only in this project. "
    "Its REVIEW status does not contribute to the overall scenario verdict."
)

# --------------------------------------------------
# 11. DOWNLOAD CSV REPORT
# --------------------------------------------------

# 10. Download CSV report
st.subheader("Download Evaluation Report")

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered CSV Report",
    data=csv_data,
    file_name="ragas_filtered_evaluation_report.csv",
    mime="text/csv"
)

st.divider()

st.caption("RAG Evaluation Dashboard | Built with Streamlit and Pandas")
