from src.core import logger
from src.integrations.external_api_calls import call_llm_api
from src.registry import metaheuristics_registry, process_registry


@metaheuristics_registry.register("Feature Analysis", metadata={
    "description": "Analyzes features of a dataset and provides statistics.",
    "type": "analysis",
    "steps": [
        {"type": "process", "name": "Detect Numeric Features"},
        {"type": "process", "name": "Detect Categorical Features"},
        {"type": "process", "name": "Generate Correlation Matrix"},
        {"type": "external_call", "api": "llm", "parameters": {
            "prompt": "Summarize the most important features of the dataset.",
            "output_column": "llm_summary"
        }}
    ]
})
def feature_analysis(df, context=None):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    metadata = {
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "total_columns": len(df.columns),
        "total_rows": len(df)
    }
    logger.info(f"Feature Analysis Metadata: {metadata}")

    df = process_registry.execute("Detect Numeric Features", df)
    df = process_registry.execute("Detect Categorical Features", df)

    summary = call_llm_api(f"Summarize the dataset: {df.describe().to_string()}")
    df["LLM Summary"] = summary

    return df
