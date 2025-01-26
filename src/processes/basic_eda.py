
from src.registry.process_registry import register_process
import pandas as pd
@register_process("EDA Summary", metadata={"description": "Generates summary statistics for a dataset."})
def eda_summary(data: pd.DataFrame) -> pd.DataFrame:
    return data.describe()

@register_process("Generate Correlation Matrix", metadata={
    "type": "analysis",
    "description": "Generates a correlation matrix for numeric columns.",
    "parameters": {
        "df": {"type": "pd.DataFrame", "description": "Input DataFrame."}
    },
    "response": {"type": "pd.DataFrame", "description": "Correlation matrix."}
})
def generate_correlation_matrix(df):
    return df.corr()

@register_process("Plot Histogram", metadata={
    "type": "visualization",
    "description": "Plots a histogram for a specified column.",
    "parameters": {
        "df": {"type": "pd.DataFrame", "description": "Input DataFrame."},
        "column": {"type": "str", "description": "Column to plot."}
    },
    "response": {"type": "plot", "description": "Histogram plot."}
})
def plot_histogram(df, column):
    import plotly.express as px
    return px.histogram(df, x=column)