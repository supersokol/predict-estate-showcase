from src.registry.process_registry import register_process
from src.core.logger import logger
import pandas as pd

@register_process("Remove Duplicates", metadata={
    "type": "cleaning",
    "description": "Removes duplicate rows from the DataFrame.",
    "parameters": {
        "df": {
            "type": "pd.DataFrame",
            "description": "The input DataFrame."
        }
    },
    "response": {
        "type": "pd.DataFrame",
        "description": "The DataFrame without duplicate rows."
    }
})
def remove_duplicates(df):
    return df.drop_duplicates()

@register_process("Standardize Columns", metadata={
    "type": "transformation",
    "description": "Standardizes numeric columns to zero mean and unit variance.",
    "parameters": {
        "df": {"type": "pd.DataFrame", "description": "Input DataFrame."}
    },
    "response": {"type": "pd.DataFrame", "description": "Standardized DataFrame."}
})
def standardize_columns(df):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

@register_process("Categorical Encoding")
def categorical_encoding(df):
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = encoder.fit_transform(df[col])
    return df

@register_process("One-Hot Encoding", metadata={
    "type": "transformation",
    "description": "Performs one-hot encoding on categorical columns.",
    "parameters": {
        "df": {"type": "pd.DataFrame", "description": "Input DataFrame."},
        "columns": {"type": "list", "description": "List of columns to encode."}
    },
    "response": {"type": "pd.DataFrame", "description": "DataFrame with encoded columns."}
})
def one_hot_encode(df, columns):
    return pd.get_dummies(df, columns=columns)


@register_process("Remove Duplicates")
def remove_duplicates(df, **kwargs):
    logger.info("Removing duplicates from DataFrame.")
    return df.drop_duplicates()

@register_process("Fill Missing Values", metadata={
    "type": "cleaning",
    "description": "Fills missing values with a specified value.",
    "parameters": {
        "df": {"type": "pd.DataFrame", "description": "Input DataFrame."},
        "value": {"type": "any", "description": "Value to fill missing cells with."}
    },
    "response": {"type": "pd.DataFrame", "description": "DataFrame with missing values filled."}
})
def fill_missing_values(df, value="N/A"):
    return df.fillna(value)

@register_process("Drop Rows with Many Missing Values", metadata={
    "type": "cleaning",
    "description": "Drops rows where the percentage of missing values exceeds a threshold.",
    "parameters": {
        "df": {"type": "pd.DataFrame", "description": "Input DataFrame."},
        "threshold": {"type": "float", "description": "Threshold for missing values (0 to 1)."}
    },
    "response": {"type": "pd.DataFrame", "description": "DataFrame with rows dropped."}
})
def drop_rows_with_many_missing(df, threshold=0.5):
    return df[df.isnull().mean(axis=1) < threshold]

@register_process("Normalize Columns", metadata={
    "type": "transformation",
    "description": "Normalizes numeric columns to a range [0, 1].",
    "parameters": {
        "df": {"type": "pd.DataFrame", "description": "Input DataFrame."}
    },
    "response": {"type": "pd.DataFrame", "description": "Normalized DataFrame."}
})
def normalize_columns(df):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

@register_process("Normalize Numeric Columns", metadata={
    "type": "transformation",
    "description": "Normalizes numeric columns in the DataFrame.",
    "parameters": {
        "df": {
            "type": "pd.DataFrame",
            "description": "The input DataFrame."
        }
    },
    "response": {
        "type": "pd.DataFrame",
        "description": "The DataFrame with normalized numeric columns."
    }
})
def normalize_numeric_columns(df):
    for col in df.select_dtypes(include="number").columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


@register_process("Merge CSV Files", metadata={
    "type": "file_processing",
    "description": "Merges multiple CSV files into one DataFrame.",
    "parameters": {
        "file_paths": {"type": "list", "description": "List of file paths to merge."}
    },
    "response": {"type": "pd.DataFrame", "description": "Merged DataFrame."}
})
def merge_csv_files(file_paths):
    df_list = [pd.read_csv(file) for file in file_paths]
    return pd.concat(df_list, ignore_index=True)