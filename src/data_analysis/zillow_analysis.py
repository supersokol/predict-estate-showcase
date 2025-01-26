import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.core.logger import logger
from src.data_analysis.base_analysis import BaseDataAnalysis
import statistics

def analyze_and_log_zillow_data(json_data):
    """
    Analyzes the Zillow datasets JSON, logs key metrics, and updates metadata.

    :param json_data: Dictionary representing the Zillow datasets JSON structure.
    :return: Updated JSON with added metadata.
    """
    # Initialize counters and collectors
    dataset_counts = 0
    file_counts = 0
    type_counts = []

    files_per_type = []
    files_per_dataset = []

    # Iterate through datasets
    for dataset_name, datasets in json_data.get("zillow_datasets", {}).items():
        dataset_counts += 1
        type_count = len(datasets)
        type_counts.append(type_count)

        dataset_file_count = 0
        for dataset in datasets:
            type_file_count = 0
            for housing_type in dataset.get("housing_types", []):
                for feature in housing_type.get("features", []):
                    geography_files = len(feature.get("geography", []))
                    type_file_count += geography_files
                    dataset_file_count += geography_files
                    file_counts += geography_files
            files_per_type.append(type_file_count)
        files_per_dataset.append(dataset_file_count)

    # Calculate metrics
    avg_types_per_dataset = statistics.mean(type_counts)
    min_types_per_dataset = min(type_counts)
    max_types_per_dataset = max(type_counts)

    avg_files_per_type = statistics.mean(files_per_type) if files_per_type else 0
    min_files_per_type = min(files_per_type) if files_per_type else 0
    max_files_per_type = max(files_per_type) if files_per_type else 0

    avg_files_per_dataset = statistics.mean(files_per_dataset) if files_per_dataset else 0
    min_files_per_dataset = min(files_per_dataset) if files_per_dataset else 0
    max_files_per_dataset = max(files_per_dataset) if files_per_dataset else 0

    # Log results
    logger.info(f"Total number of datasets: {dataset_counts}")
    logger.info(f"Total number of files: {file_counts}")
    logger.info(f"Average types per dataset: {avg_types_per_dataset}")
    logger.info(f"Min/Max types per dataset: {min_types_per_dataset}/{max_types_per_dataset}")
    logger.info(f"Average files per type: {avg_files_per_type}")
    logger.info(f"Min/Max files per type: {min_files_per_type}/{max_files_per_type}")
    logger.info(f"Average files per dataset: {avg_files_per_dataset}")
    logger.info(f"Min/Max files per dataset: {min_files_per_dataset}/{max_files_per_dataset}")

    # Update metadata
    json_data["metadata"]["analysis"] = {
        "total_datasets": dataset_counts,
        "total_files": file_counts,
        "types_per_dataset": {
            "average": avg_types_per_dataset,
            "min": min_types_per_dataset,
            "max": max_types_per_dataset
        },
        "files_per_type": {
            "average": avg_files_per_type,
            "min": min_files_per_type,
            "max": max_files_per_type
        },
        "files_per_dataset": {
            "average": avg_files_per_dataset,
            "min": min_files_per_dataset,
            "max": max_files_per_dataset
        }
    }

    return json_data


class ZillowDataAnalysis(BaseDataAnalysis):
    def _prepare_dataframe(self):
        """
        Prepare the Zillow dataset for analysis.
        Convert nested JSON structure into a DataFrame.
        """
        records = []

        for label, datasets in self.data.get("zillow_datasets", {}).items():
            for dataset in datasets:
                for housing_type in dataset.get("housing_types", []):
                    for feature in housing_type.get("features", []):
                        record = {
                            "dataset_label": label,
                            "type": dataset.get("type"),
                            "data_type": dataset.get("data_type"),
                            "housing_type": housing_type.get("housing_type"),
                            "geography": feature.get("geography"),
                            "additional_features": feature.get("additional_features"),
                            "full_name": feature.get("full_name"),
                        }
                        records.append(record)

        return pd.DataFrame(records)



    def summary_statistics(self):
        """
        Generate summary statistics for the dataset.
        """
        print("\n--- Summary Statistics ---\n")
        print(self.df.describe(include="all"))

    def missing_data_analysis(self):
        """
        Analyze missing data in the dataset.
        """
        print("\n--- Missing Data Analysis ---\n")
        missing = self.df.isnull().sum()
        missing_percentage = (missing / len(self.df)) * 100
        print(pd.DataFrame({"Missing Count": missing, "Missing Percentage": missing_percentage}))

    def visualize_housing_types_distribution(self):
        """
        Visualize the distribution of housing types.
        """
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.df, y="housing_type", order=self.df["housing_type"].value_counts().index)
        plt.title("Distribution of Housing Types")
        plt.xlabel("Count")
        plt.ylabel("Housing Type")
        plt.show()

    def visualize_geography_distribution(self):
        """
        Visualize the geography distribution based on unique geography values.
        """
        if self.df["geography"].notnull().any():
            # Explode the 'geography' column to separate each entry
            geo_df = self.df.explode("geography").dropna(subset=["geography"])
        
            # Extract only unique 'geography_value' from each geography entry
            geo_df["geography_value"] = geo_df["geography"].apply(
                lambda x: x["geography_value"] if isinstance(x, dict) and "geography_value" in x else None
            )

            # Drop rows with missing 'geography_value'
            geo_df = geo_df.dropna(subset=["geography_value"])

            # Group by 'geography_value' to count occurrences
            geo_counts = geo_df["geography_value"].value_counts().reset_index()
            geo_counts.columns = ["geography_value", "count"]
            geo_counts.sort_values("count", ascending=False, inplace=True)

            # Visualization
            plt.figure(figsize=(12, 6))
            sns.barplot(data=geo_counts, x="count", y="geography_value")
            plt.title("Geography Value Distribution")
            plt.xlabel("Count")
            plt.ylabel("Geography Value")
            plt.show()
        else:
            print("No geography data available for visualization.")


    def additional_features_analysis(self):
        """
        Analyze additional features to understand their distribution.
        """
        feature_counts = {}
        
        for features in self.df["additional_features"].dropna():
            for feature in features:
                for key in feature.keys():
                    feature_counts[key] = feature_counts.get(key, 0) + 1

        feature_df = pd.DataFrame.from_dict(feature_counts, orient="index", columns=["Count"])
        feature_df.sort_values("Count", ascending=False, inplace=True)

        print("\n--- Additional Features Analysis ---\n")
        print(feature_df)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=feature_df.index, y=feature_df["Count"])
        plt.title("Additional Features Distribution")
        plt.xlabel("Feature")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.show()

    def analyze_correlation(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.show()
        
    def generate_report(self, report_path="reports/analysis_report.txt"):
        with open(report_path, "w") as f:
            f.write("Analysis Report\n")
            f.write("==============\n")
            f.write(self.df.describe().to_string())
            f.write("\n\nSummary statistics\n")
            f.write(self.summary_statistics().to_string())
            f.write("\n\nMissing Data Analysis\n")
            f.write(self.missing_data_analysis().to_string())
        print(f"Report saved to {report_path}")

# Example usage
def run_analysis(data):
    """
    Run analysis on the Zillow dataset.
    """

    analysis = ZillowDataAnalysis(data)

    # Summary statistics
    analysis.summary_statistics()

    # Missing data analysis
    analysis.missing_data_analysis()

    # Visualizations
    analysis.visualize_housing_types_distribution()
    analysis.visualize_geography_distribution()
    analysis.additional_features_analysis()
