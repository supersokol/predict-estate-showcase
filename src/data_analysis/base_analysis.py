import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BaseDataAnalysis:
    def __init__(self, data):
        self.data = data
        self.df = self._prepare_dataframe()

    def _prepare_dataframe(self):
        raise NotImplementedError("This method should be implemented in a subclass.")
    
    def filter_data(self, column, value):
        return self.df[self.df[column] == value]
    
    def missing_data_analysis(self):
        missing = self.df.isnull().sum()
        missing_percentage = (missing / len(self.df)) * 100
        print(pd.DataFrame({"Missing Count": missing, "Missing Percentage": missing_percentage}))

    def visualize_categorical_distribution(self, column):
        sns.countplot(data=self.df, x=column)
        plt.title(f"Distribution of {column}")
        plt.show()