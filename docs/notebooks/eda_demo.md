```python
# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Path to the file
file_path = "../data/sample_data.csv"

# Load the data
df = pd.read_csv(file_path)

# Initial analysis
print(f"Data dimensions (rows, columns): {df.shape}")
print("Data types for each column:")
print(df.dtypes)
print("\nMissing values in each column:")
print(df.isnull().sum())

# Visualization of the distribution of a specific column
sns.histplot(df["column_name"], kde=True)
plt.title("Distribution of values in 'column_name'")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

# Correlation matrix to understand relationships between numeric features
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

```
