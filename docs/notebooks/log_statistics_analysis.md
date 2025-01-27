```python
# Import libraries
import re
import pandas as pd
import matplotlib.pyplot as plt

# Path to the log file
log_file = "../logs/app.log"

# Parsing logs
logs = []
with open(log_file, "r") as f:
    for line in f:
        # Regular expression to match log entries with time, level, and message
        match = re.search(r"(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?(?P<level>INFO|ERROR|WARNING).*?: (?P<message>.+)", line)
        if match:
            logs.append(match.groupdict())

# Creating a DataFrame from the parsed logs
log_df = pd.DataFrame(logs)

# Logging level statistics
log_stats = log_df["level"].value_counts()
print("Log level statistics:")
print(log_stats)

# Visualizing log activity over time
log_df["time"] = pd.to_datetime(log_df["time"])
log_df.set_index("time", inplace=True)
log_df.resample("1H").size().plot(kind="line", title="Log Activity", xlabel="Time", ylabel="Number of Entries")
plt.show()

```
