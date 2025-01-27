```python
# Import libraries
import json
from tabulate import tabulate

# Load the process registry from a JSON file
with open("../data/process_registry.json", "r") as f:
    process_registry = json.load(f)

# Display all registered processes in a tabular format
print("Registered Processes:")
process_table = [(p["id"], p["name"], p["status"], p["description"]) for p in process_registry]
print(tabulate(process_table, headers=["ID", "Name", "Status", "Description"]))

# Filter processes by status
status_filter = "completed"
filtered_processes = [p for p in process_registry if p["status"] == status_filter]
print(f"\nProcesses with status '{status_filter}':")
print(tabulate([(p["id"], p["name"]) for p in filtered_processes], headers=["ID", "Name"]))

```
