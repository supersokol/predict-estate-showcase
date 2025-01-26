from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from src.registry import model_registry

@model_registry.register("Random Forest with Grid Search", metadata={
    "type": "optimization",
    "description": "Optimizes a Random Forest model using Grid Search.",
    "parameters": {
        "param_grid": {"type": "dict", "description": "Grid of hyperparameters for optimization."}
    },
    "response": {"type": "model", "description": "Optimized Random Forest model."}
})
def train_random_forest_with_grid_search(X, y, param_grid):
    model = RandomForestRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error")
    grid_search.fit(X, y)
    return grid_search.best_estimator_
