from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from src.registry.model_registry import ModelRegistry

model_registry = ModelRegistry()

@model_registry.register("Linear Regression", metadata={
    "type": "regression",
    "description": "Fits a linear regression model to the data.",
    "parameters": {"fit_intercept": {"type": "bool", "description": "Whether to calculate the intercept."}},
    "response": {"type": "model", "description": "Trained linear regression model."}
})
def train_linear_regression(X, y, fit_intercept=True):
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X, y)
    return model

@model_registry.register("Polynomial Regression", metadata={
    "type": "regression",
    "description": "Fits a polynomial regression model to the data.",
    "parameters": {"degree": {"type": "int", "description": "Degree of the polynomial."}},
    "response": {"type": "model", "description": "Trained polynomial regression model."}
})
def train_polynomial_regression(X, y, degree=2):
    model = Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree)),
        ("linear_reg", LinearRegression())
    ])
    model.fit(X, y)
    return model
