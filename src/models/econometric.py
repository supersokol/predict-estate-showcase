from src.registry import model_registry
import statsmodels.api as sm

@model_registry.register("ARIMA", metadata={
    "type": "econometric",
    "description": "Fits an ARIMA model to the data.",
    "parameters": {
        "order": {"type": "tuple", "description": "The (p, d, q) order of the model."}
    },
    "response": {"type": "model", "description": "Trained ARIMA model."}
})
def train_arima_model(data, order=(1, 1, 1)):
    model = sm.tsa.ARIMA(data, order=order)
    fitted_model = model.fit()
    return fitted_model
