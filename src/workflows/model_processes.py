from src.registry import model_registry


@model_registry.register(name="Train Model", metadata={
    "type": "model_training",
    "description": "Trains a model using the Model Registry.",
    "parameters": {
        "model_name": {"type": "str", "description": "Name of the model in the Model Registry."}
    },
    "response": {"type": "model", "description": "Trained model."}
})
def train_model(df, model_name, target_column="target"):
    model_info = model_registry.get_model(model_name)
    if not model_info:
        raise ValueError(f"Model {model_name} not found in registry.")
    model_func = model_info["model_func"]
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return model_func(X, y)