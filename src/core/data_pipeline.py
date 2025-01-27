# TODO rewrite this logic
import json
import os
from src.core.logger import logger
from src.core.errors import handle_error
from utils.process_registry import process_registry
from utils.model_registry import model_registry

class PipelineExecutor:
    def __init__(self, config_path, max_iterations=100):
        self.config_path = config_path
        self.max_iterations = max_iterations
        self.iteration_count = 0

    def execute(self, df, context=None):
        """
        Execute the pipeline based on the JSON configuration.
        """
        with open(self.config_path, "r") as f:
            pipeline_config = json.load(f)
        
        pipeline_name = pipeline_config["name"]
        logger.info(f"Starting pipeline: {pipeline_name}")
        
        termination_condition = pipeline_config.get("termination_condition", None)
        max_iterations = pipeline_config.get("max_iterations", self.max_iterations)
        steps = pipeline_config["steps"]

        while self.iteration_count < max_iterations:
            logger.info(f"Iteration {self.iteration_count + 1}/{max_iterations}")
            for step in steps:
                step_type = step["type"]

                if step_type == "process":
                    self._execute_process(step, df)
                elif step_type == "model":
                    self._execute_model(step, df)
                elif step_type == "pipeline":
                    self._execute_nested_pipeline(step, df)
                else:
                    logger.warning(f"Unknown step type: {step_type}. Skipping.")
            
            self.iteration_count += 1

            if termination_condition and eval(termination_condition):
                logger.info("Termination condition met. Stopping pipeline.")
                break
        
        if self.iteration_count >= max_iterations:
            logger.warning("Maximum iterations reached. Pipeline terminated.")
        
        logger.info(f"Pipeline {pipeline_name} completed.")

    def _execute_process(self, step, df):
        process_name = step["name"]
        parameters = step.get("parameters", {})
        logger.info(f"Executing process: {process_name}")
        process_registry.execute(process_name, df, **parameters)

    def _execute_model(self, step, df):
        model_name = step["name"]
        action = step["action"]
        parameters = step.get("parameters", {})
        if action == "train":
            logger.info(f"Training model: {model_name}")
            model_func = model_registry.get_model(model_name)["model_func"]
            model_func(df, **parameters)
        elif action == "predict":
            logger.info(f"Making predictions with model: {model_name}")
            # Use trained model for prediction
        else:
            logger.warning(f"Unknown model action: {action}")

    def _execute_nested_pipeline(self, step, df):
        nested_pipeline_name = step["name"]
        logger.info(f"Executing nested pipeline: {nested_pipeline_name}")
        nested_pipeline_path = f"pipelines/{nested_pipeline_name}.json"
        nested_executor = PipelineExecutor(nested_pipeline_path, self.max_iterations)
        nested_executor.execute(df)



def load_pipeline_config(config_path="pipeline_config.json"):
    """
    Load pipeline configurations from a JSON file.
    :param config_path: Path to the JSON configuration file.
    :return: Dictionary of pipelines.
    """
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)["pipelines"]
    else:
        logger.error(f"Pipeline configuration file not found at {config_path}.")
        return {}
    
    
'''import pandas as pd

# Example usage
if __name__ == "__main__":
    df = pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie", "Alice"],
        "Age": [25, 30, 35, None],
        "Score": [90, 85, 88, 90]
    })

    # Load pipeline config
    pipelines = load_pipeline_config()

    # Select a pipeline
    selected_pipeline = pipelines.get("basic_cleaning")

    if selected_pipeline:
        logger.info(f"Running pipeline: {selected_pipeline['description']}")
        pipeline = DataPipeline(df)
        processed_df = pipeline.apply_pipeline(selected_pipeline["steps"])
        logger.info(f"Pipeline metadata: {pipeline.get_metadata()}")
        print(processed_df)'''