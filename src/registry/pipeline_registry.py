import json
from src.registry import process_registry
from src.registry import model_registry
from src.core.logger import logger

class PipelineRegistry:
    def __init__(self):
        self.registry = {}

    def register_pipeline(self, name, config_path):
        if name in self.registry:
            logger.warning(f"Pipeline {name} is already registered. Overwriting.")
        self.registry[name] = config_path
        logger.info(f"Pipeline {name} registered successfully.")

    def get_pipeline(self, name):
        if name not in self.registry:
            logger.error(f"Pipeline {name} not found in the registry.")
            raise ValueError(f"Pipeline {name} not found.")
        return self.registry[name]

    def list_pipelines(self):
        return list(self.registry.keys())

pipeline_registry = PipelineRegistry()

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
    
        try:
            nested_pipeline_path = pipeline_registry.get_pipeline(nested_pipeline_name)
            nested_executor = PipelineExecutor(nested_pipeline_path, self.max_iterations)
            nested_executor.execute(df)
        except ValueError as e:
            logger.error(e)
