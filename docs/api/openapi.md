```json
{
  "openapi": "3.1.0",
  "info": {
    "title": "Intelligent Automation API",
    "description": "\nThis API provides a flexible and extensible interface for managing and executing processes, pipelines, models, and metaheuristics.\nUse it to:\n- Automate data processing workflows.\n- Train and use machine learning models.\n- Execute high-level metaheuristics to dynamically generate new pipelines.\n- Manage data files and configurations.\n\nEach section below details the available endpoints and their functionality.\nSwagger combined with MkDocs for comprehensive API documentation:\n- [API Documentation](http://127.0.0.1:8000/docs)\n- [Detailed User Guide](http://127.0.0.1:8000/mkdocs/index.html)\n",
    "version": "1.0.0"
  },
  "paths": {
    "/healthcheck": {
      "get": {
        "summary": "Healthcheck",
        "operationId": "healthcheck_healthcheck_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {

                }
              }
            }
          }
        }
      }
    },
    "/processes": {
      "get": {
        "tags": [
          "Processes"
        ],
        "summary": "List all registered processes",
        "description": "Returns a list of all processes registered in the system.\nProcesses are reusable functions designed for data manipulation or analysis.\nEach process comes with detailed metadata, including a description, parameters, and expected outputs.",
        "operationId": "list_processes_processes_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {

                }
              }
            }
          }
        }
      }
    },
    "/processes/{process_name}": {
      "post": {
        "tags": [
          "Processes"
        ],
        "summary": "Execute a process",
        "description": "Execute a registered process by its name.\nPass parameters specific to the process as input.\nExample use cases:\n- Normalizing numeric columns.\n- Removing duplicates from a dataset.\n- Generating correlation matrices.",
        "operationId": "execute_process_processes__process_name__post",
        "parameters": [
          {
            "name": "process_name",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Process Name"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ProcessInput"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {

                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/pipelines": {
      "get": {
        "tags": [
          "Pipelines"
        ],
        "summary": "List all registered pipelines",
        "description": "Returns a list of all pipelines registered in the system.\nPipelines are predefined workflows consisting of sequential steps (processes, models, or nested pipelines).\nUse pipelines to automate repetitive tasks such as:\n- Data cleaning.\n- Feature engineering.\n- Complex multi-step workflows.",
        "operationId": "list_pipelines_pipelines_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {

                }
              }
            }
          }
        }
      }
    },
    "/pipelines/{pipeline_name}": {
      "post": {
        "tags": [
          "Pipelines"
        ],
        "summary": "Execute a pipeline",
        "description": "Execute a registered pipeline by its name.\nPass the dataset to be processed as input.\nPipelines dynamically execute their steps, which may include processes, models, and nested pipelines.",
        "operationId": "execute_pipeline_pipelines__pipeline_name__post",
        "parameters": [
          {
            "name": "pipeline_name",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Pipeline Name"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/PipelineInput"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {

                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/models": {
      "get": {
        "tags": [
          "Models"
        ],
        "summary": "List all registered models",
        "description": "Returns a list of all models registered in the system.\nModels can be used for:\n- Training: Fit a model to a given dataset.\n- Inference: Make predictions using a trained model.\nEach model comes with metadata that describes its type, parameters, and usage.",
        "operationId": "list_models_models_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {

                }
              }
            }
          }
        }
      }
    },
    "/models/{model_name}/train": {
      "post": {
        "tags": [
          "Models"
        ],
        "summary": "Train a model",
        "description": "Train a registered model using the provided dataset and parameters.\nExample use cases:\n- Train a linear regression model on numeric data.\n- Fit a logistic regression model for classification tasks.",
        "operationId": "train_model_models__model_name__train_post",
        "parameters": [
          {
            "name": "model_name",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Model Name"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ModelInput"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {

                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/metaheuristics": {
      "get": {
        "tags": [
          "Metaheuristics"
        ],
        "summary": "List all registered metaheuristics",
        "description": "Returns a list of all metaheuristics registered in the system.\nMetaheuristics are high-level strategies designed to dynamically:\n- Generate new pipelines based on data analysis.\n- Optimize workflows and suggest improvements.\nEach metaheuristic is described in detail with its usage and logic.",
        "operationId": "list_metaheuristics_metaheuristics_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {

                }
              }
            }
          }
        }
      }
    },
    "/metaheuristics/{heuristic_name}": {
      "post": {
        "tags": [
          "Metaheuristics"
        ],
        "summary": "Execute a metaheuristic",
        "description": "Execute a registered metaheuristic by its name.\nMetaheuristics can analyze data, generate new pipelines, and produce high-level insights.",
        "operationId": "execute_metaheuristic_metaheuristics__heuristic_name__post",
        "parameters": [
          {
            "name": "heuristic_name",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "title": "Heuristic Name"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "title": "Input Data"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {

                }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "HTTPValidationError": {
        "properties": {
          "detail": {
            "items": {
              "$ref": "#/components/schemas/ValidationError"
            },
            "type": "array",
            "title": "Detail"
          }
        },
        "type": "object",
        "title": "HTTPValidationError"
      },
      "ModelInput": {
        "properties": {
          "data": {
            "type": "object",
            "title": "Data"
          },
          "parameters": {
            "type": "object",
            "title": "Parameters"
          }
        },
        "type": "object",
        "required": [
          "data",
          "parameters"
        ],
        "title": "ModelInput"
      },
      "PipelineInput": {
        "properties": {
          "data": {
            "type": "object",
            "title": "Data"
          }
        },
        "type": "object",
        "required": [
          "data"
        ],
        "title": "PipelineInput"
      },
      "ProcessInput": {
        "properties": {
          "parameters": {
            "type": "object",
            "title": "Parameters"
          }
        },
        "type": "object",
        "required": [
          "parameters"
        ],
        "title": "ProcessInput"
      },
      "ValidationError": {
        "properties": {
          "loc": {
            "items": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "integer"
                }
              ]
            },
            "type": "array",
            "title": "Location"
          },
          "msg": {
            "type": "string",
            "title": "Message"
          },
          "type": {
            "type": "string",
            "title": "Error Type"
          }
        },
        "type": "object",
        "required": [
          "loc",
          "msg",
          "type"
        ],
        "title": "ValidationError"
      }
    }
  }
}
```