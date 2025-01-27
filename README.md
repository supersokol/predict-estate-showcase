# PredictEstateShowcase: Automating Real Estate Data Analysis and Forecasting

PredictEstateShowcase is an advanced, interactive real estate analytics and prediction platform designed to integrate multiple data sources and provide insightful visualizations and forecasts for the housing market in the United States. This repository includes tools for data collection, preprocessing, analysis, and visualization, leveraging state-of-the-art technologies for machine learning, geospatial analysis, and pipeline orchestration.

## Vision and Idea
The core idea behind PredictEstateShowcase is to enable seamless automation of data gathering from dynamic web sources in the real estate domain. Real estate trends and analytics rely heavily on timely, accurate, and diverse data, and this platform bridges the gap by automating the collection and processing of such data from trusted sources. 

Currently, the platform focuses on two primary sources of real estate data (datasets with real estate metrics collected through web scraping using BeautifulSoup, Requests, and Selenium.):

- **Zillow**: Data is extracted monthly in CSV format, including updated statistics on prices and a variety of other key metrics. With over 150 datasets available, Zillow serves as a cornerstone for comprehensive real estate analytics.

- **HUD User** (U.S. Department of Housing and Urban Development): The platform downloads and structures state-specific and region-specific reports and publications in PDF format. These hundreds of documents cover diverse topics, trends, and periods, adding depth and reliability to the analysis.

Additionally integration of more data sources coming soon! Starting with:

- **Nominatim (OpenStreetMap)**: Geocoding and reverse geocoding with detailed address metadata using API queries. 

- **Realtor.com**: Leading platform for real estate listings in the United States. It provides detailed information about properties for sale and rent.

## Features and Capabilities
The project demonstrates how modern technologies can be applied to a practical problem using real-world data. Its primary goal is to serve as a showcase for understanding and leveraging cutting-edge tools for data-intensive tasks. PredictEstateShowcase can be used as:

- A foundation for building and customizing user-specific solutions.
- An educational tool for exploring the application of modern technologies in the context of real estate analytics.
- A portfolio project to showcase technical expertise and familiarity with a variety of tools and technologies.


## Key components include:

1. **Automated Data Collection**:
   - The platform automatically collects and updates real estate data from dynamic sources like Zillow and HUD User.
   - Monthly updates ensure access to the latest statistics and reports.

2. **Interactive Data Exploration**: (WIP)
   - With built-in tools for Exploratory Data Analysis (EDA), users can explore and understand data trends quickly.
   - Data preprocessing pipelines are provided to clean and transform data for further use.

3. **Pipeline and Model Demonstrations**: (WIP)
   - Demonstrates how to build and execute data processing pipelines for real-world datasets.
   - Supports the development of predictive models and workflows.

4. **Data Processing and Analysis**: (WIP)
     - Data cleaning and transformation using Pandas.
     - Initial machine learning models implemented with Scikit-Learn.

5. **Documentation**:
   - Comprehensive guides created using MkDocs.
   - Configuration details expanded for easier deployment and usage.

6. **Logging and Testing**:
   - **Loguru**: Robust logging for all workflows.
   - **pytest**: Partial test coverage for modules, with plans for expansion.

7. **Accessible Interfaces**:
   - The project includes an interactive Streamlit-based user interface that allows users to interact with the data and pipelines intuitively.
   - APIs are available to programmatically access the core functionalities.

8. **Structured Logging and Monitoring**: (WIP)
   - The integration of Elasticsearch, Logstash, and Kibana (ELK Stack) demonstrates how to implement centralized logging and monitoring for applications.

9. **Orchestrated Automation**: (WIP)
   - Apache Airflow is used to automate data updates and pipeline executions, showcasing how workflows can be scheduled and managed efficiently.

10. **Containerized Deployment**: 
   - The entire project is containerized using Docker, with Kubernetes manifests prepared for scalable deployments.

11. **Dynamic Parsing and Web Integration**:
   - Added integration with Wikipedia for supplementary data sources.
   - Work in progress to improve parsing and scraping to make it more universal and flexible. (WIP)

By combining these components, PredictEstateShowcase provides a comprehensive demonstration of how to solve a real-world analytical problem using a combination of data engineering, machine learning, and software development techniques. It is designed not only to address practical use cases but also to serve as a learning resource for professionals and students alike.

## Technologies Used

### Core Libraries
- **Python**: Core language for development.
- **BeautifulSoup, Requests, Selenium**: Web scraping.
- **Pandas, Scikit-Learn**: Data manipulation and analysis.
- **matplotlib, seaborn, Plotly**: Visualization tools.

### Frameworks and Tools
- **Streamlit**: Interactive user interface and dashboards.
- **FastAPI**: Backend API development.
- **MkDocs**: Documentation management.
- **Loguru**: Advanced logging.

### Infrastructure and Orchestration
- **Docker**: Containerization.
- **Kubernetes**: Container control.
- **Airflow**: Orchestrating daily data workflows. (WIP)
- **ZenML**: Experimental pipeline orchestration. (planned integration)
- **AWS S3**: Cloud storage. (planned integration)

## Getting Started

### Prerequisites
1. **Python 3.8+**
2. **Docker** (optional for deployment)
3. **Kubernetes** (optional for deployment)

### Installation and local run

1. Clone the repository:
   ```bash
   git clone https://github.com/supersokol/predict-estate-showcase.git
   cd predict-estate-showcase
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file:
     ```
     MASTER_CONFIG_PATH=config/master_config.json
     DATA_PATH=data/
     DB_PATH=data/db_files/
     API_URL=http://127.0.0.1:8000
     ```

4. Run the tests with:
    ```bash
    pytest
    ```

5. Run full setup (Stramlit+FastAPI+Static MkDocs):
   ```bash
   python src/run_services.py 
   ```

6. Launch MkDocs documentation:
   ```bash
   mkdocs serve
   ```

7. Run the Streamlit Dashboard
Launch the user interface:
```bash
streamlit run src/interfaces/app.py
```

8. Access the API
Start the FastAPI server:
```bash
uvicorn src.api.entrypoint:app --reload
```
Visit `http://127.0.0.1:8000/docs` for API documentation.

9. Launch using Docker containers and Kubernetes.
Please see the [`README_KUBERNETES.md`](https://github.com/supersokol/blob/main/README_KUBERNETES.md) file for guidelines.

10. Pipeline Management and Execution
Create and run data pipelines (coming soon):


## Project Structure
```
PredictEstateShowcase/
├── config/                  # Configuration files
├── data/                    # Generated and processed data (created by the app)
├── docker/                  # Docker setup and container configurations
├── docs/                    # Documentation files (MkDocs markdown files)
├── site/                    # Static files for MkDocs-generated documentation
├── logs/                    # Log files (created by the app)
├── notebooks/               # Jupyter Notebooks for analysis and examples
├── tests/                   # Unit and integration tests
├── src/                     # Main source code
│   ├── data_analysis/       # Modules for data analysis and visualization
│   ├── api/                 # FastAPI implementation for the project
│   ├── core/                # Core utilities and foundational modules
│   │   ├── integrations/        # External API and service integrations
│   ├── interfaces/          # Interfaces for interacting with the user
│   │   ├── sections/        # Specific UI sections and components
│   ├── models/              # Machine learning and predictive models
│   ├── workflows/           # Workflow and orchestration logic
│   ├── registry/            # Registries for managing data, pipelines, and configurations
│   └── run_services.py      # Script to run core services
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Future Plans
- Complete HUD User data processing and integration.
- Expand pytest coverage for all modules.
- Implement advanced machine learning models with PyTorch.
- Add orchestration with Airflow.

## Contact
For questions or support, contact `supersokol777@gmail.com` or create an issue in the repository.

