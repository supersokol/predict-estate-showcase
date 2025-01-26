# PredictEstateShowcase

PredictEstateShowcase is an advanced real estate analytics and prediction platform designed to integrate multiple data sources and provide insightful visualizations and forecasts for the housing market. This repository includes tools for data collection, preprocessing, analysis, and visualization, leveraging state-of-the-art technologies for machine learning, geospatial analysis, and pipeline orchestration.

## Features

### Data Sources
PredictEstateShowcase integrates multiple data sources:
- **Zillow**: Over 150 datasets with real estate metrics collected through web scraping using BeautifulSoup, Requests, and Selenium.
- **HUD User**: Links to hundreds of PDF reports, structured by region and state, from the U.S. Department of Housing and Urban Development.
- **Nominatim (OpenStreetMap)**: Geocoding and reverse geocoding with detailed address metadata using API queries.

### Core Components

#### Data Collection and Management
- **DataSourceRegistry**:
  - A fully refactored module to manage data sources.
  - Handles data and local storage operations efficiently.
  - Supports seamless integration with new sources.

- **PipelineRegistry and PipelineExecutor**:
  - Refactored for simplified pipeline management.
  - Enhanced logic for registering and executing data pipelines.

- **APIs**:
  - Unified `/data_sources` endpoint consolidates data-related operations for streamlined API access.

#### Data Processing and Analysis
- **Preprocessing**:
  - Data cleaning and transformation using Pandas.
  - Initial machine learning models implemented with Scikit-Learn.

- **Visualization**:
  - Interactive visualizations with matplotlib, seaborn, and Streamlit.

#### Documentation
- Comprehensive guides created using MkDocs.
- Configuration details expanded for easier deployment and usage.

#### Logging and Testing
- **Loguru**: Robust logging for all workflows.
- **pytest**: Partial test coverage for modules, with plans for expansion.

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
- **ZenML**: Experimental pipeline orchestration.
- **Airflow**: Orchestrating daily data workflows (planned integration).
- **AWS S3**: Cloud storage.

## Getting Started

### Prerequisites
1. **Python 3.8+**
2. **Docker** (optional for containerized deployment)
3. API keys for Zillow, HUD User, and Nominatim (if applicable)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PredictEstateShowcase.git
   cd PredictEstateShowcase
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file:
     ```
     DATABASE_URL=sqlite:///data/database.db
     ZILLOW_API_KEY=your_key_here
     NOMINATIM_API_URL=https://nominatim.openstreetmap.org
     ```

4. Run the tests with:
    ```bash
    pytest
    ```

5. Run full setup (Stramlit+FastAPI+Static MkDocs):
   ```bash
   python src/app/run_services.py 
   ```

### Usage

#### Run the Streamlit Dashboard
Launch the user interface:
```bash
streamlit run src/interfaces/app.py
```

#### Access the API
Start the FastAPI server:
```bash
uvicorn src.api.entrypoint:app --reload
```
Visit `http://127.0.0.1:8000/docs` for API documentation.

#### Data Pipeline Execution
Run data pipelines (coming soon):


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
│   ├── analysis/            # Modules for data analysis and visualization
│   ├── api/                 # FastAPI implementation for the project
│   ├── app/                 # Application entry points and higher-level logic
│   ├── core/                # Core utilities and foundational modules
│   ├── eda/                 # Exploratory Data Analysis scripts and functions
│   ├── integrations/        # External API and service integrations
│   ├── interfaces/          # Interfaces for interacting with the user
│   │   ├── sections/        # Specific UI sections and components
│   ├── metaheuristics/      # Algorithms and optimization routines
│   ├── models/              # Machine learning and predictive models
│   ├── processes/           # Workflow and orchestration logic
│   ├── registry/            # Registries for managing data, pipelines, and configurations
│   ├── sdk/                 # Software Development Kit for custom utilities
│   └── main.py              # Entry script for demonstration and testing
```

## Contributing
We welcome contributions! Please see the `CONTRIBUTING.md` file for guidelines.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Future Plans
- Complete HUD User data processing and integration.
- Expand pytest coverage for all modules.
- Implement advanced machine learning models with PyTorch.
- Add orchestration with Airflow.

## Contact
For questions or support, contact `your.email@example.com` or create an issue in the repository.

