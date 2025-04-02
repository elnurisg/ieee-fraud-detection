# IEEE Fraud Detection Project

Welcome to the IEEE Fraud Detection Project! This project leverages machine learning techniques to detect fraudulent e-commerce transactions using real-world data from the IEEE-CIS Fraud Detection competition.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Files & Notebooks](#project-files--notebooks)
- [Installation](#installation)
- [Local Development](#local-development)
  - [Running the Backend](#running-the-backend)
  - [Running the Frontend](#running-the-frontend)
  - [Running Tests](#running-tests)
- [API Usage](#api-usage)
- [Dockerization & Deployment](#dockerization--deployment)
- [Testing](#testing)
- [Future Improvements](#future-improvements)

## Overview

This project implements a full-stack fraud detection solution using Python. It includes:

- **Data Preprocessing & Feature Engineering:**  
  Handling missing values, extracting time features, grouping email domains, processing address and distance information, and aggregating binary flags.

- **Machine Learning Models:**  
  Multiple models were developed and compared (XGBoost, LightGBM, Random Forest, and a Neural Network prototype) with boosting models (XGBoost/LightGBM) achieving strong AUC (up to 0.949) and balanced precision/recall performance.

- **API Backend:**  
  A FastAPI backend serves predictions. It processes incoming JSON data (transaction and identity tables), applies preprocessing and feature engineering, and returns a fraud probability.

- **Frontend:**  
  A simple Streamlit-based frontend allows users to input data and view predictions, demonstrating an end-to-end solution.

- **Dockerization & Deployment:**  
  The application is containerized using Docker and deployed on Google Cloud Run, making it available online:
  [https://fraud-detection-frontend-x2ugjgse3q-uc.a.run.app](https://fraud-detection-frontend-x2ugjgse3q-uc.a.run.app)

## Features

- **Robust Data Processing:**  
  Handles a variety of feature types (numerical, categorical, binary) and performs extensive feature engineering.
  
- **Modeling:**  
  Implements gradient boosting models (XGBoost and LightGBM) with competitive performance and an initial Random Forest baseline.
  
- **End-to-End Pipeline:**  
  From data ingestion to API-based inference, ensuring consistency across training and production.
  
- **Interactive Frontend:**  
  A Streamlit-based UI for demoing predictions interactively.
  
- **Production-Ready Deployment:**  
  Dockerized application deployed on Google Cloud Run.

## Project Files & Notebooks

The project is organized into several key files and folders to facilitate development and experimentation:

- **data_processing.py**  
  Responsible for loading, merging, and orchestrating data processing along with related functions.

- **feature_engineering.py**  
  Contains methods for encoding, transforming features, and other feature engineering techniques.

- **EDA.ipynb**  
  A Jupyter notebook for Exploratory Data Analysis (EDA) to analyze raw data before processing.

- **FeatureEngineering.ipynb**  
  Explores feature engineering in detail, including close-up analysis of features and the feature importance from applied models.

- **ModelDevelopment.ipynb**  
  Notebook for training various models and comparing their performances.

- **helpers.py**  
  A collection of helper functions used throughout the project.

- **models folder**  
  Contains model-related Python files and a `config.py` file to store model configurations.

## Installation

1. **Clone the Repository:**

  ```bash
  git clone https://github.com/elnurisg/ieee-fraud-detection.git
  ```
2. **Set Up Virtual Environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate 
```
3. **Install Dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Local Development

**Running the Backend:**

Navigate to the project root and run:
```bash
uvicorn app.backend.main:app --reload
```
The API will be available at http://localhost:8000.

**Running the Frontend:**

Navigate to the app/frontend directory and run:
```bash
cd app/frontend
streamlit run app.py
```
This opens a browser window with the Streamlit app.

**Running Tests:**

From the project root, run:
```bash
pytest
```

## API Usage

**Endpoints**

- **GET /**

Returns a welcome message.

- **GET /health**

Health check endpoint that returns the status of the API.

- **POST /predict**

Accepts a JSON payload with two keys: transaction_table and identity_table.

**Example payload:**
```bash
{
  "transaction_table": { ... },
  "identity_table": { ... }
}
```
**Response:**

Returns the predicted fraud probability.

## Dockerization & Deployment

**Dockerization:**

The project is containerized using a Dockerfile located at the root of the repository. To build and run locally:
```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```
**Deployment:**

The application is deployed on Google Cloud Run. Use the provided deploy.sh script to build, push, and deploy your container:
```bash
bash deploy.sh
```
## Testing

Unit tests are written using pytest and are located in the tests/ directory. They cover:

- Data processing and merging
- Feature engineering functions
- Helper utilities for model saving/loading and evaluation
- API endpoints using FastAPI’s TestClient

To run the tests, execute:
```bash
pytest
```

## Future Improvements

- **Model Tuning & Ensembling:**

  Further optimize hyperparameters, and possibly ensemble multiple models (e.g., stacking XGBoost and LightGBM).

- **Advanced Feature Engineering:**

  Explore additional feature interactions, frequency encoding, and domain-specific transformations.

- **Neural Network Models:**

  Experiment with MLPs or more advanced neural architectures for tabular data.

- **Enhanced Frontend:**

  Expand the Streamlit app with more interactive visualizations and a user-friendly interface.

- **CI/CD & Monitoring:**
  Implement CI/CD (e.g., with GitHub Actions) and integrate monitoring/logging for production readiness.
