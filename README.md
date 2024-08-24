# Customer Churn Prediction Project
By Ruramai Muchenga

This project involves predicting customer churn using machine learning techniques. It includes generating synthetic customer data, performing exploratory data analysis (EDA), preprocessing the data, training machine learning models, and making predictions. The goal is to identify key factors contributing to churn and recommend personalized retention strategies.

## Project Structure

- **`generate_customer_data.py`**: Generates synthetic customer data for the analysis.
- **`eda.py`**: Performs Exploratory Data Analysis to visualize and understand the dataset.
- **`preprocess_and_train.py`**: Preprocesses data and trains machine learning models (Logistic Regression, Random Forest, Gradient Boosting).
- **`evaluate_models.py`**: Evaluates the trained models using various metrics and visualizes the results.
- **`predictive_analytics.py`**: Uses the trained models to predict at-risk customers and suggests retention strategies.
- **`requirements.txt`**: Lists the Python packages required for this project.
- **`customer_data.csv`**: The dataset used for analysis and modeling.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/RuraMuchenga/Customer-Churn-Prediction.git

2. **Navigate into the project directory**:
     ```bash
     cd Customer-Churn-Prediction
3. **Create and activate a virtual environment**:
     ```bash
     python -m venv churn-prediction-env
    churn-prediction-env\Scripts\activate  # On Windows
    source churn-prediction-env/bin/activate  # On macOS/Linux

4. **Install the required packages**:
     ```bash
     pip install -r requirements.txt

## Usage
1. **Generate Customer Data**:
     ```bash
       python generate_customer_data.py
2. **Perform Exploratory Data Analysis (EDA)**:
     ```bash
       python eda.py
3. **Preprocess Data and Train Models**:
     ```bash
       python preprocess_and_train.py
4. **Evaluate Models**:
     ```bash
       python evaluate_models.py
5. **Predictive Analytics**:
      ```bash
        python predictive_analytics.py


## Requirements
- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib


**Install the required packages using**:
   pip install -r requirements.txt


## Contributing
Feel free to submit issues or pull requests. For any questions or suggestions, please contact ruramaimuchenga@gmail.com












