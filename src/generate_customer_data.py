import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define number of samples
num_samples = 1000

# Generate synthetic customer data
data = {
    'CustomerID': np.arange(1, num_samples + 1),
    'Gender': np.random.choice(['Male', 'Female'], num_samples),
    'Customer_Age': np.random.randint(18, 70, num_samples),
    'Balance': np.random.uniform(0, 100000, num_samples),
    'Num_of_Products': np.random.randint(1, 5, num_samples),
    'Is_Active': np.random.choice([0, 1], num_samples, p=[0.2, 0.8]),
    'Tenure': np.random.randint(1, 11, num_samples),
    'Has_Credit_Card': np.random.choice([0, 1], num_samples),
    'EstimatedSalary': np.random.uniform(30000, 150000, num_samples),
    'Churn': np.random.choice([0, 1], num_samples, p=[0.8, 0.2])  # 0: No churn, 1: Churn
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('customer_data.csv', index=False)

print("Synthetic customer data created and saved to 'customer_data.csv'.")
