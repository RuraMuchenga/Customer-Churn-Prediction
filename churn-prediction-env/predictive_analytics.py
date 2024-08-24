import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
data = pd.read_csv('customer_data.csv')

# Load the model
gb = joblib.load('gb_model.pkl')

# Preprocessing steps
label_encoder = LabelEncoder()
scaler = StandardScaler()

# Encode categorical variables
data['Gender'] = label_encoder.fit_transform(data['Gender'])  # Ensure consistent encoding

# Feature scaling
numeric_features = ['Customer_Age', 'Balance', 'Num_of_Products']  # Example numerical features
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Prepare features for prediction
X = data.drop('Churn', axis=1)

# Predict the probabilities of churn for each customer
churn_probabilities = gb.predict_proba(X)[:, 1]

# Consider customers with a churn probability above a certain threshold as 'at-risk'
threshold = 0.5
at_risk_customers = data[churn_probabilities > threshold]

# Show the first few at-risk customers
print(at_risk_customers.head())
