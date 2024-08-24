import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load the dataset
data = pd.read_csv('customer_data.csv')

# Handle missing values (example)
data = data.dropna()

# Encode categorical variables
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Feature scaling
scaler = StandardScaler()
numeric_features = ['Customer_Age', 'Balance', 'Num_of_Products']  # Example numerical features
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Split data into features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Train Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Train Gradient Boosting
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# Save the models
joblib.dump(log_reg, 'log_reg_model.pkl')
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(gb, 'gb_model.pkl')

print("Models trained and saved successfully.")
