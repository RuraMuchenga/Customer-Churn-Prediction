import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder


data = pd.read_csv('customer_data.csv')


gb = joblib.load('gb_model.pkl')


label_encoder = LabelEncoder()
scaler = StandardScaler()


data['Gender'] = label_encoder.fit_transform(data['Gender'])  


numeric_features = ['Customer_Age', 'Balance', 'Num_of_Products'] 
data[numeric_features] = scaler.fit_transform(data[numeric_features])


X = data.drop('Churn', axis=1)


churn_probabilities = gb.predict_proba(X)[:, 1]


threshold = 0.5
at_risk_customers = data[churn_probabilities > threshold]


print(at_risk_customers.head())
