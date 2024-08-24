import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


data = pd.read_csv('customer_data.csv')


data = data.dropna()


label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])


scaler = StandardScaler()
numeric_features = ['Customer_Age', 'Balance', 'Num_of_Products']  
data[numeric_features] = scaler.fit_transform(data[numeric_features])


X = data.drop('Churn', axis=1)
y = data['Churn']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)


rf = RandomForestClassifier()
rf.fit(X_train, y_train)


gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)


joblib.dump(log_reg, 'log_reg_model.pkl')
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(gb, 'gb_model.pkl')

print("Models trained and saved successfully.")
