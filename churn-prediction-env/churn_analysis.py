import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the synthetic dataset
data = pd.read_csv('customer_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Example: Splitting data into features and target
X = data.drop(['Churn', 'CustomerID'], axis=1)
y = data['Churn']

# Example: Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Example: Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Additional models like RandomForest and GradientBoosting can be added similarly
