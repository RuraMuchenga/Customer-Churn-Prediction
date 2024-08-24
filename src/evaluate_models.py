import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('customer_data.csv')

# Load models
log_reg = joblib.load('log_reg_model.pkl')
rf = joblib.load('rf_model.pkl')
gb = joblib.load('gb_model.pkl')

# Preprocessing steps
label_encoder = LabelEncoder()
scaler = StandardScaler()

# Encode categorical variables
data['Gender'] = label_encoder.fit_transform(data['Gender'])  # Ensure consistent encoding

# Feature scaling
numeric_features = ['Customer_Age', 'Balance', 'Num_of_Products']  # Example numerical features
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Split data into features and target
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate Logistic Regression
log_reg_pred = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_reg_pred))
print("Logistic Regression Classification Report:")
print(classification_report(y_test, log_reg_pred))

# Evaluate Random Forest
rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# Evaluate Gradient Boosting
gb_pred = gb.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_pred))
print("Gradient Boosting Classification Report:")
print(classification_report(y_test, gb_pred))


conf_matrix = confusion_matrix(y_test, gb_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


if hasattr(gb, 'feature_importances_'):
    feature_importance = pd.Series(gb.feature_importances_, index=X.columns)
    feature_importance.sort_values(ascending=False).plot(kind='bar', title='Feature Importance')
    plt.show()
else:
    print("Gradient Boosting model does not have feature_importances_ attribute.")

