import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('customer_data.csv')

# Visualize the distribution of the target variable
data['Churn'].value_counts().plot(kind='bar', title='Churn Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Explore relationships between features
sns.boxplot(x='Churn', y='Customer_Age', data=data)
plt.title('Churn by Customer Age')
plt.show()
