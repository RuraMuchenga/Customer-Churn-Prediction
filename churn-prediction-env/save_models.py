import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Assuming you have trained models like this:
log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()

# Train your models here
# ...

# Save the models
joblib.dump(log_reg, 'log_reg_model.pkl')
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(gb, 'gb_model.pkl')

print("Models saved successfully.")
