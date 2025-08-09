# student_performance_prediction.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------

# Download this CSV from: https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
data = pd.read_csv("StudentsPerformance.csv")

# Display the first few rows
print("First 5 rows of the dataset:")
print(data.head())

# ------------------------------------------
# Step 2: Data Preprocessing
# ------------------------------------------

# Encode categorical columns
le = LabelEncoder()
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Create a target column: 'passed' if average score >= 50
data['average_score'] = data[['math score', 'reading score', 'writing score']].mean(axis=1)
data['passed'] = data['average_score'].apply(lambda x: 1 if x >= 50 else 0)

# Define features and target
X = data.drop(columns=['math score', 'reading score', 'writing score', 'average_score', 'passed'])
y = data['passed']

# ------------------------------------------
# Step 3: Train-Test Split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------
# Step 4: Train Models
# ------------------------------------------

# Decision Tree
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# ------------------------------------------
# Step 5: Evaluation
# ------------------------------------------

print("\n--- Decision Tree Accuracy ---")
print(accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

print("\n--- Random Forest Accuracy ---")
print(accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ------------------------------------------
# Step 6: Visualize Feature Importance
# ------------------------------------------

# Random Forest feature importance
importances = rf_model.feature_importances_
features = X.columns
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feat_df)
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.show()

# Decision Tree visualization
plt.figure(figsize=(14, 8))
plot_tree(dt_model, feature_names=X.columns, class_names=['Fail', 'Pass'], filled=True)
plt.title("Decision Tree Structure")
plt.show()
