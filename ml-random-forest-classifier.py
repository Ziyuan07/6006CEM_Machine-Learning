import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Assuming you have the dataset file named 'cancer_prediction_dataset.csv'
df = pd.read_csv('cancer_prediction_dataset.csv')

# Display the first few rows of the dataset
df.head()

# Separate features (X) and target variable (y)
X = df[['Gender', 'Age', 'Smoking', 'Fatigue', 'Allergy']]
y = df['Cancer']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
print("\nRandom Forest Classifier Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))


# Define the parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV for Random Forest
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_grid=param_grid_rf,
                               scoring='accuracy',
                               cv=3,
                               n_jobs=-1)

grid_search_rf.fit(X_train, y_train)

# Best parameters from the grid search
best_params_rf = grid_search_rf.best_params_
print("\nBest Parameters for Random Forest:", best_params_rf)

# Evaluate the tuned Random Forest model
y_pred_rf_tuned = grid_search_rf.best_estimator_.predict(X_test)
print("\nRandom Forest Classifier Performance (Tuned):")
print("Accuracy:", accuracy_score(y_test, y_pred_rf_tuned))
print("Classification Report:\n", classification_report(y_test, y_pred_rf_tuned))
