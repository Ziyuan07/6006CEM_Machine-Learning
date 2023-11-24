import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Assuming you have the dataset file named 'cancer_prediction_dataset.csv'
df = pd.read_csv('cancer_prediction_dataset.csv')

# Display the first few rows of the dataset
df.head()

# Separate features (X) and target variable (y)
X = df[['Gender', 'Age', 'Smoking', 'Fatigue', 'Allergy']]
y = df['Cancer']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_logreg = logreg_model.predict(X_test)

# Evaluate the model
print("Logistic Regression Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Classification Report:\n", classification_report(y_test, y_pred_logreg, zero_division=1))
