import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import pickle

warnings.filterwarnings('ignore')

# Load Stroke Dataset
data = pd.read_csv("C:\Users\BRYAN\Documents\Final project\modified_healthcare_data.csv")

# Handle missing values
data['bmi'].fillna(data['bmi'].mean(), inplace=True)

# Encode categorical variables
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Split the data into input (X) and output (y)
X = data.drop(['id', 'stroke'], axis=1)  # Drop 'id' and target variable
y = data['stroke']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Title of the app
st.title("Stroke Prediction and Churn Analysis App")

# Data overview
st.header("Stroke Data Overview (First 10 Rows)")
st.write(data.head(10))

# Churn Dataset
st.header("Churn Dataset Analysis")
churn_data = pd.read_csv('/content/sample_data/Churn_Modelling.csv')

# Preprocess Churn Data
churn_data.info()
st.write("Null values:")
st.write(churn_data.isnull().sum())
st.write("Duplicate values:")
st.write(churn_data.duplicated().sum())

X_churn = churn_data.drop(['Exited'], axis=1)
y_churn = churn_data['Exited']
X_churn.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Encode categorical variables for Churn Dataset
X_churn['Gender'] = le.fit_transform(X_churn['Gender'])
ohe = OneHotEncoder()
encoder = ohe.fit_transform(X_churn[['Geography']])
encoder_arr = encoder.toarray()
df_geog = pd.DataFrame(encoder_arr, columns=ohe.get_feature_names_out(['Geography']), dtype='int')
X_churn = pd.concat([X_churn, df_geog], axis=1)
X_churn.drop(['Geography'], axis=1, inplace=True)

# Standardize numerical columns for Churn Dataset
num_cols = X_churn[['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'CreditScore']]
num_cols = scaler.fit_transform(num_cols)
num_df = pd.DataFrame(num_cols, columns=['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'CreditScore'])
X_churn_final = pd.concat([num_df, X_churn.drop(num_cols.columns, axis=1)], axis=1)

# Split Churn data into training and testing
X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(X_churn_final, y_churn, test_size=0.25, random_state=1)

# Model selection
st.header("Model Selection")
model_choice = st.selectbox("Select a Model", ("Linear Regression", "Random Forest", "Decision Tree", "Logistic Regression"))

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression()
}

selected_model = models[model_choice]
selected_model.fit(X_train, y_train)

# Evaluate Stroke Model
y_pred = selected_model.predict(X_test)
st.subheader("Stroke Model Evaluation")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Train and Evaluate Churn Models
lr = LogisticRegression()
lr.fit(X_train_churn, y_train_churn)
y_pred_churn = lr.predict(X_test_churn)
st.subheader("Churn Model Evaluation")
st.write(f"Logistic Regression Accuracy: {accuracy_score(y_test_churn, y_pred_churn):.2f}")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test_churn, y_pred_churn))
st.write(f"Precision: {precision_score(y_test_churn, y_pred_churn):.2f}")
st.write(f"Recall: {recall_score(y_test_churn, y_pred_churn):.2f}")
st.write(f"F1 Score: {f1_score(y_test_churn, y_pred_churn):.2f}")

# Hyperparameter Tuning for Random Forest
st.header("Hyperparameter Tuning")
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 5, 6, 7, 8, 9, 10],
    'max_features': ['auto', 'sqrt', 'log2'],
    'n_estimators': [100, 150, 200],
    'n_jobs': [-1]
}
grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=5, verbose=3, scoring='f1')
grid.fit(X_train_churn, y_train_churn)

best_params = grid.best_params_
st.write("Best Parameters for Random Forest:")
st.write(best_params)
st.write(f"Best F1 Score: {grid.best_score_:.2f}")

# Save the tuned model
rf_new = RandomForestClassifier(**best_params)
rf_new.fit(X_train_churn, y_train_churn)
with open('model.pkl', 'wb') as f:
    pickle.dump(rf_new, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

st.success("Model and scaler saved successfully.")
