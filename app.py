import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pickle

# Streamlit app title
st.title("Stroke Prediction App")

# Step 1: Upload dataset
st.header("Upload Dataset")
uploaded_file = st.file_uploader("Upload your dataset in CSV format", type="csv")

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset loaded successfully!")
    
    # Data overview
    st.header("Data Overview")
    st.write("First 10 rows of the dataset:")
    st.write(df.head(10))

    # Step 2: Preprocessing
    st.header("Data Preprocessing")
    st.write("Preparing data for model training...")

    # Identify and encode categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Select features and target column
    st.write("Select Features and Target Column:")
    features = st.multiselect("Select Features", df.columns.tolist(), default=df.columns.tolist()[:-1])
    target = st.selectbox("Select Target Column", df.columns.tolist(), index=len(df.columns) - 1)

    if not features or not target:
        st.warning("Please select at least one feature and a target column.")
    else:
        X = df[features]
        y = df[target]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Step 3: Model Selection
        st.header("Model Selection")
        model_type = st.selectbox("Choose a Model", ["Logistic Regression", "Decision Tree", "Random Forest"])

        if model_type == "Logistic Regression":
            st.write("Training Logistic Regression...")
            model = LogisticRegression(solver='liblinear')
            model.fit(X_train, y_train)

        elif model_type == "Decision Tree":
            max_depth = st.slider("Max Depth of Decision Tree", min_value=1, max_value=20, value=10)
            st.write("Training Decision Tree...")
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

        elif model_type == "Random Forest":
            n_estimators = st.slider("Number of Estimators", min_value=10, max_value=200, value=100)
            max_depth_rf = st.slider("Max Depth of Random Forest", min_value=1, max_value=20, value=10)
            st.write("Training Random Forest...")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth_rf, random_state=42)
            model.fit(X_train, y_train)

        # Evaluate model
        st.header("Model Performance")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Save the trained model
        model_file = f"{model_type.replace(' ', '_')}_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)

        st.success(f"Model saved as {model_file}")

        # Step 4: Prediction for New Data
        st.header("Predict Stroke Risk for New Data")
        user_input = {}
        for feature in features:
            user_input[feature] = st.number_input(f"Enter value for {feature}", value=float(df[feature].mean()))
        user_input_df = pd.DataFrame([user_input])

        # Standardize user input
        user_input_scaled = scaler.transform(user_input_df)

        # Predict stroke risk
        user_prediction = model.predict(user_input_scaled)
        st.write(f"Predicted Stroke Risk: {'Yes' if user_prediction[0] == 1 else 'No'}")
else:
    st.write("Please upload a dataset to proceed.")
