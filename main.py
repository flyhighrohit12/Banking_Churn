import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import io
import os
from scipy.stats import randint, uniform

# Data Handler functions
def load_data(file):
    """Load data from uploaded file"""
    return pd.read_csv(file)

def preprocess_data(data):
    """Preprocess the data"""
    # Drop unnecessary columns
    data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    # Encode categorical variables
    data = pd.get_dummies(data, drop_first=True)
    
    # Separate features and target
    X = data.drop('Exited', axis=1)
    y = data['Exited']
    
    # Scale features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y

# EDA functions
def perform_eda(data):
    """Perform Exploratory Data Analysis"""
    st.subheader("Exploratory Data Analysis")
    
    # Show sample data
    st.write("Sample Data:")
    st.write(data.head())
    
    # Show info
    st.write("Data Info:")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    # Show description
    st.write("Data Description:")
    st.write(data.describe())
    
    # Visualizations
    st.write("Visualizations:")
    
    # Histogram of numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    fig, ax = plt.subplots(figsize=(10, 6))
    data[numeric_columns].hist(ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Correlation heatmap for numeric columns only
    if len(numeric_columns) > 1:  # Check if there are at least 2 numeric columns
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data[numeric_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
        plt.title("Correlation Heatmap (Numeric Columns)")
        st.pyplot(fig)
    else:
        st.write("Not enough numeric columns for a correlation heatmap.")
    
    # Display counts for categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        st.write(f"Counts for {col}:")
        st.write(data[col].value_counts())
        
        # Bar plot for categorical columns
        fig, ax = plt.subplots(figsize=(10, 6))
        data[col].value_counts().plot(kind='bar', ax=ax)
        plt.title(f"Bar Plot for {col}")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)
# Model Trainer functions
def train_and_evaluate(X, y):
    """Train and evaluate models"""
    st.subheader("Training and Evaluation")
    
    # Model selection
    model_name = st.selectbox("Select algorithm", ["Decision Tree", "Random Forest", "SVM"])
    
    # Training data percentage
    train_size = st.slider("Select percentage of data for training", 50, 90, 80) / 100
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
    
    # Model training
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = SVC()
    
    model.fit(X_train, y_train)
    
    # Save model
    model_filename = f"{model_name.replace(' ', '')}.pkl"
    joblib.dump(model, model_filename)
    st.write(f"Model saved as {model_filename}")
    
    # Evaluation
    y_pred = model.predict(X_test)
    
    st.write("Performance Metrics:")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")
    st.write(f"F1-score: {f1_score(y_test, y_pred):.2f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    plt.title("Confusion Matrix")
    st.pyplot(fig)

# Predictor functions
def make_prediction(X):
    """Make predictions using trained model"""
    st.subheader("Predict")
    
    # Select algorithm
    model_name = st.selectbox("Select algorithm", ["Decision Tree", "Random Forest", "SVM"])
    
    model_filename = f"{model_name.replace(' ', '')}.pkl"
    
    if not os.path.exists(model_filename):
        st.error(f"Model file '{model_filename}' not found. Please train the model first.")
        return
    
    try:
        model = joblib.load(model_filename)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}. Please train the model again.")
        return
    
    # Input features
    feature_input = {}
    for feature in X.columns:
        if X[feature].dtype == 'float64':
            feature_input[feature] = st.slider(f"Select {feature}", float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))
        elif X[feature].dtype == 'int64':
            feature_input[feature] = st.number_input(f"Enter {feature}", min_value=int(X[feature].min()), max_value=int(X[feature].max()), value=int(X[feature].mean()))
        else:
            feature_input[feature] = st.selectbox(f"Select {feature}", X[feature].unique())
    
    # Make prediction
    input_df = pd.DataFrame([feature_input])
    
    if st.button("Predict"):
        try:
            prediction = model.predict(input_df)
            st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Hyperparameter Tuner functions
def tune_hyperparameters(X, y):
    """Perform hyperparameter tuning"""
    st.subheader("Hyperparameter Tuning")
    
    # Model selection
    model_name = st.selectbox("Select algorithm", ["Decision Tree", "Random Forest", "SVM"])
    
    # Hyperparameter ranges
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier()
        param_dist = {
            'max_depth': randint(1, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20)
        }
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
        param_dist = {
            'n_estimators': randint(10, 200),
            'max_depth': randint(1, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20)
        }
    else:
        model = SVC()
        param_dist = {
            'C': uniform(0.1, 10),
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': uniform(0.01, 1)
        }
    
    # Number of iterations
    n_iter = st.slider('Number of iterations', 10, 100, 50)
    
    # Perform RandomizedSearchCV
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=5, n_jobs=-1, random_state=42)
    random_search.fit(X, y)
    
    st.write("Best parameters:", random_search.best_params_)
    st.write("Best score:", random_search.best_score_)
    
    # Display all results
    results = pd.DataFrame(random_search.cv_results_)
    st.write("All results:")
    st.dataframe(results)
    
    # Plot parameter importances
    importances = pd.DataFrame({'feature': random_search.best_estimator_.feature_importances_}, 
                               index=X.columns)
    importances = importances.sort_values('feature', ascending=False)
    
    fig, ax = plt.subplots()
    importances.plot.bar(ax=ax)
    plt.title("Feature Importances")
    st.pyplot(fig)

# Cross Validator functions
def perform_cross_validation(X, y):
    """Perform k-fold cross-validation"""
    st.subheader("Cross Validation")
    
    # Model selection
    model_name = st.selectbox("Select algorithm", ["Decision Tree", "Random Forest", "SVM"])
    
    # Select k for k-fold
    k = st.slider("Select k for k-fold validation", 2, 10, 5)
    
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = SVC()
    
    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=k)
    
    st.write(f"Average accuracy for {k}-fold validation: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

# Main function
def main():
    st.title("Bank Customer Churn Prediction App")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load and preprocess data
        data = load_data(uploaded_file)
        X, y = preprocess_data(data)

        # Sidebar for navigation
        page = st.sidebar.selectbox("Choose a page", ["EDA", "Training and Evaluation", "Predict", "Hyperparameter Tuning", "Cross Validation"])

        if page == "EDA":
            perform_eda(data)
        elif page == "Training and Evaluation":
            train_and_evaluate(X, y)
        elif page == "Predict":
            make_prediction(X)
        elif page == "Hyperparameter Tuning":
            tune_hyperparameters(X, y)
        elif page == "Cross Validation":
            perform_cross_validation(X, y)

if __name__ == "__main__":
    main()


