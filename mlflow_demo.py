# Description: This script trains a simple Logistic Regression model on the Iris dataset and logs the model to MLflow.
# To run this script, you need to have an MLflow server running. You can start one with the following command:
# mlflow ui
# Then, you can run this script with the following command:
# python mlflow_demo.py


import mlflow  
from sklearn.linear_model import LogisticRegression  
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  

# Load dataset  
data = load_iris()  
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)  

# Start MLflow run  
with mlflow.start_run():  
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")

    # Log parameters  
    mlflow.log_param("model_type", "LogisticRegression")  
    mlflow.log_param("max_iter", 200)  # 200 

    # Train model  
    model = LogisticRegression(max_iter=100)  
    model.fit(X_train, y_train)  

    # Evaluate model  
    y_pred = model.predict(X_test)  
    accuracy = accuracy_score(y_test, y_pred)  

    print("Accuracy: ", accuracy)
    # Log metrics  
    mlflow.log_metric("accuracy", accuracy)  

    # Log model  
    mlflow.sklearn.log_model(model, "model")  

