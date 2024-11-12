# Churn_prediction_DS_project_2
Customer Churn Prediction
This project predicts customer churn based on a dataset from a telecommunications company. By analyzing customer data, this model aims to help businesses identify customers likely to leave and take proactive measures to improve retention.

Project Overview
Customer churn is a critical issue for businesses as retaining existing customers is often more cost-effective than acquiring new ones. This project uses a machine learning model, specifically a Random Forest Classifier, to predict whether a customer will churn based on factors such as tenure, contract type, and monthly charges.

Dataset
The dataset used is Telco Customer Churn, which includes customer information such as demographics, account details, and service usage.

Project Structure
Data Preprocessing:
Handled missing values.
Encoded categorical variables.
Converted certain features to numeric values.
Modeling:
Used a Random Forest Classifier for training.
Split data into training and testing sets for evaluation.
Evaluation:
Evaluated the model using a confusion matrix, accuracy score, and classification report.
Visualized feature importance to understand the influence of each feature on the model.
Key Files
churn_prediction.ipynb: Jupyter Notebook containing the code for data loading, preprocessing, model training, and evaluation.
WA_Fn-UseC_-Telco-Customer-Churn.csv: The dataset file used in the project.
README.md: Project documentation.
Requirements
pandas
numpy
seaborn
matplotlib
scikit-learn
To install these libraries, you can run:

bash
Copy code
pip install -r requirements.txt
Results
The model achieved an accuracy of approximately 79%. Key insights include:

Confusion Matrix: Shows the distribution of true positives, true negatives, false positives, and false negatives.
Feature Importance: Highlights the most influential features in predicting churn, such as contract type and monthly charges.
Getting Started
To run this project locally:

Clone the repository.
Install the required libraries.
Run churn_prediction.ipynb to see the model training and evaluation steps.
