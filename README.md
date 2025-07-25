# Ehanced_Fraud_Detection_For_Ecommerce_Banking

Project Overview
This repository contains the solution for the 10 Academy Week 8 & 9 Challenge, completed by Michael Zewdu Lemma as part of the KAIM 5/6 Cohort. The project focuses on building a robust fraud detection system for e-commerce and banking transactions using three datasets: Fraud_Data.csv, creditcard.csv, and IpAddress_to_Country.csv. The solution encompasses data preprocessing, exploratory data analysis (EDA), feature engineering, class imbalance handling, and machine learning model development. The final models, trained and evaluated using Logistic Regression, Decision Tree, and Random Forest, achieve high accuracy and reliable fraud detection, with Random Forest outperforming other models.

Project Objective
The goal of this project is to develop machine learning models to detect fraudulent transactions in e-commerce and banking datasets. The solution addresses data cleaning, feature engineering, class imbalance, and model evaluation using metrics such as accuracy, precision, recall, and F1-score. The project was completed in two phases: an Interim-1 report (submitted July 20, 2025) and a final submission (July 25, 2025).
Datasets
The project uses three datasets:

Fraud_Data.csv: Contains e-commerce transaction data with columns like user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address, and class (1 for fraud, 0 for legitimate). Approximately 1.2% of transactions are fraudulent.
creditcard.csv: Includes credit card transaction data with Time, V1–V28 (PCA-transformed features), Amount, and Class (1 for fraud, 0 for legitimate). Approximately 0.17% of transactions are fraudulent.
IpAddress_to_Country.csv: Maps IP addresses to countries with columns lower_bound_ip_address, upper_bound_ip_address, and country.

Installation and Setup
To run the code in this repository, follow these steps:

Clone the Repository:
[git clone https://github.com/Mzluci9/Ehanced_Fraud_Detection_Ecommerce_Banking 
cd fraud-detection-challenge


Install Dependencies:Ensure Python 3.8+ is installed. Install required packages using:
pip install -r requirements.txt


Requirements:The requirements.txt includes:

pandas
numpy
scikit-learn
imblearn
mlflow
joblib
scipy


Set Up MLflow:MLflow is used for experiment tracking. Ensure the tracking URI is set to file:///mlruns or configure a remote server.

Directory Setup:The code automatically creates directories for model storage (models/fraud_data and models/credit_card_data).


Data Preprocessing
The preprocessing pipeline ensures data quality and prepares datasets for modeling:

Fraud_Data.csv:

Checked for missing values (none found).
Removed duplicates using dp_fraud.remove_duplicates().
Converted signup_time and purchase_time to datetime64[ns].
Applied frequency encoding to categorical columns (device_id, source, browser, sex).
Downcasted numeric types to int32 or float32 for memory efficiency.
Used SimpleImputer (mean strategy) for precautionary missing value handling.
Converted features to sparse matrices for efficiency.


creditcard.csv:

Confirmed no missing values; applied mean imputation as a precaution.
Verified data types (Time, V1–V28, Amount as float64; Class as int64).
Normalized Time and Amount using StandardScaler.


IpAddress_to_Country.csv:

No missing values or duplicates.
Verified data types (lower_bound_ip_address as float64, upper_bound_ip_address as int64, country as object).



Exploratory Data Analysis (EDA)
EDA provided insights into data distributions and fraud patterns:

Univariate Analysis:

Fraud_Data.csv: purchase_value showed a right-skewed distribution (peak at 0, outliers above 4). age ranged from 20–40 (mean ~30). Fraud rate: 1.2%.
creditcard.csv: Amount was skewed (mean ~88.35, max 378.66). Fraud rate: 0.17%.


Bivariate Analysis:

Fraud_Data.csv: Fraudulent transactions peaked at purchase values of 20–40 (frequency 800), while legitimate transactions had a broader peak (5,000). The "Top 30 Countries by IP Address Count" plot showed the United States dominating (~40,000 IPs), followed by Canada, Russia, and Australia.
creditcard.csv: Higher Amount values (e.g., 378.66) were associated with fraud compared to non-fraud (e.g., 2.69).


Visualizations:

Histograms for purchase_value and Amount.
Bar plots for fraudulent vs. legitimate transaction distributions.
Bar plot for top 30 countries by IP address count.



Feature Engineering
Features were engineered to enhance model performance:

Fraud_Data.csv:

Time-Based Features: Extracted signup_hour, signup_day, signup_month, signup_year, purchase_hour, purchase_day, purchase_month, purchase_year, hour_of_day, and day_of_week.
Transaction Metrics: Computed time_since_signup (seconds between purchase_time and signup_time), transaction_count (per user_id), and transaction_velocity (purchase_value / time_since_signup).
Geolocation: Mapped IP addresses to countries using IpAddress_to_Country.csv.


creditcard.csv:

Normalized Time and Amount.
Retained V1–V28 as PCA features.


Encoding: Applied frequency encoding to categorical columns in Fraud_Data.csv to reduce dimensionality.


Class Imbalance Handling
Both datasets exhibited significant class imbalance:

Fraud_Data.csv: 1.2% fraud.
creditcard.csv: 0.17% fraud.

The Synthetic Minority Oversampling Technique (SMOTE) was applied to the training set using imblearn.over_sampling.SMOTE (random_state=42). Models were also configured with class_weight='balanced' to prioritize the minority class.
Model Training and Evaluation
Three models were trained and evaluated using scikit-learn and tracked with MLflow:

Logistic Regression: max_iter=1000, class_weight='balanced'.
Decision Tree: class_weight='balanced'.
Random Forest: n_estimators=100, class_weight='balanced'.

The datasets were split into 80% training and 20% testing sets (random_state=42). Models were evaluated using accuracy, precision, recall, and F1-score.
Results
Fraud_Data.csv

Logistic Regression:

Accuracy: 0.6478
Precision (Fraud): 0.17, Recall (Fraud): 0.70, F1-Score (Fraud): 0.27
High recall but low precision, indicating many false positives.


Decision Tree:

Accuracy: 0.9211
Precision (Fraud): 0.58, Recall (Fraud): 0.59, F1-Score (Fraud): 0.59
Balanced performance, significantly better than Logistic Regression.


Random Forest:

Accuracy: 0.9565
Precision (Fraud): 1.00, Recall (Fraud): 0.54, F1-Score (Fraud): 0.70
Best model, with perfect precision but moderate recall.



creditcard.csv

Logistic Regression:

Accuracy: 0.9767
Precision (Fraud): 0.06, Recall (Fraud): 0.89, F1-Score (Fraud): 0.11
High recall but very low precision; convergence issues noted.


Decision Tree:

Accuracy: 0.9989
Precision (Fraud): 0.66, Recall (Fraud): 0.63, F1-Score (Fraud): 0.64
Strong, balanced performance.


Random Forest:

Accuracy: 0.9995
Precision (Fraud): 0.97, Recall (Fraud): 0.71, F1-Score (Fraud): 0.82
Best model, with near-perfect accuracy and strong fraud detection.



Model Selection

Fraud_Data.csv: Random Forest was selected for its high accuracy (0.9565) and perfect precision (1.00).
creditcard.csv: Random Forest was chosen for its near-perfect accuracy (0.9995) and high F1-score (0.82).

Directory Structure
fraud-detection-challenge/
├── data/
│   ├── processed/
│   │   ├── cleaned_fraud_data_by_country.csv
│   │   ├── cleaned_credit_card_data.csv
├── models/
│   ├── fraud_data/
│   │   ├── logistic_regression.joblib
│   │   ├── decision_tree.joblib
│   │   ├── random_forest.joblib
│   ├── credit_card_data/
│   │   ├── logistic_regression.joblib
│   │   ├── decision_tree.joblib
│   │   ├── random_forest.joblib
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── eda.ipynb
│   ├── model_training.ipynb
├── mlruns/
├── requirements.txt
├── README.md

Usage

Preprocessing:Run preprocessing.ipynb to clean and prepare the datasets.

EDA:Execute eda.ipynb to generate visualizations and analyze data distributions.

Model Training:Run model_training.ipynb to train and evaluate models. Models are saved in the models/ directory, and experiment logs are stored in mlruns/.

Prediction:Load saved models (e.g., models/fraud_data/random_forest.joblib) using joblib.load() for inference on new data.


Challenges and Solutions

Class Imbalance: Addressed using SMOTE and class_weight='balanced' in models.
Convergence Issues: Logistic Regression on creditcard.csv faced convergence problems, suggesting future scaling improvements or alternative solvers.
Memory Efficiency: Managed by downcasting numeric types and using sparse matrices for Fraud_Data.csv.

Future Work

Improve recall for fraud detection, especially for Fraud_Data.csv Random Forest (recall: 0.54).
Explore advanced models like XGBoost or neural networks.
Address Logistic Regression convergence issues with better scaling or solver options.
Incorporate additional features, such as device fingerprinting or behavioral patterns.

