# Personalized_Medicine_for_CVD_Management
Personalized Medicine for CAD Prediction using MBA and Machine Learning 
## Clinical Decision Suppport System for Coronary Artery Disease (CAD)

### Project Overview
This project illustrates a web-based clinical decision support system (CDSS) for coronary artery disease (CAD) risk assessment and phenotype stratification.

The prototype integrates machine learning (ML) with interpretable decision logic to support personalized CAD risk evaluation and management.

The CDSS was developed as part of an MSc project on Personalized Medicine (PM) for Cardiovascular Diseases (CVDs), combining:

1. Market Basket Analysis (MBA) for phenotype discovery.
2. Supervised Machine Learning for CAD risk prediction.
3. Explainable AI techniques for clinical interpretability.

### Project Objectives
The key objectives of the project are:
1. Identifying significant associations among CAD risk factors in African populations  using MBA.
2. Enhance CAD risk prediction by integrating MBA-derived features into supervised machine learning models to improve model interpretability.
3. Predict individual CAD risk probability.
4.  Classify patients into clinically meaningful CAD phenotypes.
5.   rovide transparent explanations for model predictions.
6.   Design a prototype clinical decision support system (CDSS) to aid clinicians in delivering personalized treatment recommendations for CAD patients.



## Summary of Methodology
1. Data Processing
* Clinical, demographic, lifestyle, and metabolic features
* Ordinal and nominal categorical encoding
* Robust handling of missing or incomplete patient data

2. Machine Learning
* Models evaluated using Stratified K-Fold Cross-Validation
Algorithms considered:
* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

Logistic Regression selected based on:

* Best cross-validated F1-score
* Interpretability and clinical transparency

3.  Explainability
* Model coefficients transformed into odds ratios
* Patient-level explanations showing key contributing factors
* Exclusion of imputed-only features from explanations

### CDSS Features

The CDSS interface allows users to input key clinical variables, including:

Age, Sex, Systolic Blood Pressure, Body Mass Index (BMI), Total Cholesterol,

Smoking Status, Diabetes Status, Hypertension Status, Chest Pain Type, Family History of CAD.

### Outputs Provided

The output of the system include:
* CAD Risk Probability
* Risk Category (Low / High CAD Risk)
* CAD Phenotype Classification
* 5 top Contributing Clinical Factors


### Deployment

The CDSS is deployed as a Streamlit web application, making it accessible remotely without local installation.

* Framework: Streamlit

* Hosting: Streamlit Community Cloud

* Public URL: (pmmbamlcadsamuelcyril.streamlit.app)

### Project Structure
cad-cdss/
│
├── app.py                  # Streamlit application interface
├── cdss.py                 # CDSS logic and explainability functions
├── requirements.txt        # Python dependencies
├── model/
│   └── cdss_model.pkl      # Trained ML pipeline
└── README.md               # Project documentation



### Running it locally: 
To run the project locally on your PC, write the code below to install the requirements
pip install -r requirements.txt


### To run the app

streamlit run app.py

### Disclaimer:
The application is for educational and research purpose. It does not replace health professionals in medical context, CAD diagnosis and treatment.


### Academic Context

This CDSS was developed as part of an MSc dissertation focused on:

Personalized Medicine for Cardiovascular Disease integrating Market Basket Analysis and Machine Learning.

The system emphasizes:

* Clinical relevance
* Explainability and 
* Practical deployability in real-world healthcare settings

### Author
### Ekpe, Samuel Cyril
### M.Sc Candidate 
### Department of Computer Science
### University of Nigeria Nsukka
### Supervisior: Professor Colins Udeanor




