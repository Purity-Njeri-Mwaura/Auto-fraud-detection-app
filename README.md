🛡️ FraudShield — XAI-Driven Auto Insurance Fraud Detection
A machine learning system that detects fraudulent auto insurance claims and explains every prediction using SHAP and LIME, built for the Kenyan insurance market context.
Live App: https://auto-fraud-detection-app-cuxcpbub658hwappssrexpd.streamlit.app
Author: Purity Njeri Mwaura (SCT213-C002-0021/2022)
Supervisor: Mr. Daniel Njuguna
Institution: JKUAT Karen, BSc. Data Science and Analytics
________________________________________
📌 Problem Statement
Auto insurance fraud causes significant financial losses to insurers and drives up premiums for honest policyholders. Manual review processes are slow, inconsistent, and unable to scale with growing claim volumes.
This project develops an ensemble machine learning system that detects fraud accurately on imbalanced data while explaining every prediction at the feature level, making it trustworthy for real-world insurance investigation workflows.
________________________________________
📊 Dataset
•	Source: Zenodo (open-access research data repository)
•	Size: 10,000 auto insurance claim records, 40 original features
•	Target Variable: fraud_reported (Y/N → mapped to 1/0)
•	Class Distribution:
o	Legitimate: 75.3%
o	Fraud: 24.7%
•	After preprocessing: 145 features following one-hot encoding, feature engineering, and removal of leakage columns
________________________________________
🤖 Models Compared
Four ensemble classifiers were trained and evaluated under identical conditions using 5-fold Stratified Cross-Validation, with SMOTE applied only within training folds.
Model	CV Recall	CV F1-Score	CV ROC-AUC
Random Forest	0.8785	0.6838	0.9123
XGBoost	0.9168	0.7301	0.9603
LightGBM (Best)	0.9483	0.7909	0.9805
CatBoost	0.9438	0.7798	0.9767
Final Model Selection — LightGBM
LightGBM was selected as the final model because of:
•	Superior performance across evaluation metrics
•	Leaf-wise tree growth architecture for capturing non-linear fraud patterns
•	Gradient-based One-Side Sampling (GOSS) for preserving difficult fraud cases
•	Exclusive Feature Bundling (EFB) for efficient handling of the 145-feature encoded dataset
________________________________________
🧠 Explainability — SHAP and LIME
The system uses two complementary explainability methods to ensure every prediction remains transparent.
SHAP (Global Explainability)
Identifies which features matter most across all claims.
LIME (Local Explainability)
Explains why a specific claim was classified as fraudulent or legitimate.
Top SHAP Fraud Predictors
1.	incident_severity_Minor Damage
2.	incident_severity_Total Loss
3.	policy_year
4.	insured_hobbies_chess
5.	incident_year
A cross-method consistency analysis confirmed strong agreement between SHAP and LIME on influential features, supporting explanation reliability.
________________________________________
⚖️ Handling Class Imbalance — SMOTE
SMOTE (Synthetic Minority Oversampling Technique) was applied only to training folds and never to validation or testing data.
Metric	Without SMOTE	With SMOTE
Accuracy	0.96	0.88
Recall (Fraud)	0.85	0.95
Precision (Fraud)	0.97	0.67
F1-Score (Fraud)	0.91	0.78
SMOTE substantially improved fraud recall, detecting more fraudulent claims at the cost of additional false positives.
This trade-off was considered acceptable because missed fraud carries a higher operational cost than additional manual review.
________________________________________
🖥️ FraudShield Web Application
Built with Streamlit and deployed using Streamlit Cloud + GitHub.
Features
•	Full 34-field insurance claim input form
•	Real-time fraud probability scoring
•	SHAP explanation tab — top contributing features
•	LIME explanation tab — local prediction explanations
•	SHAP vs LIME comparison with agreement scoring
•	Dynamic plain-English explanation generation
•	Claim summary export to CSV
Screenshots
Add application screenshots here.
Example:
![Home Screen](screenshots/home.png)
![Prediction Results](screenshots/results.png)
![SHAP Explanation](screenshots/shap.png)
________________________________________
📈 Power BI Dashboard
An interactive Power BI dashboard was developed to visualize fraud patterns and model performance.
Dashboard Pages
Page 1 — Fraud Overview
•	Fraud rate by severity
•	Collision type analysis
•	Hobby-based fraud trends
•	Yearly fraud trends
•	Geographic patterns by state
•	Interactive slicers
![Fraud Overview](power-bi-dashboard/dashboard_page1_overview.png)
Page 2 — Model Performance
•	Model comparison
•	ROC-AUC ranking
•	SMOTE impact analysis
![Model Performance](power-bi-dashboard/dashboard_page2_models.png)
________________________________________
🛠️ Tech Stack
Category	Tools
Language	Python 3.10
ML Libraries	scikit-learn, XGBoost, LightGBM, CatBoost
Imbalance Handling	imbalanced-learn (SMOTE)
Explainability	SHAP, LIME
Web Application	Streamlit
Visualization	Matplotlib, Power BI
Deployment	Streamlit Cloud, GitHub
________________________________________
🚀 Running Locally
git clone <your-repository-url>
cd <repository-folder>
pip install -r requirements.txt
streamlit run app.py
________________________________________
📁 Repository Structure
FraudShield/
│
├── app.py
├── requirements.txt
├── fraud_lgbm_model.pkl
├── model_features.pkl
│
├── notebook/
│   └── myfinal_project_auto_insurance.ipynb
│
├── power-bi-dashboard/
│   ├── FraudShield_Dashboard.pbix
│   ├── dashboard_page1_overview.png
│   └── dashboard_page2_models.png
│
└── README.md
________________________________________
📚 Final Year Project
Submitted in partial fulfilment of the requirements for the award of the Bachelor of Science in Data Science and Analytics at Jomo Kenyatta University of Agriculture and Technology (JKUAT).

