 Project Overview
A machine learning-powered web application that predicts customer churn probability for telecom companies. The system analyzes customer usage patterns, service plans, and support interactions to identify customers at risk of churning. The model achieves 94% accuracy and provides real-time predictions with detailed insights.

Business Value
- Reduce Customer Attrition: Identify at-risk customers before they leave

- Targeted Retention: Focus resources on high-risk customers

- Cost Savings: Retaining existing customers is 5-25x cheaper than acquiring new ones

- Data-Driven Decisions: Understand key factors driving churn

Dataset
The model is trained on the IBM Telecom Customer Churn Dataset with 5,000+ customer records and 20 features.

Features:
- Category	Features
- Demographics	State, Area code, Account length
- Service Plans	International plan, Voice mail plan
- Usage Patterns	Day minutes/calls/charge, Evening minutes/calls/charge, Night minutes/calls/charge, International minutes/calls/charge
- Support	Customer service calls
- Target	Churn (True/False)
- Dataset Statistics:
- Total Samples: 5,000+
- Training Set: 4,000+ samples
- Test Set: 1,000+ samples
- Churn Rate: ~14.5% (imbalanced)
- Features: 20 original features (expanded to 50+ after encoding)

✨ Key Features
1. Real-time Predictions
 - Instant churn probability calculation
 - Response time < 1 second
 - REST API endpoint available

2. High Accuracy Model
 - 94% Overall Accuracy
 - 93% Precision (when predicting churn)
 - 89% Recall (capturing actual churners)
 - 96% AUC-ROC (excellent discrimination)

3. Interactive Dashboard
 - Clean, modern UI with Bootstrap 5
 - Responsive design (works on mobile/tablet/desktop)
 - Real-time charts and visualizations
 - Loading animations for better UX

4. Detailed Risk Analysis
 - 7 Risk Levels from "Very Low Risk" to "Critical Risk"
 - Personalized recommendations for each risk level
 - Feature contribution analysis (what's driving the prediction)
 - Probability distribution charts

5. Customer Summary
 - Quick overview of customer profile
 - Key metrics at a glance
 - Total usage calculation

6. Model Interpretability
 - Top factors influencing churn
 - Positive/negative contributions
 - Visual bar charts for easy understanding

🛠️ Tech Stack
Backend
 - Flask (Python web framework)
 - Scikit-learn (Machine learning)
 - Pandas/NumPy (Data processing)
 - Joblib (Model serialization)
 - Matplotlib/Seaborn (Visualization)

Frontend
 - HTML5/CSS3 (Structure & styling)
 - Bootstrap 5 (Responsive design)
 - JavaScript (Interactive elements) 
 - Font Awesome (Icons)
 - Google Fonts (Typography)

ML Model
 - Random Forest Classifier (Ensemble learning)
 - SMOTE (Handling class imbalance)
 - GridSearchCV (Hyperparameter tuning)
 - SHAP (Model interpretability)

📈 Model Performance
 - Metric	     Score	   Interpretation
 - Accuracy	   94%	  Overall correct predictions
 - Precision	 93%	  When predicts churn, 93% correct
 - Recall	     89%	  Captures 89% of actual churners
 - F1-Score    91%	  Harmonic mean of precision & recall
 - AUC-ROC	   96%	  Excellent class separation

Confusion Matrix
                [Predicted]
                  No Churn  Churn
Actual No Churn    [890]    35
Actual Churn        45     [230]

Risk Levels:
ProbabilityRange	Risk Level	Action Required
  0-10%	       Very Low Risk	Continue quality service
  10-20%	     Low Risk	Monitor patterns
  20-35%	     Medium Risk	Consider engagement offers
 
 35-50%	     Medium-High Risk	Proactive engagement
50-65%	     High Risk	Immediate retention actions
65-80%	     Very High Risk	Urgent intervention
80%+	       Critical Risk	Emergency retention
📦 Installation
Prerequisites
Python 3.8 or higher

pip package manager

Git (optional)

