# 💳 Credit Card Fraud Detection System (End-to-End Machine Learning with Explainable AI)

## 📌 Project Description
The Credit Card Fraud Detection System is a production-style, end-to-end Machine Learning project designed to detect fraudulent financial transactions in highly imbalanced datasets. The system integrates supervised learning (XGBoost) with anomaly detection techniques (Isolation Forest) to improve detection accuracy and robustness. To enhance transparency and trust, Explainable AI (SHAP) is incorporated to interpret model predictions at both global and local levels. The entire pipeline is deployed using Streamlit, enabling real-time predictions and interactive visualization.

## 🎯 Objective
To design and implement a scalable fraud detection pipeline that accurately identifies fraudulent transactions, handles severe class imbalance, provides model interpretability, and delivers a user-friendly web interface for real-time inference.

## 🛠 Technologies Used
- Python  
- Scikit-learn  
- XGBoost  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- SHAP (Explainable AI)  
- Streamlit  
- Imbalanced-learn (SMOTE)  
- Joblib  
- Git & GitHub  

## ✨ Key Features
- End-to-end ML pipeline from data preprocessing to deployment  
- High-performance classification using XGBoost  
- Anomaly detection using Isolation Forest for outlier analysis  
- Advanced handling of class imbalance using SMOTE  
- Comprehensive evaluation using ROC-AUC, Precision, Recall, F1-score  
- Visualization with Confusion Matrix and ROC Curve  
- Model interpretability using SHAP (feature importance & impact)  
- Interactive Streamlit dashboard for real-time predictions  
- Modular and scalable code architecture  

## 📂 Project Structure
fraud-detection-ml/
│
├── data/                # Raw dataset  
├── models/              # Saved trained models (.pkl)  
├── outputs/             # Evaluation plots and results  
├── src/                 # Core ML pipeline  
│   ├── config.py  
│   ├── data_preprocessing.py  
│   ├── train.py  
│   ├── evaluate.py  
│   ├── explain.py  
│
├── app/                 # Streamlit application  
│   └── app.py  
│
├── requirements.txt  
├── README.md  
└── .gitignore  

## ▶️ How to Run the Project
Step 1: Install dependencies  
pip install -r requirements.txt  

Step 2: Train the model  
python src/train.py  

Step 3: Evaluate the model  
python src/evaluate.py  

Step 4: Generate explainability insights  
python src/explain.py  

Step 5: Launch the application  
streamlit run app/app.py  

## 🧠 Core Concepts Implemented
- Imbalanced Data Handling: Applied SMOTE to address skewed class distribution  
- Supervised Learning: Trained XGBoost classifier for fraud prediction  
- Anomaly Detection: Used Isolation Forest to detect unusual transaction patterns  
- Model Evaluation: Used ROC-AUC, Precision, Recall, F1-score, Confusion Matrix  
- Explainable AI: Used SHAP for interpreting feature importance and predictions  
- Deployment: Built and deployed an interactive dashboard using Streamlit  

## 📊 Sample Output
Prediction Result  
🚨 Fraud Detected (Confidence: 0.92)  

Model Insights:  
- ROC Curve visualization  
- Confusion Matrix visualization  

Explainability:  
- SHAP summary plot  
- Feature importance analysis  

## 📈 Results
- Achieved high ROC-AUC score (~1.0)  
- Strong recall for fraud detection (critical for minimizing false negatives)  
- Effective performance on highly imbalanced dataset  

## 🎓 Learning Outcomes
- Handling real-world imbalanced datasets  
- Building and evaluating advanced ML models  
- Combining supervised and unsupervised techniques  
- Applying Explainable AI for model transparency  
- Developing deployable ML applications  
- Structuring scalable and modular ML pipelines  
