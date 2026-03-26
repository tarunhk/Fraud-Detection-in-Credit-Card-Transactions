# 💳 Credit Card Fraud Detection System

## 📌 Project Description
The **Credit Card Fraud Detection System** is an end-to-end machine learning application designed to identify fraudulent transactions from financial data.

This system uses **XGBoost for classification**, **Isolation Forest for anomaly detection**, and **SHAP for explainable AI**. It also includes an interactive **Streamlit web application** for real-time predictions and visualization.

The project demonstrates the use of:
- Machine Learning Models
- Data Preprocessing
- Imbalanced Data Handling
- Model Evaluation Metrics
- Explainable AI (SHAP)
- Web Application Development

---

## 🎯 Objective
To build a fraud detection system that:
- Identifies fraudulent transactions accurately  
- Handles imbalanced datasets effectively  
- Provides explainable predictions  
- Visualizes model performance  
- Deploys as an interactive web application  

---

## 🛠 Technologies Used
- Python  
- Scikit-learn  
- XGBoost  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- SHAP  
- Streamlit  
- Git & GitHub  

---

## ✨ Features
- Fraud detection using XGBoost  
- Anomaly detection using Isolation Forest  
- Handles imbalanced data using SMOTE  
- ROC Curve and Confusion Matrix visualization  
- SHAP-based model explainability  
- Interactive Streamlit dashboard  
- Real-time transaction prediction  
- Clean and modular code structure  

---

## 📂 Project Structure
```bash
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
```

---

## ▶️ How to Run the Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 2: Train the Model
```bash
python3 src/train.py
```
### Step 3: Evaluate the Model
```bash
python3 src/evaluate.py
```
### Step 4: Generate Explainability
```bash
python3 src/explain.py
```
### Step 5: Run the Application
```bash
streamlit run app/app.py
```

---

## 🧠 Concepts Used

### 1️⃣ Machine Learning
Used XGBoost for classification of transactions.

### 2️⃣ Anomaly Detection
Used Isolation Forest to detect unusual patterns.

### 3️⃣ Imbalanced Data Handling
Used SMOTE to balance fraud and non-fraud data.

### 4️⃣ Model Evaluation
Used ROC Curve, Confusion Matrix, Precision, Recall, F1-score

### 5️⃣ Explainable AI
Used SHAP to interpret model predictions.

### 6️⃣ Web Application
Built an interactive UI using Streamlit.

---

## 📊 Sample Output
```
Prediction Result

🚨 Fraud Detected (Confidence: 0.92)

Model Insights:
- ROC Curve displayed
- Confusion Matrix displayed

Explainability:
- SHAP summary plot
- Feature importance graph
```

---

## 🎓 Learning Outcomes
- Understanding imbalanced datasets in ML
- Building classification and anomaly detection models
- Evaluating models using advanced metrics
- Applying Explainable AI techniques
- Developing and deploying ML applications
- Structuring real-world ML projects
